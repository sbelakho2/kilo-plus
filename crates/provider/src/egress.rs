//! Egress enforcement seam for outbound HTTP (audits 36-37).
//!
//! Every app-level outbound call must decide on a **parsed** destination
//! before the connection is attempted. [`CheckedHttpClient`] wraps a
//! `reqwest::Client` with an optional installed [`DestinationPolicy`]
//! (security crate): the decision runs against the `reqwest::Url` of the
//! *exact request object* that will be sent — scheme, host and port pulled
//! from the parsed URL, never from strings — and the connection goes to
//! that same request (`execute` sends the checked `reqwest::Request`
//! itself: there is no resolve-then-connect split, so a DNS rebinding that
//! changes what a hostname resolves to between decision and connect cannot
//! bypass the gate; the host rule already compared the URL's host text).
//!
//! Semantics (documented, identical to the security crate):
//!
//! | policy state                          | outcome |
//! |---------------------------------------|---------|
//! | `None` (no policy installed)          | allow (default-allow) |
//! | installed allowlist, rule matches     | allow |
//! | installed allowlist (even empty)      | deny before connect |
//!
//! A denied destination returns a typed [`EgressError::Denied`] carrying
//! which rule fired and how far its match got. The authoritative check runs
//! at `execute` time on the final request object (so no caller can
//! construct a request that bypasses the gate); `get`/`post` validate the
//! URL shape up front with typed errors, and [`CheckedHttpClient::check`]
//! exposes the same decision for callers that want to refuse early.
//!
//! Provider egress is config-derived (a configured `base_url` is trusted
//! config, not prompt-derived data); [`validate_provider_base_url`] is the
//! config-load validator (URL must parse, scheme http(s), host present, no
//! userinfo). As defense in depth, provider adapters that adopt
//! `CheckedHttpClient` get the same request-time gate on every send.
//!
//! Provider adapter egress (P0-36): every adapter executes HTTP through the
//! [`HttpTransport`] seam — a `reqwest::Client` exists ONLY inside this
//! module ([`PolicyCheckedHttpTransport`] is the production implementation;
//! [`MockHttpTransport`] is the no-network test seam). Adapters receive an
//! `Arc<dyn HttpTransport>` at construction and never construct or execute
//! a raw client themselves; request building / raw `Client::execute` happen
//! only in [`execute_get`]/[`execute_post_json`] here, so the request-time
//! destination gate applies to every adapter send identically to wave-11
//! semantics (parsed scheme/host/port, deny before connect).

use futures::future::BoxFuture;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use reqwest::{Body, Method, Request, RequestBuilder, Response, ResponseBuilderExt, Url};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::{ProviderError, ProviderErrorKind};
use faktor_security::destination::{Decision, DeniedReason, DestinationPolicy, RequestTarget};
use faktor_security::payload::{scan_payload, ScanOutcome, ScanPolicy};
use faktor_security::registry::SecretRegistry;

/// Outbound full-body secret-scan configuration (audit P0-37/P0-38).
/// `None` on a client/transport = no secret scanning (historical
/// behaviour). When installed, every request body is scanned — the whole
/// payload via the security crate's streaming scanner plus the exact
/// configured-secret registry, when one is attached — BEFORE the
/// connection is attempted:
///
/// | scan outcome                              | behaviour |
/// |-------------------------------------------|-----------|
/// | `Clean`                                   | send |
/// | `TooLargeForPolicy` (absolute cap set)    | [`EgressError::BodyTooLarge`], no bytes sent |
/// | hits + `block_on_secret: true`            | [`EgressError::SecretBlocked`], no bytes sent |
/// | hits + `block_on_secret: false`           | warn and send (deny-allowed-by-policy stays) |
/// | body is a stream (not materialized)       | [`EgressError::BodyNotMaterialized`] — the full payload cannot be inspected; fail closed |
#[derive(Debug, Clone)]
pub struct OutboundScanConfig {
    /// The scan policy: overlap window plus optional absolute payload cap
    /// (`max_payload_bytes: Some(n)` makes an oversized body a typed
    /// `BodyTooLarge` denial — never silently clean).
    pub policy: ScanPolicy,
    /// `true` (default): any detected secret hard-blocks the send with
    /// [`EgressError::SecretBlocked`]. `false`: warn and send.
    pub block_on_secret: bool,
    /// Exact configured-secret registry scanned over the same body
    /// (fingerprints only; the registry never stores or prints plaintext).
    pub registry: Option<Arc<SecretRegistry>>,
}

impl Default for OutboundScanConfig {
    fn default() -> Self {
        OutboundScanConfig {
            policy: ScanPolicy::default(),
            block_on_secret: true,
            registry: None,
        }
    }
}

/// How adapter-level failures map onto provider errors. A policy/builder
/// refusal (denied destination, unsupported scheme, unparseable URL, bad
/// request object) is a NON-retryable `BadRequest` — retrying a denied
/// destination can never succeed and must not hammer the runtime. Only a
/// genuine transport failure (`Transport`) is a retryable `Network` error.
impl From<EgressError> for ProviderError {
    fn from(e: EgressError) -> Self {
        let message = e.to_string();
        match e {
            EgressError::Transport(_) => ProviderError::new(ProviderErrorKind::Network, message),
            _ => ProviderError::new(ProviderErrorKind::BadRequest, message),
        }
    }
}

/// The single outbound-HTTP seam every adapter must execute through.
///
/// An implementation takes an already-built [`reqwest::Request`] and runs it
/// (usually through the parsed-destination gate first) and hands back the
/// response *with its body still streaming* — adapters then consume the
/// response body exactly as they consumed a `reqwest::Client::execute`
/// response before, so SSE/NDJSON streaming behavior is unchanged.
///
/// Implementations MUST be `Send + Sync` (providers are shared across
/// runtime tasks). `reqwest::Client` construction, request building and raw
/// `Client::execute` calls live ONLY inside this module; adapter code never
/// names a `reqwest::Client` (certified by the source-level test
/// `no_raw_client_execute_outside_egress`).
pub trait HttpTransport: Send + Sync {
    /// Execute one built request. The response body is a live stream; call
    ///ers consume it (never buffer it here).
    fn execute(&self, req: Request) -> BoxFuture<'_, Result<Response, EgressError>>;
}

/// The policy-checked production transport: [`CheckedHttpClient`] behind the
/// [`HttpTransport`] seam. `policy: None` = no destination policy installed
/// = default-allow (documented, wave-11 semantics); `Some(policy)` = the
/// allowlist governs every request on its parsed URL before any connect.
#[derive(Debug, Clone)]
pub struct PolicyCheckedHttpTransport {
    inner: CheckedHttpClient,
}

impl PolicyCheckedHttpTransport {
    /// Wrap an explicit client with an optional allowlist.
    pub fn new(inner: reqwest::Client, policy: Option<DestinationPolicy>) -> Self {
        Self {
            inner: CheckedHttpClient::new(inner, policy),
        }
    }

    /// The default production transport with the adapter-standard connect
    /// timeout and NO installed policy (default-allow). Compat path for
    /// callers that do not (yet) own a destination policy — the daemon
    /// config sites listed in the egress audit replace this with
    /// [`PolicyCheckedHttpTransport::with_policy`] once the sandbox
    /// `NetworkGate` policy is threaded into provider construction.
    pub fn permissive() -> Self {
        Self::new(default_timeout_client(), None)
    }

    /// A transport with the given allowlist installed (`None` keeps
    /// default-allow; `Some(DestinationPolicy::empty())` denies everything
    /// before connect).
    pub fn with_policy(policy: Option<DestinationPolicy>) -> Self {
        Self::new(default_timeout_client(), policy)
    }

    /// A transport with the given allowlist AND an installed outbound
    /// secret scan (audit P0-37/P0-38): every request body passes the
    /// full-payload scan before any connect (see [`OutboundScanConfig`]).
    pub fn with_policy_and_scan(
        policy: Option<DestinationPolicy>,
        outbound_scan: Option<OutboundScanConfig>,
    ) -> Self {
        Self {
            inner: CheckedHttpClient::with_policy_and_scan(policy, outbound_scan),
        }
    }

    /// The installed outbound secret scan (`None` = no secret scanning).
    pub fn outbound_scan(&self) -> Option<&OutboundScanConfig> {
        self.inner.outbound_scan()
    }

    /// The installed allowlist (`None` = default-allow).
    pub fn policy(&self) -> Option<&DestinationPolicy> {
        self.inner.policy()
    }
}

impl HttpTransport for PolicyCheckedHttpTransport {
    fn execute(&self, req: Request) -> BoxFuture<'_, Result<Response, EgressError>> {
        let inner = self.inner.clone();
        Box::pin(async move { inner.execute(req).await })
    }
}

/// `CheckedHttpClient` itself is a valid transport (it already encapsulates
/// policy + client and can yield streaming responses) — callers that hold
/// one can hand it to adapters directly.
impl HttpTransport for CheckedHttpClient {
    fn execute(&self, req: Request) -> BoxFuture<'_, Result<Response, EgressError>> {
        let inner = self.clone();
        Box::pin(async move { inner.execute(req).await })
    }
}

/// The adapter-standard client: connect-timeout only (the streaming hang
/// controls live in the adapter transport guards, never here).
fn default_timeout_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

/// Execute a GET through a transport. Request building and the raw
/// `Client::execute` call happen only here (adapter code never constructs a
/// `reqwest::Client`).
pub async fn execute_get(
    transport: &dyn HttpTransport,
    url: &str,
) -> Result<Response, EgressError> {
    let parsed = Url::parse(url).map_err(|e| EgressError::UnparseableUrl(e.to_string()))?;
    transport.execute(Request::new(Method::GET, parsed)).await
}

/// Execute a JSON POST through a transport (no extra headers).
pub async fn execute_post_json(
    transport: &dyn HttpTransport,
    url: &str,
    headers: HeaderMap,
    body: &serde_json::Value,
) -> Result<Response, EgressError> {
    execute_post_json_with_extras(transport, url, headers, &[], body).await
}

/// Execute a JSON POST through a transport with extra name/value headers
/// applied OVER `headers` (per-name replace, exactly the semantics the
/// gateway path relied on when it layered extra headers after the auth
/// headers). The content-type is forced to `application/json` last, exactly
/// like `RequestBuilder::json` did.
pub async fn execute_post_json_with_extras(
    transport: &dyn HttpTransport,
    url: &str,
    headers: HeaderMap,
    extra_headers: &[(String, String)],
    body: &serde_json::Value,
) -> Result<Response, EgressError> {
    let parsed = Url::parse(url).map_err(|e| EgressError::UnparseableUrl(e.to_string()))?;
    let mut request = Request::new(Method::POST, parsed);
    *request.headers_mut() = headers;
    for (name, value) in extra_headers {
        if let (Ok(k), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            request.headers_mut().insert(k, v);
        }
    }
    request
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let json = serde_json::to_string(body).map_err(|e| EgressError::Build(e.to_string()))?;
    *request.body_mut() = Some(Body::from(json));
    transport.execute(request).await
}

/// Test-seam transport: never touches the network. Returns a canned
/// response body (status + text) for every executed request and records the
/// executed (method, url) pairs, so an adapter's parser can be driven
/// without real HTTP. Optional deny mode returns a canned [`EgressError`]
/// before any response exists.
pub struct MockHttpTransport {
    status: reqwest::StatusCode,
    body: String,
    deny: Option<EgressError>,
    requests: std::sync::Mutex<Vec<(String, String)>>,
}

impl MockHttpTransport {
    /// A transport that answers every request with `status` + `body`.
    pub fn new(status: u16, body: impl Into<String>) -> Self {
        Self {
            status: reqwest::StatusCode::from_u16(status).unwrap_or(reqwest::StatusCode::OK),
            body: body.into(),
            deny: None,
            requests: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// A transport that refuses every request with `deny` (before any
    /// response could exist — exercises the adapters' error mapping with no
    /// HTTP involved).
    pub fn denying(deny: EgressError) -> Self {
        Self {
            status: reqwest::StatusCode::OK,
            body: String::new(),
            deny: Some(deny),
            requests: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// How many requests were executed through this transport.
    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    /// Executed (method, url) pairs in order.
    pub fn requests(&self) -> Vec<(String, String)> {
        self.requests.lock().unwrap().clone()
    }
}

impl HttpTransport for MockHttpTransport {
    fn execute(&self, req: Request) -> BoxFuture<'_, Result<Response, EgressError>> {
        self.requests
            .lock()
            .unwrap()
            .push((req.method().to_string(), req.url().to_string()));
        let status = self.status;
        let body = self.body.clone();
        let deny = self.deny.clone();
        let url = req.url().clone();
        Box::pin(async move {
            if let Some(err) = deny {
                return Err(err);
            }
            let response = http::Response::builder()
                .status(status)
                .url(url)
                .body(Body::from(body))
                .map_err(|e| EgressError::Build(e.to_string()))?;
            Ok(Response::from(response))
        })
    }
}

/// Why an outbound call refused to leave the process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EgressError {
    /// The installed allowlist denied the destination before any connect.
    /// `url` is the parsed request target (never a re-split string).
    Denied { url: String, reason: DeniedReason },
    /// The outbound secret scan found hits and `block_on_secret` is set.
    /// `kinds` carries only canonical kind labels (never snippets or
    /// candidate bytes), so logs cannot leak secret material.
    SecretBlocked { kinds: Vec<String> },
    /// The request body exceeded the scan policy's `max_payload_bytes`;
    /// the unscanned suffix must never be treated as clean (fail closed
    /// before any connect).
    BodyTooLarge { limit_bytes: u64 },
    /// Secret scanning is configured but the request body is a stream that
    /// is not fully materialized, so the FULL payload cannot be inspected.
    /// Fail closed: refused before any connect.
    BodyNotMaterialized(String),
    /// The URL scheme is not fetchable through this seam (only http/https
    /// are; a `ws`/`wss` policy may exist for websocket tools, but this
    /// client does not speak them).
    UnsupportedScheme(String),
    /// The URL (or provider base URL) does not parse / is not a valid
    /// absolute http(s) URL.
    UnparseableUrl(String),
    /// Building the request object failed.
    Build(String),
    /// The transport itself failed (connect/io); the request was allowed
    /// by the policy.
    Transport(String),
}

impl std::fmt::Display for EgressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EgressError::Denied { url, reason } => {
                write!(f, "egress to {url} denied: {reason}")
            }
            EgressError::SecretBlocked { kinds } => {
                write!(
                    f,
                    "egress denied before connect: secret scan blocked the request body ({})",
                    kinds.join(", ")
                )
            }
            EgressError::BodyTooLarge { limit_bytes } => {
                write!(
                    f,
                    "egress denied before connect: request body exceeds the secret-scan \
                     limit of {limit_bytes} bytes; the unscanned suffix would be sent \
                     unverified (raise the policy cap explicitly to allow)"
                )
            }
            EgressError::BodyNotMaterialized(url) => {
                write!(
                    f,
                    "egress denied before connect: cannot scan the full body of {url} \
                     (streamed bodies are not materialized); full-payload scanning \
                     requires a materialized body"
                )
            }
            EgressError::UnsupportedScheme(s) => {
                write!(
                    f,
                    "egress denied: unsupported scheme {s:?} (http/https only)"
                )
            }
            EgressError::UnparseableUrl(s) => write!(f, "not a parseable http(s) URL: {s}"),
            EgressError::Build(s) => write!(f, "request build failed: {s}"),
            EgressError::Transport(s) => write!(f, "transport error: {s}"),
        }
    }
}

impl std::error::Error for EgressError {}

/// The request-time destination gate, shared by the checked client and any
/// adapter that wants the defense-in-depth check before its own send.
/// `policy: None` means **no policy installed → default-allow**
/// (documented); `Some(policy)` means the allowlist governs (default-deny
/// on no full scheme+host+port match).
pub fn check_url(policy: Option<&DestinationPolicy>, url: &Url) -> Result<(), EgressError> {
    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(EgressError::UnsupportedScheme(scheme.to_string()));
    }
    let Some(host) = url.host_str() else {
        return Err(EgressError::UnparseableUrl(format!(
            "URL has no host: {url}"
        )));
    };
    if host.is_empty() {
        return Err(EgressError::UnparseableUrl(format!(
            "URL has an empty host: {url}"
        )));
    }
    // Pull scheme/host/port from the PARSED url — never from strings. When
    // the URL layer already parsed an IPv4 address, its octets beat any
    // textual spelling, so `127.0.0.01`-style variants compare canonical.
    // (The URL crate emits canonical dotted text for IPv4 hosts, so a
    // parse of the host text is an equivalent detector here.)
    let (is_ipv4, ip) = match host.parse::<std::net::Ipv4Addr>() {
        Ok(v4) => (true, Some(v4.octets())),
        Err(_) => (false, None),
    };
    let explicit_port = url.port();
    let target = RequestTarget::from_parts(Some(scheme), host, explicit_port, is_ipv4, ip)
        .map_err(EgressError::UnparseableUrl)?;
    match policy {
        None => Ok(()), // default-allow: no destination policy installed
        Some(policy) => match target.check_against(policy) {
            Decision::Allowed => Ok(()),
            Decision::Denied(reason) => Err(EgressError::Denied {
                url: target.describe(),
                reason,
            }),
        },
    }
}

/// `reqwest::Client` whose every send first passes the parsed destination
/// gate — and, when an [`OutboundScanConfig`] is installed, a full-body
/// secret scan of the FINAL request object (audit P0-37/P0-38). This is
/// the enforcement point a fetch/web tool (the genuinely user-prompt-
/// derived egress) must use for its outbound calls.
#[derive(Debug, Clone)]
pub struct CheckedHttpClient {
    inner: reqwest::Client,
    policy: Option<DestinationPolicy>,
    outbound_scan: Option<OutboundScanConfig>,
}

impl CheckedHttpClient {
    /// Wrap a client with an optional allowlist. `None` = no policy
    /// installed = default-allow (documented). No secret scanning.
    pub fn new(inner: reqwest::Client, policy: Option<DestinationPolicy>) -> CheckedHttpClient {
        CheckedHttpClient {
            inner,
            policy,
            outbound_scan: None,
        }
    }

    /// A default client with the given allowlist.
    pub fn with_policy(policy: Option<DestinationPolicy>) -> CheckedHttpClient {
        CheckedHttpClient::new(reqwest::Client::new(), policy)
    }

    /// A client with the given allowlist AND an installed outbound secret
    /// scan (see [`OutboundScanConfig`]).
    pub fn with_policy_and_scan(
        policy: Option<DestinationPolicy>,
        outbound_scan: Option<OutboundScanConfig>,
    ) -> CheckedHttpClient {
        let mut client = CheckedHttpClient::with_policy(policy);
        client.outbound_scan = outbound_scan;
        client
    }

    /// Install (or clear) the outbound secret scan on this client.
    pub fn set_outbound_scan(&mut self, outbound_scan: Option<OutboundScanConfig>) {
        self.outbound_scan = outbound_scan;
    }

    /// The installed outbound secret scan (`None` = no secret scanning).
    pub fn outbound_scan(&self) -> Option<&OutboundScanConfig> {
        self.outbound_scan.as_ref()
    }

    /// The installed allowlist (`None` = default-allow).
    pub fn policy(&self) -> Option<&DestinationPolicy> {
        self.policy.as_ref()
    }

    /// The typed pre-send check for one parsed request URL.
    pub fn check(&self, url: &Url) -> Result<(), EgressError> {
        check_url(self.policy.as_ref(), url)
    }

    /// Start a GET on a raw URL string. The URL is parsed strictly here
    /// (typed errors for non-http(s) schemes, missing hosts, userinfo,
    /// port 0); the authoritative destination-policy check runs at send
    /// time on the final request object ([`CheckedHttpClient::execute`]).
    pub fn get(&self, url: &str) -> Result<RequestBuilder, EgressError> {
        let parsed = parse_fetch_url(url)?;
        Ok(self.inner.get(parsed))
    }

    /// Start a POST on a raw URL string (see [`CheckedHttpClient::get`]).
    pub fn post(&self, url: &str) -> Result<RequestBuilder, EgressError> {
        let parsed = parse_fetch_url(url)?;
        Ok(self.inner.post(parsed))
    }

    /// Build the request and send it through the gate. The gate runs on the
    /// request's OWN parsed URL immediately before `Client::execute`, and
    /// that same request object is what gets sent (no resolve-then-connect
    /// split, no re-parsing of strings anywhere).
    pub async fn send_checked(&self, builder: RequestBuilder) -> Result<Response, EgressError> {
        let request = builder
            .build()
            .map_err(|e| EgressError::Build(e.to_string()))?;
        self.execute(request).await
    }

    /// Execute an already-built request through the gate. This is the
    /// single choke point: any `reqwest::Request` (however it was built)
    /// is checked against the policy on its parsed URL — and against the
    /// installed outbound secret scan on its FULL body — before `execute`.
    pub async fn execute(&self, request: Request) -> Result<Response, EgressError> {
        self.check(request.url())?;
        if let Some(cfg) = &self.outbound_scan {
            self.gate_outbound_body(&request, cfg)?;
        }
        let url = request.url().clone();
        let response = self
            .inner
            .execute(request)
            .await
            .map_err(|e| EgressError::Transport(format!("{url}: {e}")))?;
        Ok(response)
    }

    /// Full-payload secret gate on the FINAL request body (audit P0-37 /
    /// P0-38): generic pattern scan plus the exact configured-secret
    /// registry scan over every byte, decided BEFORE any connect. A
    /// streamed (non-materialized) body under a scan config is refused —
    /// the full payload cannot be inspected, so the send must not happen.
    fn gate_outbound_body(
        &self,
        request: &Request,
        cfg: &OutboundScanConfig,
    ) -> Result<(), EgressError> {
        let Some(body) = request.body() else {
            return Ok(()); // no body (e.g. GET): nothing to scan
        };
        let Some(bytes) = body.as_bytes() else {
            return Err(EgressError::BodyNotMaterialized(format!(
                "{}",
                request.url()
            )));
        };
        let outcome = scan_payload(bytes, &cfg.policy);
        let too_large = outcome == ScanOutcome::TooLargeForPolicy;
        let mut kinds: Vec<String> = match outcome {
            ScanOutcome::Found(hits) => {
                let mut seen = Vec::new();
                for kind in hits.into_iter().map(|h| h.kind) {
                    if !seen.contains(&kind) {
                        seen.push(kind);
                    }
                }
                seen
            }
            _ => Vec::new(),
        };
        if let Some(registry) = &cfg.registry {
            for hit in registry.scan_exact(bytes) {
                let kind = hit.kind;
                if !kinds.contains(&kind) {
                    kinds.push(kind);
                }
            }
        }
        if too_large {
            return Err(EgressError::BodyTooLarge {
                limit_bytes: cfg.policy.max_payload_bytes.unwrap_or(0),
            });
        }
        if !kinds.is_empty() {
            if cfg.block_on_secret {
                return Err(EgressError::SecretBlocked { kinds });
            }
            tracing::warn!(
                "outbound body contains a detected secret ({}); policy allows, sending",
                kinds.join(", ")
            );
        }
        Ok(())
    }

    /// The wrapped client (for callers that need other builder families).
    pub fn inner(&self) -> &reqwest::Client {
        &self.inner
    }
}

/// Parse one fetch URL strictly: absolute, http/https only, host present,
/// no userinfo, no port 0.
fn parse_fetch_url(raw: &str) -> Result<Url, EgressError> {
    let url = Url::parse(raw).map_err(|e| EgressError::UnparseableUrl(e.to_string()))?;
    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(EgressError::UnsupportedScheme(scheme.to_string()));
    }
    if url.host_str().is_none_or(str::is_empty) {
        return Err(EgressError::UnparseableUrl(format!(
            "URL has no host: {raw}"
        )));
    }
    if url.username() != "" || url.password().is_some() {
        return Err(EgressError::UnparseableUrl(format!(
            "URL must not carry userinfo: {raw}"
        )));
    }
    if url.port() == Some(0) {
        return Err(EgressError::UnparseableUrl(format!(
            "URL port 0 is invalid: {raw}"
        )));
    }
    Ok(url)
}

/// Config-load validation for provider base URLs (provider egress is
/// config-derived and trusted, but the URL must still parse and be
/// http(s)). Erroring at config load keeps a bad base URL from surfacing
/// as a confusing runtime failure.
pub fn validate_provider_base_url(raw: &str) -> Result<Url, EgressError> {
    parse_fetch_url(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{MockAction, MockServer};
    use faktor_security::destination::{DestinationPolicy, RuleMatch};
    use std::sync::Arc;

    fn policy_for(port: u16) -> DestinationPolicy {
        DestinationPolicy::parse_lines([&format!("http://127.0.0.1:{port}")]).unwrap()
    }

    /// The audit row set against a live mock server. `allowed_port` is the
    /// mock's real port; the denied rows use a neighbouring port, a
    /// different IP, https, and a rebinding-style hostname.
    async fn audit_rows(server: Arc<MockServer>, allowed_port: u16) {
        let wrong_port = if allowed_port == u16::MAX {
            1
        } else {
            allowed_port + 1
        };
        let client = CheckedHttpClient::with_policy(Some(policy_for(allowed_port)));
        let base = format!("http://127.0.0.1:{allowed_port}");

        // 1. Policy allows only http://127.0.0.1:<allowed>: allowed and
        //    reaches the mock exactly once.
        let resp = client
            .send_checked(client.get(&format!("{base}/probe")).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.text().await.unwrap(), "ok");
        assert_eq!(server.request_count(), 1);

        // 2. http://127.0.0.1:<wrong port>: port mismatch denied BEFORE
        //    connect (the mock never sees it); the reason names the rule
        //    and the depth (host matched, port missed).
        let err = client
            .send_checked(
                client
                    .get(&format!("http://127.0.0.1:{wrong_port}/probe"))
                    .unwrap(),
            )
            .await
            .unwrap_err();
        match &err {
            EgressError::Denied { reason, .. } => {
                assert_eq!(
                    reason.rule_fired.as_deref(),
                    Some(format!("http://127.0.0.1:{allowed_port}")).as_deref(),
                    "{err}"
                );
                assert_eq!(reason.matched, RuleMatch::Host);
            }
            other => panic!("expected Denied, got {other:?}"),
        }
        assert_eq!(server.request_count(), 1, "no connect on port mismatch");

        // 3. http://127.0.0.2:<allowed port>: host mismatch denied before
        //    connect.
        let err = client
            .send_checked(
                client
                    .get(&format!("http://127.0.0.2:{allowed_port}/probe"))
                    .unwrap(),
            )
            .await
            .unwrap_err();
        match &err {
            EgressError::Denied { reason, .. } => {
                assert_eq!(reason.rule_fired, None, "{err}");
                assert_eq!(reason.matched, RuleMatch::None);
            }
            other => panic!("expected Denied, got {other:?}"),
        }
        assert_eq!(server.request_count(), 1, "no connect on host mismatch");

        // 4. https://127.0.0.1:<allowed port> against an http-only rule:
        //    scheme mismatch denied before connect; the reason names the
        //    rule with host+port matched.
        let err = client
            .send_checked(
                client
                    .get(&format!("https://127.0.0.1:{allowed_port}/probe"))
                    .unwrap(),
            )
            .await
            .unwrap_err();
        match &err {
            EgressError::Denied { reason, .. } => {
                assert_eq!(
                    reason.rule_fired.as_deref(),
                    Some(format!("http://127.0.0.1:{allowed_port}")).as_deref()
                );
                assert_eq!(reason.matched, RuleMatch::HostAndPort);
            }
            other => panic!("expected Denied, got {other:?}"),
        }
        assert_eq!(server.request_count(), 1, "no connect on scheme mismatch");

        // 5. DNS-rebinding class: `localhost` resolves to the same address
        //    family as 127.0.0.1 on this host, but the rule names the IP
        //    literal; the gate compares the URL host TEXT, so the request
        //    is denied before connect. A rebinding attacker pointing an
        //    evil hostname at the allowed IP gains nothing, and because the
        //    decision and the connect share one URL object (execute sends
        //    the checked request), no second resolution can slip in.
        let err = client
            .send_checked(
                client
                    .get(&format!("http://localhost:{allowed_port}/probe"))
                    .unwrap(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, EgressError::Denied { .. }), "{err:?}");
        assert_eq!(
            server.request_count(),
            1,
            "no connect on host-text mismatch"
        );
    }

    /// Serializes the two tests that want the literal audit port 9911:
    /// parallel tokio tests would otherwise race bind() on the same port.
    static PORT_9911_GUARD: std::sync::OnceLock<tokio::sync::Mutex<()>> =
        std::sync::OnceLock::new();
    async fn port_9911_guard() -> tokio::sync::MutexGuard<'static, ()> {
        PORT_9911_GUARD
            .get_or_init(|| tokio::sync::Mutex::new(()))
            .lock()
            .await
    }

    #[tokio::test]
    async fn checked_client_denies_port_host_scheme_before_connect() {
        let _guard = port_9911_guard().await;
        let server = MockServer::new();
        server.route(
            "GET",
            "/probe",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        // Prefer the audit's literal port 9911 (with 9912 as the denied
        // mismatch port); fall back to an ephemeral port when 9911 is busy
        // (identical semantics, no CI flake).
        let (addr, _handle) = match server.serve_on(9911).await {
            Ok(pair) => pair,
            Err(_) => server.serve().await,
        };
        audit_rows(server, addr.port()).await;
    }

    #[tokio::test]
    async fn audit_rows_on_literal_9911_when_free() {
        // The audit's own numbers: policy allows only 127.0.0.1:9911;
        // 127.0.0.1:9912 (port mismatch) must be denied before connect.
        let _guard = port_9911_guard().await;
        let server = MockServer::new();
        server.route(
            "GET",
            "/probe",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let Ok((addr, _handle)) = server.serve_on(9911).await else {
            return; // port busy: covered by the ephemeral-port matrix
        };
        assert_eq!(addr.port(), 9911);
        audit_rows(server, 9911).await;
    }
    #[tokio::test]
    async fn default_allow_with_no_policy_and_deny_all_with_empty_policy() {
        let server = MockServer::new();
        server.route(
            "GET",
            "/probe",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let url = format!("http://{addr}/probe");

        // No policy installed => default-allow (documented).
        let open = CheckedHttpClient::with_policy(None);
        let resp = open.send_checked(open.get(&url).unwrap()).await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(server.request_count(), 1);

        // Installed empty policy => default-deny, before connect.
        let closed = CheckedHttpClient::with_policy(Some(DestinationPolicy::empty()));
        let err = closed
            .send_checked(closed.get(&url).unwrap())
            .await
            .unwrap_err();
        assert!(
            matches!(&err, EgressError::Denied { reason, .. }
                if reason.rule_fired.is_none()),
            "{err:?}"
        );
        assert_eq!(server.request_count(), 1, "deny before connect");
    }

    #[tokio::test]
    async fn execute_is_the_choke_point_even_for_hand_built_requests() {
        let server = MockServer::new();
        server.route(
            "GET",
            "/probe",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let policy = DestinationPolicy::parse_lines([&format!("http://{addr}")]).unwrap();
        let client = CheckedHttpClient::with_policy(Some(policy));

        // A request built directly (bypassing get()/post()) is still gated
        // at execute(): the URL object of the SENT request is what runs
        // through the policy.
        let allowed_url = Url::parse(&format!("http://{addr}/probe")).unwrap();
        let request = reqwest::Request::new(reqwest::Method::GET, allowed_url);
        let resp = client.execute(request).await.unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(server.request_count(), 1);

        // Wrong port through the raw-request path: denied before connect.
        let wrong_port = addr.port() + 1;
        let wrong = Url::parse(&format!(
            "http://{}/probe",
            std::net::SocketAddr::from(([127, 0, 0, 1], wrong_port))
        ))
        .unwrap();
        let request = reqwest::Request::new(reqwest::Method::GET, wrong);
        let err = client.execute(request).await.unwrap_err();
        assert!(matches!(err, EgressError::Denied { .. }), "{err:?}");
        assert_eq!(server.request_count(), 1);
    }

    #[tokio::test]
    async fn unsupported_schemes_are_rejected_at_parse_and_at_execute() {
        let client = CheckedHttpClient::with_policy(Some(DestinationPolicy::empty()));
        // get() rejects non-http(s) schemes at builder time.
        let err = client.get("ftp://example.com/file").unwrap_err();
        assert!(matches!(err, EgressError::UnsupportedScheme(_)), "{err:?}");
        let err = client.get("gopher://example.com/").unwrap_err();
        assert!(matches!(err, EgressError::UnsupportedScheme(_)), "{err:?}");
        let err = client.get("not-a-url").unwrap_err();
        assert!(matches!(err, EgressError::UnparseableUrl(_)), "{err:?}");
        // execute() also gates scheme: a hand-built ws:// request never
        // leaves (checked before connect).
        let ws = Url::parse("ws://example.com/").unwrap();
        let request = reqwest::Request::new(reqwest::Method::GET, ws);
        let err = client.execute(request).await.unwrap_err();
        assert!(matches!(err, EgressError::UnsupportedScheme(_)), "{err:?}");
    }

    #[test]
    fn validate_provider_base_url_enforces_config_load_contract() {
        for good in [
            "https://api.openai.com/v1",
            "http://127.0.0.1:9911",
            "https://api.deepseek.com",
        ] {
            let url = validate_provider_base_url(good).unwrap();
            assert!(url.host_str().is_some(), "{good}");
        }
        for bad in [
            "ftp://api.example.com",
            "api.example.com",               // relative: no scheme
            "https://",                      // no host
            "https://user:pass@example.com", // userinfo
            "not a url",
            "https://example.com:0",
            "https://example.com:99999",
        ] {
            let err = validate_provider_base_url(bad).unwrap_err();
            assert!(
                matches!(
                    err,
                    EgressError::UnparseableUrl(_) | EgressError::UnsupportedScheme(_)
                ),
                "{bad:?} => {err:?}"
            );
        }
    }

    #[test]
    fn checked_client_policy_is_visible_and_default_allow() {
        let p = DestinationPolicy::parse_lines(["https://example.com"]).unwrap();
        let c = CheckedHttpClient::with_policy(Some(p.clone()));
        assert_eq!(c.policy(), Some(&p));
        let c = CheckedHttpClient::with_policy(None);
        assert_eq!(c.policy(), None);
    }

    // ------------------------------------------------------------ transport

    fn policy_for_port(port: u16) -> DestinationPolicy {
        DestinationPolicy::parse_lines([&format!("http://127.0.0.1:{port}")]).unwrap()
    }

    async fn execute_once(
        transport: &dyn HttpTransport,
        url: &str,
        body: Option<&serde_json::Value>,
    ) -> Result<reqwest::Response, EgressError> {
        match body {
            Some(json) => {
                execute_post_json(transport, url, reqwest::header::HeaderMap::new(), json).await
            }
            None => execute_get(transport, url).await,
        }
    }

    #[tokio::test]
    async fn policy_checked_transport_denies_before_connect_and_records_no_bytes() {
        let server = MockServer::new();
        server.route(
            "GET",
            "/probe",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let allowed: Arc<dyn HttpTransport> = Arc::new(PolicyCheckedHttpTransport::with_policy(
            Some(policy_for_port(addr.port())),
        ));
        // Allowed: same parsed destination, streams normally.
        let resp = execute_once(
            allowed.as_ref(),
            &format!("http://127.0.0.1:{}/probe", addr.port()),
            None,
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.text().await.unwrap(), "ok");
        assert_eq!(server.request_count(), 1);

        // Denied (installed policy, wrong port): refused BEFORE any network
        // byte — the live mock never sees the request.
        let denied: Arc<dyn HttpTransport> = Arc::new(PolicyCheckedHttpTransport::with_policy(
            Some(policy_for_port(addr.port().wrapping_add(1))),
        ));
        let err = execute_once(
            denied.as_ref(),
            &format!("http://127.0.0.1:{}/probe", addr.port()),
            None,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, EgressError::Denied { .. }), "{err:?}");
        assert_eq!(server.request_count(), 1, "deny happened before connect");

        // Permissive (no policy installed) = default-allow (documented).
        let open: Arc<dyn HttpTransport> = Arc::new(PolicyCheckedHttpTransport::permissive());
        let resp = execute_once(
            open.as_ref(),
            &format!("http://127.0.0.1:{}/probe", addr.port()),
            None,
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(server.request_count(), 2);
    }

    #[tokio::test]
    async fn helpers_build_json_posts_with_extra_headers_and_streaming_body() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/json",
            MockAction::Respond {
                status: 200,
                body: "data: {\"ok\":true}\n\n".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let transport: Arc<dyn HttpTransport> = Arc::new(PolicyCheckedHttpTransport::permissive());
        let mut auth = reqwest::header::HeaderMap::new();
        auth.insert("authorization", "Bearer base".parse().unwrap());
        let json = serde_json::json!({"a": 1});
        let resp = execute_post_json_with_extras(
            transport.as_ref(),
            &format!("http://{addr}/json"),
            auth,
            &[("authorization".into(), "Bearer extra".into())],
            &json,
        )
        .await
        .unwrap();
        assert_eq!(resp.status(), 200);
        // The body must stream: text() reads it after execute returned.
        assert_eq!(resp.text().await.unwrap(), "data: {\"ok\":true}\n\n");
        let (_, path, body) = server.last_request().unwrap();
        assert_eq!(path, "/json");
        let sent: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(sent, json);
        let headers = server.last_request_headers();
        let get = |name: &str| {
            headers
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(
            get("authorization").as_deref(),
            Some("Bearer extra"),
            "extra headers replace same-named base headers"
        );
        assert_eq!(get("content-type").as_deref(), Some("application/json"));
    }

    #[test]
    fn mock_transport_serves_canned_bodies_and_deny_mode_without_http() {
        let transport = MockHttpTransport::new(200, "canned body");
        let url = reqwest::Url::parse("http://example.invalid/p").unwrap();
        let req = reqwest::Request::new(reqwest::Method::GET, url);
        let resp = futures::executor::block_on(transport.execute(req)).unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(
            futures::executor::block_on(resp.text()).unwrap(),
            "canned body"
        );
        assert_eq!(
            transport.requests(),
            vec![("GET".to_string(), "http://example.invalid/p".to_string())]
        );
        let transport = MockHttpTransport::denying(EgressError::Denied {
            url: "http://example.invalid".into(),
            reason: DeniedReason {
                rule_fired: None,
                matched: faktor_security::destination::RuleMatch::None,
            },
        });
        let url = reqwest::Url::parse("http://example.invalid/p").unwrap();
        let req = reqwest::Request::new(reqwest::Method::POST, url);
        let err = futures::executor::block_on(transport.execute(req)).unwrap_err();
        assert!(matches!(err, EgressError::Denied { .. }), "{err:?}");
        assert_eq!(transport.request_count(), 1);
    }

    #[test]
    fn egress_error_maps_to_provider_error_retryability() {
        let denied = EgressError::Denied {
            url: "http://127.0.0.1:1".into(),
            reason: DeniedReason {
                rule_fired: None,
                matched: faktor_security::destination::RuleMatch::None,
            },
        };
        let e: ProviderError = denied.into();
        assert!(!e.retryable, "a denied destination is never retried");
        assert!(e.message.contains("denied"), "{}", e.message);
        let transport: ProviderError = EgressError::Transport("connect refused".into()).into();
        assert_eq!(transport.kind, ProviderErrorKind::Network);
        assert!(transport.retryable);
    }

    #[test]
    fn checked_http_client_is_a_transport_and_reuses_the_execute_gate() {
        let client = CheckedHttpClient::with_policy(Some(DestinationPolicy::empty()));
        let transport: &dyn HttpTransport = &client;
        let url = reqwest::Url::parse("http://127.0.0.1:9/probe").unwrap();
        let req = reqwest::Request::new(reqwest::Method::GET, url);
        let err = futures::executor::block_on(transport.execute(req)).unwrap_err();
        assert!(
            matches!(err, EgressError::Denied { .. }),
            "the empty installed policy denies before connect: {err:?}"
        );
    }

    // ------------------------------------------------------- source scan

    /// The P0-36 static certification: no raw `reqwest` egress may exist
    /// outside this file. Scanning our own sources is acceptable here; the
    /// walk is bounded to the six adapter dirs + provider/src.
    #[test]
    fn no_raw_client_execute_outside_egress() {
        const MARKERS: [&str; 3] = [".execute(", "reqwest::Client::new", "Client::builder"];
        // Adjudicated exemptions. Empty by default; the ONLY entries are
        // the MockServer harness self-tests (provider/src/testing.rs)
        // which drive the mock with a raw client to prove the harness
        // works — they are the harness testing itself, never adapter
        // egress. Keyed by relative path + trimmed line so unrelated new
        // offenders are still listed loudly.
        const ALLOWLIST: &[(&str, &str)] = &[(
            "provider/src/testing.rs",
            "let client = reqwest::Client::new();",
        )];

        let crates_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/provider sits directly under crates/")
            .to_path_buf();
        let mut roots: Vec<std::path::PathBuf> = Vec::new();
        for dir in [
            "openai",
            "anthropic",
            "google",
            "deepseek",
            "gateway",
            "ollama",
        ] {
            roots.push(crates_root.join(dir).join("src"));
        }
        roots.push(crates_root.join("provider").join("src"));

        let mut offenders: Vec<String> = Vec::new();
        let mut scanned = 0usize;
        for root in roots {
            let mut stack = vec![root];
            while let Some(dir) = stack.pop() {
                let entries = match std::fs::read_dir(&dir) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                for entry in entries.flatten() {
                    let path = entry.path();
                    let file_type = match entry.file_type() {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    if file_type.is_dir() {
                        stack.push(path);
                        continue;
                    }
                    if file_type.is_file()
                        && path.extension().and_then(|e| e.to_str()) == Some("rs")
                    {
                        scanned += 1;
                        let rel = path
                            .strip_prefix(&crates_root)
                            .unwrap_or(&path)
                            .display()
                            .to_string();
                        if rel == "provider/src/egress.rs" {
                            continue; // the ONE allowed file
                        }
                        let Ok(source) = std::fs::read_to_string(&path) else {
                            continue;
                        };
                        for (idx, line) in source.lines().enumerate() {
                            let trimmed = line.trim();
                            if MARKERS.iter().any(|m| line.contains(m)) {
                                let allowlisted = ALLOWLIST
                                    .iter()
                                    .any(|(p, text)| p == &rel && *text == trimmed);
                                if !allowlisted {
                                    offenders.push(format!("{rel}:{}: {trimmed}", idx + 1));
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(scanned >= 7, "scan walked nothing: {scanned}");
        assert!(
            offenders.is_empty(),
            "raw reqwest egress outside crates/provider/src/egress.rs:\n  {}\n\
             Every adapter send must go through the HttpTransport seam in egress.rs.",
            offenders.join("\n  ")
        );
    }
}

#[cfg(test)]
mod outbound_scan_tests {
    use super::*;
    use crate::testing::{MockAction, MockServer};
    use faktor_security::payload::ScanPolicy;
    use faktor_security::registry::SecretRegistry;
    use std::sync::Arc;

    fn allowed_policy_for(port: u16) -> DestinationPolicy {
        DestinationPolicy::parse_lines([&format!("http://127.0.0.1:{port}")]).unwrap()
    }

    fn blocking_scan() -> OutboundScanConfig {
        OutboundScanConfig {
            policy: ScanPolicy::default(),
            block_on_secret: true,
            registry: None,
        }
    }

    #[tokio::test]
    async fn secret_body_is_denied_before_connect_zero_bytes_sent() {
        // Adversarial: the server route EXISTS and would answer; the mock
        // counter must stay 0 because the scan denies BEFORE any connect —
        // no byte of the body ever leaves the process.
        let server = MockServer::new();
        server.route(
            "POST",
            "/send",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let client = CheckedHttpClient::with_policy_and_scan(
            Some(allowed_policy_for(addr.port())),
            Some(blocking_scan()),
        );
        let body = serde_json::json!({
            "text": "please forward AKIA0123456789ABCDEF to the endpoint",
        });
        let err = client
            .send_checked(
                client
                    .post(&format!("http://{addr}/send"))
                    .unwrap()
                    .json(&body),
            )
            .await
            .unwrap_err();
        match &err {
            EgressError::SecretBlocked { kinds } => {
                assert_eq!(kinds, &vec!["aws_key".to_string()], "{err}");
                assert!(!err.to_string().contains("AKIA0123456789ABCDEF"));
            }
            other => panic!("expected SecretBlocked, got {other:?}"),
        }
        assert_eq!(
            server.request_count(),
            0,
            "no bytes sent for a blocked body: {err}"
        );
        // The clean twin of the same body IS sent (same policy allows it).
        let clean_body = serde_json::json!({ "text": "please forward nothing sensitive" });
        let resp = client
            .send_checked(
                client
                    .post(&format!("http://{addr}/send"))
                    .unwrap()
                    .json(&clean_body),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(server.request_count(), 1);
    }

    #[tokio::test]
    async fn configured_secret_registry_hits_deny_before_connect() {
        let mut registry = SecretRegistry::new();
        let token: Vec<u8> = (0..48u8).collect();
        registry.register(&token);
        let server = MockServer::new();
        server.route(
            "POST",
            "/send",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let client = CheckedHttpClient::with_policy_and_scan(
            Some(allowed_policy_for(addr.port())),
            Some(OutboundScanConfig {
                registry: Some(Arc::new(registry)),
                ..blocking_scan()
            }),
        );
        // Token at the END of the body: the exact scan must still see it.
        let mut body_bytes = b"prefix content ".to_vec();
        body_bytes.extend_from_slice(&token);
        let err = client
            .send_checked(
                client
                    .post(&format!("http://{addr}/send"))
                    .unwrap()
                    .body(body_bytes),
            )
            .await
            .unwrap_err();
        match &err {
            EgressError::SecretBlocked { kinds } => {
                assert!(kinds.iter().any(|k| k == "configured_secret"), "{err}");
            }
            other => panic!("expected SecretBlocked, got {other:?}"),
        }
        assert_eq!(
            server.request_count(),
            0,
            "registry hit denies before connect"
        );
        // The token value never appears in the error text.
        assert!(!err.to_string().contains("content"), "{err}");
        assert!(!format!("{err:?}").contains("48"));
    }

    #[tokio::test]
    async fn oversized_body_is_too_large_before_connect() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/send",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let client = CheckedHttpClient::with_policy_and_scan(
            Some(allowed_policy_for(addr.port())),
            Some(OutboundScanConfig {
                policy: ScanPolicy {
                    max_payload_bytes: Some(16),
                    ..ScanPolicy::default()
                },
                ..blocking_scan()
            }),
        );
        let body = "x".repeat(64);
        let err = client
            .send_checked(
                client
                    .post(&format!("http://{addr}/send"))
                    .unwrap()
                    .body(body),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(&err, EgressError::BodyTooLarge { limit_bytes: 16 }),
            "{err:?}"
        );
        assert_eq!(server.request_count(), 0, "too-large denies before connect");
    }

    #[tokio::test]
    async fn deny_allowed_by_policy_sends_and_warns() {
        // block_on_secret=false: Found still sends (deny-allowed-by-policy
        // behaviour stays) — the mock receives exactly one request.
        let server = MockServer::new();
        server.route(
            "POST",
            "/send",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let client = CheckedHttpClient::with_policy_and_scan(
            Some(allowed_policy_for(addr.port())),
            Some(OutboundScanConfig {
                block_on_secret: false,
                ..blocking_scan()
            }),
        );
        let resp = client
            .send_checked(
                client
                    .post(&format!("http://{addr}/send"))
                    .unwrap()
                    .body("sk-0123456789abcdefghijklmnopqrstuv"),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(server.request_count(), 1);
    }

    #[tokio::test]
    async fn streamed_body_under_scan_config_fails_closed_before_connect() {
        // A streamed (non-materialized) body cannot be scanned in full:
        // the send refuses with a typed error before any connect rather
        // than sending bytes that were never inspected.
        let server = MockServer::new();
        server.route(
            "POST",
            "/send",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let client = CheckedHttpClient::with_policy_and_scan(
            Some(allowed_policy_for(addr.port())),
            Some(blocking_scan()),
        );
        // The body holds an actual secret — but as an un-materialized
        // stream, so no full-payload scan is possible. Fail closed.
        let stream =
            futures::stream::iter(vec![Ok::<_, std::io::Error>(&b"AKIA0123456789ABCDEF"[..])]);
        let req = client
            .post(&format!("http://{addr}/send"))
            .unwrap()
            .body(reqwest::Body::wrap_stream(stream))
            .build()
            .unwrap();
        let err = client.execute(req).await.unwrap_err();
        assert!(
            matches!(&err, EgressError::BodyNotMaterialized(_)),
            "{err:?}"
        );
        assert_eq!(server.request_count(), 0, "unscannable body never connects");
        // No-body request under a scan config passes (nothing to scan).
        let resp = client
            .send_checked(client.get(&format!("http://{addr}/nobody")).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), 404, "unrouted path answers loudly");
        assert_eq!(server.request_count(), 1);
    }

    #[tokio::test]
    async fn transport_seam_applies_the_same_gate() {
        // PolicyCheckedHttpTransport (the wave-11 adapter seam) carries the
        // scan config into the same choke point.
        let server = MockServer::new();
        server.route(
            "POST",
            "/send",
            MockAction::Respond {
                status: 200,
                body: "ok".into(),
            },
        );
        let (addr, _handle) = server.serve().await;
        let transport = PolicyCheckedHttpTransport::with_policy_and_scan(
            Some(allowed_policy_for(addr.port())),
            Some(blocking_scan()),
        );
        assert!(transport.outbound_scan().is_some());
        let url = format!("http://{addr}/send");
        let body = serde_json::json!({ "key": "ghp_0123456789abcdefghijklmnopqrstuv" });
        let err = execute_post_json(&transport, &url, HeaderMap::new(), &body)
            .await
            .unwrap_err();
        assert!(matches!(err, EgressError::SecretBlocked { .. }), "{err:?}");
        assert_eq!(server.request_count(), 0);
        // Same transport, clean body: sent.
        let ok_body = serde_json::json!({ "key": "benign" });
        let resp = execute_post_json(&transport, &url, HeaderMap::new(), &ok_body)
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(server.request_count(), 1);
    }
}
