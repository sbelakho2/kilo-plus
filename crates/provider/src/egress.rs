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

use reqwest::{Request, RequestBuilder, Response, Url};
use serde::{Deserialize, Serialize};

use faktor_security::destination::{Decision, DeniedReason, DestinationPolicy, RequestTarget};

/// Why an outbound call refused to leave the process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EgressError {
    /// The installed allowlist denied the destination before any connect.
    /// `url` is the parsed request target (never a re-split string).
    Denied { url: String, reason: DeniedReason },
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
/// gate. This is the enforcement point a fetch/web tool (the genuinely
/// user-prompt-derived egress) must use for its outbound calls.
#[derive(Debug, Clone)]
pub struct CheckedHttpClient {
    inner: reqwest::Client,
    policy: Option<DestinationPolicy>,
}

impl CheckedHttpClient {
    /// Wrap a client with an optional allowlist. `None` = no policy
    /// installed = default-allow (documented).
    pub fn new(inner: reqwest::Client, policy: Option<DestinationPolicy>) -> CheckedHttpClient {
        CheckedHttpClient { inner, policy }
    }

    /// A default client with the given allowlist.
    pub fn with_policy(policy: Option<DestinationPolicy>) -> CheckedHttpClient {
        CheckedHttpClient::new(reqwest::Client::new(), policy)
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
    /// is checked against the policy on its parsed URL before `execute`.
    pub async fn execute(&self, request: Request) -> Result<Response, EgressError> {
        self.check(request.url())?;
        let url = request.url().clone();
        let response = self
            .inner
            .execute(request)
            .await
            .map_err(|e| EgressError::Transport(format!("{url}: {e}")))?;
        Ok(response)
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
}
