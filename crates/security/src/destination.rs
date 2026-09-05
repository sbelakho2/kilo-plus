//! Parsed destination allowlisting (audits 36-37).
//!
//! Every app-level egress decision compares a **parsed** (scheme, host,
//! port) triple against allowlist rules. Host matching is *label-exact*:
//! there is no prefix/substring matching anywhere, so `evil-example.com`
//! can never match a rule for `example.com`. The only non-exact form is a
//! literal `*.example.com` wildcard rule, which matches exactly the
//! subdomains of `example.com` (one or more leading labels), never the bare
//! apex and never anything whose labels do not align.
//!
//! # Rule text forms (one rule per entry)
//!
//! ```text
//! example.com                      any scheme, any port
//! example.com:443                  any scheme, port 443
//! https://example.com              https only, any port
//! https://example.com:443          https only, port 443
//! *.example.com                    subdomains of example.com, any scheme/port
//! https://*.example.com:443        scheme+port constrained wildcard
//! 127.0.0.1                        IPv4 literal (octets canonicalized)
//! http://127.0.0.1:9911            scheme+host+port triple
//! [::1]:8080                       bracketed IPv6 literal
//! ```
//!
//! A rule with no scheme (`scheme = None`) matches any of the four wire
//! schemes (`http https ws wss`); it never matches a request whose scheme
//! is unknown/absent (an ambiguous destination is denied, never allowed).
//! A rule with no port (`port = None`) matches any port, *including* the
//! scheme-default ports of the request target. So `example.com` allows
//! `http://example.com:80` and `https://example.com:443` alike, while
//! `example.com:443` allows only the latter.
//!
//! # Parse strictness
//!
//! Rules are parsed at **configuration time** and a single bad entry is a
//! policy load error — never a silently skipped or silently permissive
//! entry. Rejected: empty entries, whitespace, path components (`/`),
//! unsupported schemes (`ftp://`…), unbracketed IPv6, ports that are
//! missing/empty/non-numeric/out of range, hosts that are neither a
//! hostname nor an IP literal (including `..`, empty labels, `http://*.com`
//! bare-suffix wildcards, `*` anywhere but a leading `*.`), and duplicate
//! exact rules (an error, never last-wins).
//!
//! # Normalization
//!
//! Hosts on both sides are canonicalized before comparison: ASCII
//! lowercased, one trailing dot stripped (`example.com.` == `example.com`),
//! IDNs punycoded through UTS-46 (`exämple.com` == `xn--exmple-cua.com`),
//! IPv4 literals reduced to canonical dotted octets (leading zeros and
//! short forms compare equal), and IPv6 literals to their canonical
//! unbracketed lowercase form.
//!
//! # Decision semantics
//!
//! | state                                   | outcome |
//! |-----------------------------------------|---------|
//! | no policy installed                     | allow (default-allow; callers represent this as `Option::None`) |
//! | policy installed, rule fully matches    | allow |
//! | policy installed (even empty), no match | deny (default-deny) |
//!
//! Multiple matching rules cannot conflict — all rules allow. Rule order is
//! deterministic longest-specificity (exact host > wildcard; then explicit
//! port, then explicit scheme), which decides which rule a [`DeniedReason`]
//! attributes the denial to for logs. A denied reason names the most
//! specific rule whose **host** matched (`rule_fired`) and how far the
//! match got (`matched`: host only, or host+port with the scheme missed).
//! A deny is never produced by prefix matching: it requires a full
//! scheme+host+port label match to allow, and denial under default-deny
//! reports `rule_fired = None` when not even a host matched.

use std::fmt;
use std::net::{Ipv4Addr, Ipv6Addr};

use serde::{Deserialize, Serialize};

/// Hard cap on one rule entry and one textual destination string.
const MAX_ENTRY_CHARS: usize = 4096;

/// The wire schemes an allowlist rule can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scheme {
    Http,
    Https,
    Ws,
    Wss,
}

impl Scheme {
    pub const ALL: [Scheme; 4] = [Scheme::Http, Scheme::Https, Scheme::Ws, Scheme::Wss];

    /// Case-insensitive parse of a scheme name.
    pub fn parse(text: &str) -> Option<Scheme> {
        Self::ALL
            .iter()
            .copied()
            .find(|s| s.as_str().eq_ignore_ascii_case(text))
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Scheme::Http => "http",
            Scheme::Https => "https",
            Scheme::Ws => "ws",
            Scheme::Wss => "wss",
        }
    }

    /// The RFC default port of the scheme.
    pub fn default_port(&self) -> u16 {
        match self {
            Scheme::Http | Scheme::Ws => 80,
            Scheme::Https | Scheme::Wss => 443,
        }
    }
}

/// Host side of a rule. Stored already canonicalized (lowercase,
/// punycode, no trailing dot, canonical IP text) by the parser.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostMatcher {
    /// Label-exact host. Never a prefix: `example.com` matches only the
    /// canonical host `example.com`.
    Exact { host: String },
    /// Literal `*.example.com` rule: matches hosts that are `example.com`
    /// with one or more leading labels (`api.example.com`,
    /// `a.b.example.com`), never `example.com` itself and never
    /// `evil-example.com` / `example.com.evil`.
    WildcardSubdomain { suffix: String },
}

impl HostMatcher {
    /// `canonical_host` must already be canonicalized (same normalization
    /// as the rule side). Matching is label-exact; the wildcard arm checks
    /// the label boundary explicitly.
    pub fn matches(&self, canonical_host: &str) -> bool {
        match self {
            HostMatcher::Exact { host } => host == canonical_host,
            HostMatcher::WildcardSubdomain { suffix } => {
                if canonical_host.len() <= suffix.len() {
                    return false;
                }
                let Some(apex) = canonical_host.strip_suffix(suffix) else {
                    return false;
                };
                // One or more full labels before the suffix: apex ends with
                // a dot and is non-empty (a bare `example.com` leaves an
                // empty apex and never matches its own wildcard).
                !apex.is_empty() && apex.ends_with('.')
            }
        }
    }

    fn specificity(&self) -> u8 {
        match self {
            HostMatcher::Exact { .. } => 2,
            HostMatcher::WildcardSubdomain { .. } => 1,
        }
    }
}

impl fmt::Display for HostMatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HostMatcher::Exact { host } => f.write_str(host),
            HostMatcher::WildcardSubdomain { suffix } => write!(f, "*.{suffix}"),
        }
    }
}

/// One parsed destination rule. `scheme: None` means "any of the four wire
/// schemes"; `port: None` means "any port". `text` is the original entry
/// (trimmed) for logs and error messages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DestinationRule {
    pub scheme: Option<Scheme>,
    pub host: HostMatcher,
    pub port: Option<u16>,
    pub text: String,
}

impl fmt::Display for DestinationRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl DestinationRule {
    /// Parse one rule entry. Errors are strict and descriptive; nothing
    /// permissive is ever produced from a bad entry.
    pub fn parse(entry: &str) -> Result<DestinationRule, RuleParseError> {
        parse_rule(entry)
    }

    /// Canonical identity used for duplicate detection: scheme + host
    /// matcher + port, ignoring the display text.
    fn canonical_key(&self) -> (Option<Scheme>, HostMatcher, Option<u16>) {
        (self.scheme, self.host.clone(), self.port)
    }

    fn specificity(&self) -> (u8, u8, u8) {
        (
            self.host.specificity(),
            u8::from(self.port.is_some()),
            u8::from(self.scheme.is_some()),
        )
    }

    /// A rule whose `scheme` is `None` matches any *known* wire scheme —
    /// never an unknown/absent scheme (ambiguous destinations stay denied
    /// under an installed policy).
    fn scheme_matches(&self, scheme: &str) -> bool {
        match self.scheme {
            None => matches!(scheme, "http" | "https" | "ws" | "wss"),
            Some(s) => s.as_str() == scheme,
        }
    }

    fn port_matches(&self, port: u16) -> bool {
        match self.port {
            // Port 0 is the "no port known" sentinel; an unknown port never
            // counts as "any port".
            None => port != 0,
            Some(p) => p == port,
        }
    }
}

/// How far a denied request got against the reported rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleMatch {
    /// No rule matched the request even at the host level.
    None,
    /// The reported rule's host matched; its port/scheme constraints missed.
    Host,
    /// The reported rule's host and port matched; its scheme missed.
    HostAndPort,
}

/// Why a destination was denied. Always carries enough for a log line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeniedReason {
    /// The most specific allow rule whose host matched the request, as its
    /// original text. `None` when nothing matched at all (plain
    /// default-deny), or when the request host itself was not a valid
    /// canonical host.
    pub rule_fired: Option<String>,
    /// How far the fired rule matched (see [`RuleMatch`]).
    pub matched: RuleMatch,
}

impl fmt::Display for DeniedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.rule_fired, self.matched) {
            (None, RuleMatch::None) => {
                write!(f, "no allowlist rule matches (default-deny)")
            }
            (Some(rule), RuleMatch::Host) => write!(
                f,
                "host matches allowlist rule {rule:?} but the request scheme/port misses it"
            ),
            (Some(rule), RuleMatch::HostAndPort) => write!(
                f,
                "host and port match allowlist rule {rule:?} but the request scheme misses it"
            ),
            (Some(rule), RuleMatch::None) => write!(
                f,
                "allowlist rule {rule:?} does not match this destination (default-deny)"
            ),
            (None, _) => {
                write!(f, "no allowlist rule matches (default-deny)")
            }
        }
    }
}

/// Outcome of one destination decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allowed,
    Denied(DeniedReason),
}

impl Decision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Decision::Allowed)
    }

    /// The deny reason when denied.
    pub fn denied_reason(&self) -> Option<&DeniedReason> {
        match self {
            Decision::Allowed => None,
            Decision::Denied(r) => Some(r),
        }
    }
}

/// A config-time rule parse failure. `line` is 0 for single-rule parses
/// and 1-based inside [`DestinationPolicy::parse_lines`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleParseError {
    pub entry: String,
    pub reason: String,
    pub line: usize,
}

impl fmt::Display for RuleParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.line > 0 {
            write!(f, "line {}: {}", self.line, self.reason)
        } else {
            write!(f, "{}", self.reason)
        }
    }
}

impl std::error::Error for RuleParseError {}

fn rule_error(entry: &str, reason: impl Into<String>) -> RuleParseError {
    RuleParseError {
        entry: entry.to_string(),
        reason: reason.into(),
        line: 0,
    }
}

// ---------------------------------------------------------------------------
// Host normalization
// ---------------------------------------------------------------------------

/// Result of canonicalizing a host (rule or request side).
#[derive(Debug, Clone, PartialEq, Eq)]
enum CanonHost {
    Name(String),
    V4([u8; 4]),
}

/// Is `h` a dotted quad made of exactly four numeric labels?
fn is_dotted_quad(h: &str) -> bool {
    let mut labels = h.split('.');
    for _ in 0..4 {
        let label = labels.next().unwrap_or("");
        if label.is_empty() || !label.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
    }
    labels.next().is_none()
}

/// Parse an ASCII dotted quad into octets. `u8::from_str` accepts leading
/// zeros, which is what we want (they canonicalize away).
fn parse_dotted_quad(h: &str) -> Option<[u8; 4]> {
    let mut octets = [0u8; 4];
    for (i, label) in h.split('.').enumerate() {
        let value: u16 = label.parse().ok()?;
        if value > 255 {
            return None;
        }
        octets[i] = value as u8;
    }
    Some(octets)
}

fn host_error(reason: impl Into<String>) -> Result<CanonHost, String> {
    Err(reason.into())
}

/// Canonicalize a textual host (no brackets, no port). Shared by the rule
/// parser and the request side so both compare equal.
///
/// - ASCII lowercasing, then at most one trailing dot is stripped.
/// - A four-label dotted quad must be a valid IPv4 literal; other numeric
///   dotted names are valid hostnames; a lone all-numeric label is rejected
///   as ambiguous (not an IP, not a name).
/// - Non-ASCII names are punycoded through UTS-46 (`idna`).
/// - ASCII names use a strict RFC-style label grammar: `[a-z0-9-]`, no
///   leading/trailing `-`, no empty labels, no `*`/`_`/punctuation.
fn canonicalize_host_text(h: &str) -> Result<CanonHost, String> {
    if h.is_empty() {
        return host_error("empty host");
    }
    if h.chars().count() > MAX_ENTRY_CHARS {
        return host_error("host too long");
    }
    let mut h = h;
    if h.ends_with('.') {
        // Strip exactly one trailing dot (`.`, `..`, `x..` produce empty or
        // double-dot labels below and are rejected).
        h = &h[..h.len() - 1];
    }
    if h.is_empty() {
        return host_error("empty host after trailing dot");
    }
    if h.contains('*') {
        return host_error("'*' is only allowed as a leading '*.' wildcard");
    }
    if h.contains(':') {
        return host_error("IPv6 literals must be bracketed: [addr]");
    }
    if is_dotted_quad(h) {
        return match parse_dotted_quad(h) {
            Some(o) => Ok(CanonHost::V4(o)),
            None => host_error(format!("not a valid IPv4 literal: {h:?}")),
        };
    }

    let lower: String = h.to_ascii_lowercase();
    if lower.is_ascii() {
        if lower.bytes().all(|b| b.is_ascii_digit()) {
            // Single all-numeric label: ambiguous IP shorthand, never a name.
            return host_error(format!("not a hostname and not an IP literal: {h:?}"));
        }
        validate_ascii_labels(&lower)?;
        Ok(CanonHost::Name(lower))
    } else {
        // IDN: hand off to UTS-46 (mapping + case folding + punycode).
        let ascii = idna::domain_to_ascii(&lower)
            .map_err(|e| format!("invalid IDN hostname {h:?}: {e}"))?;
        if ascii.chars().count() > 253 {
            return host_error("hostname exceeds 253 characters");
        }
        validate_ascii_labels(&ascii)?;
        Ok(CanonHost::Name(ascii))
    }
}

/// Validate an all-ASCII (already lowercased, punycode) hostname: non-empty
/// labels of `[a-z0-9-]`, no leading/trailing hyphen, labels <= 63 chars.
fn validate_ascii_labels(host: &str) -> Result<(), String> {
    let fail = |reason: String| Err(reason);
    for label in host.split('.') {
        if label.is_empty() {
            return fail(format!("empty label in hostname {host:?}"));
        }
        if label.len() > 63 {
            return fail(format!("hostname label exceeds 63 characters: {label:?}"));
        }
        if !label
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        {
            return fail(format!(
                "not a valid hostname label: {label:?} (letters, digits and '-' only)"
            ));
        }
        if label.starts_with('-') || label.ends_with('-') {
            return fail(format!(
                "hostname label must not start/end with '-': {label:?}"
            ));
        }
    }
    Ok(())
}

/// Canonicalize a *request* host for matching. `is_ipv4`/`ip` come from
/// the parsed URL's host (e.g. `reqwest::Url`); when the caller already
/// parsed the address, the octets win over the textual spelling so that
/// any alternate IPv4 text (`127.000.0.01`, …) compares canonical.
fn canonicalize_request_host(
    host: &str,
    is_ipv4: bool,
    ip: Option<[u8; 4]>,
) -> Result<String, String> {
    if is_ipv4 {
        if let Some(octets) = ip {
            return Ok(Ipv4Addr::from(octets).to_string());
        }
    }
    if host.contains(':') {
        // Unbracketed canonical IPv6 text as emitted by URL parsers
        // (`url::Url::host_str` never includes brackets).
        let v6: Ipv6Addr = host
            .parse()
            .map_err(|_| format!("not a valid IPv6 literal: {host:?}"))?;
        return Ok(v6.to_string());
    }
    match canonicalize_host_text(host)? {
        CanonHost::Name(n) => Ok(n),
        CanonHost::V4(o) => Ok(Ipv4Addr::from(o).to_string()),
    }
}

// ---------------------------------------------------------------------------
// Rule text grammar
// ---------------------------------------------------------------------------

fn parse_port(text: &str, entry: &str) -> Result<u16, RuleParseError> {
    if text.is_empty() {
        return Err(rule_error(entry, "empty port after ':'"));
    }
    if !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(rule_error(
            entry,
            format!("port must be a number 1..=65535, got {text:?}"),
        ));
    }
    match text.parse::<u16>() {
        Ok(0) => Err(rule_error(entry, "port 0 is not a valid destination port")),
        Ok(p) => Ok(p),
        Err(_) => Err(rule_error(
            entry,
            format!("port out of range (1..=65535): {text:?}"),
        )),
    }
}

fn parse_rule(entry_raw: &str) -> Result<DestinationRule, RuleParseError> {
    let entry = entry_raw.trim();
    if entry.is_empty() {
        return Err(rule_error(entry_raw, "empty rule entry"));
    }
    if entry.chars().count() > MAX_ENTRY_CHARS {
        return Err(rule_error(entry, "rule entry too long"));
    }
    if entry.chars().any(char::is_whitespace) {
        return Err(rule_error(entry, "whitespace is not allowed in a rule"));
    }

    // Optional `scheme://` prefix.
    let (scheme, rest) = match entry.find("://") {
        Some(i) => {
            let pre = &entry[..i];
            let Some(scheme) = Scheme::parse(pre) else {
                return Err(rule_error(
                    entry,
                    format!("unsupported scheme {pre:?} (http, https, ws, wss only)"),
                ));
            };
            (Some(scheme), &entry[i + 3..])
        }
        None => (None, entry),
    };
    if rest.contains('/') {
        return Err(rule_error(
            entry,
            "path components are not allowed in a destination rule",
        ));
    }

    // Host[:port] with optional bracketed IPv6.
    let (host_matcher, port) = if let Some(bracket_end) = rest.find(']') {
        if !rest.starts_with('[') {
            return Err(rule_error(entry, "']' without a leading '['"));
        }
        let inner = &rest[1..bracket_end];
        let tail = &rest[bracket_end + 1..];
        let v6: Ipv6Addr = inner
            .parse()
            .map_err(|_| rule_error(entry, format!("invalid IPv6 literal {inner:?}")))?;
        let canonical = v6.to_string();
        let port = if tail.is_empty() {
            None
        } else if let Some(p) = tail.strip_prefix(':') {
            Some(parse_port(p, entry)?)
        } else {
            return Err(rule_error(
                entry,
                "unexpected characters after ']' (expected ':port' or nothing)",
            ));
        };
        (HostMatcher::Exact { host: canonical }, port)
    } else {
        let (host_part, port_part) = match rest.split_once(':') {
            Some((h, p)) => {
                if p.contains(':') {
                    return Err(rule_error(entry, "IPv6 literals must be bracketed: [addr]"));
                }
                (h, Some(p))
            }
            None => (rest, None),
        };
        let port = match port_part {
            None => None,
            Some(p) => Some(parse_port(p, entry)?),
        };
        (parse_host_part(host_part, entry)?, port)
    };

    Ok(DestinationRule {
        scheme,
        host: host_matcher,
        port,
        text: entry.to_string(),
    })
}

/// Parse the host half of a rule entry (already separated from port), with
/// `*.` wildcard handling.
fn parse_host_part(host_part: &str, entry: &str) -> Result<HostMatcher, RuleParseError> {
    if host_part.is_empty() {
        return Err(rule_error(entry, "empty host"));
    }
    if let Some(suffix) = host_part.strip_prefix("*.") {
        if suffix.is_empty() || suffix.contains('*') {
            return Err(rule_error(
                entry,
                "wildcard must be written as '*.domain' with a full suffix domain",
            ));
        }
        match canonicalize_host_text(suffix).map_err(|e| rule_error(entry, e))? {
            CanonHost::Name(n) => {
                if !n.contains('.') {
                    return Err(rule_error(
                        entry,
                        format!("wildcard suffix must include at least one leading label: {n:?}"),
                    ));
                }
                Ok(HostMatcher::WildcardSubdomain { suffix: n })
            }
            _ => Err(rule_error(
                entry,
                "wildcard suffix must be a domain name, not an IP literal",
            )),
        }
    } else {
        match canonicalize_host_text(host_part).map_err(|e| rule_error(entry, e))? {
            CanonHost::Name(n) => Ok(HostMatcher::Exact { host: n }),
            CanonHost::V4(o) => Ok(HostMatcher::Exact {
                host: Ipv4Addr::from(o).to_string(),
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------

/// A parsed destination allowlist. Sorted by descending specificity at
/// construction so the first full match (and the reported denial candidate)
/// is deterministic: exact host beats wildcard, then explicit port, then
/// explicit scheme.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DestinationPolicy {
    rules: Vec<DestinationRule>,
}

impl DestinationPolicy {
    /// Parse a policy from rule entries. Every entry must parse (strict);
    /// duplicate exact rules are an error. An empty list is a legal policy
    /// and means default-deny for everything.
    pub fn parse_lines<I>(lines: I) -> Result<DestinationPolicy, RuleParseError>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        let mut rules: Vec<DestinationRule> = Vec::new();
        for (idx, line) in lines.into_iter().enumerate() {
            let line = line.as_ref();
            let parsed = parse_rule(line).map_err(|mut e| {
                e.line = idx + 1;
                e
            })?;
            if let Some(dup) = rules
                .iter()
                .find(|existing| existing.canonical_key() == parsed.canonical_key())
            {
                return Err(RuleParseError {
                    entry: parsed.text.clone(),
                    reason: format!("duplicate rule: {parsed:?} repeats the earlier rule {dup:?}"),
                    line: idx + 1,
                });
            }
            rules.push(parsed);
        }
        Ok(DestinationPolicy { rules }.sorted())
    }

    /// The empty policy: installed, denies everything.
    pub fn empty() -> DestinationPolicy {
        DestinationPolicy { rules: Vec::new() }
    }

    /// Build from already-parsed rules, erroring on duplicates.
    pub fn from_rules(rules: Vec<DestinationRule>) -> Result<DestinationPolicy, RuleParseError> {
        for (idx, rule) in rules.iter().enumerate() {
            if let Some(dup) = rules[..idx]
                .iter()
                .find(|existing| existing.canonical_key() == rule.canonical_key())
            {
                return Err(RuleParseError {
                    entry: rule.text.clone(),
                    reason: format!("duplicate rule: {rule:?} repeats the earlier rule {dup:?}"),
                    line: 0,
                });
            }
        }
        Ok(DestinationPolicy { rules }.sorted())
    }

    pub fn rules(&self) -> &[DestinationRule] {
        &self.rules
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    fn sorted(mut self) -> DestinationPolicy {
        self.rules.sort_by(|a, b| {
            b.specificity()
                .cmp(&a.specificity())
                .then_with(|| a.text.cmp(&b.text))
        });
        self
    }

    /// Decide one destination. `scheme` is the canonical lowercase scheme
    /// (`http|https|ws|wss`); an unknown/absent scheme (`""` or anything
    /// else) can never match a rule and is denied under an installed
    /// policy. `port` is the real connection port (scheme defaults already
    /// resolved); `0` means "no port known". `is_ipv4`/`ip` come from the
    /// parsed URL host.
    pub fn check(
        &self,
        scheme: &str,
        host: &str,
        port: u16,
        is_ipv4: bool,
        ip: Option<[u8; 4]>,
    ) -> Decision {
        let Ok(canonical_host) = canonicalize_request_host(host, is_ipv4, ip) else {
            // Junk host text: no rule can match (never prefix/allowed by
            // accident), report plain default-deny.
            return Decision::Denied(DeniedReason {
                rule_fired: None,
                matched: RuleMatch::None,
            });
        };
        if self.rules.is_empty() {
            return Decision::Denied(DeniedReason {
                rule_fired: None,
                matched: RuleMatch::None,
            });
        }
        for rule in &self.rules {
            if rule.scheme_matches(scheme)
                && rule.host.matches(&canonical_host)
                && rule.port_matches(port)
            {
                return Decision::Allowed;
            }
        }
        // Denial candidate: the most specific rule whose host matched
        // (rules are pre-sorted by specificity).
        let (candidate, host_matched, port_matched) =
            match self.rules.iter().find(|r| r.host.matches(&canonical_host)) {
                Some(rule) => {
                    let host_matched = rule.host.matches(&canonical_host);
                    let port_matched = rule.port_matches(port);
                    (Some(rule.text.clone()), host_matched, port_matched)
                }
                None => (None, false, false),
            };
        debug_assert!(host_matched == candidate.is_some());
        let matched = if port_matched && host_matched {
            RuleMatch::HostAndPort
        } else if host_matched {
            RuleMatch::Host
        } else {
            RuleMatch::None
        };
        Decision::Denied(DeniedReason {
            rule_fired: candidate,
            matched,
        })
    }

    /// Convenience: `true` when a request is allowed under this policy.
    pub fn allows(
        &self,
        scheme: &str,
        host: &str,
        port: u16,
        is_ipv4: bool,
        ip: Option<[u8; 4]>,
    ) -> bool {
        self.check(scheme, host, port, is_ipv4, ip).is_allowed()
    }
}

// ---------------------------------------------------------------------------
// Request side
// ---------------------------------------------------------------------------

/// A parsed request target. Parsed once from the actual URL object (or a
/// textual destination at capability boundaries); the connection must then
/// be made to that same parsed target (no resolve-then-reparse split).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestTarget {
    /// Canonical lowercase scheme; `""` when unknown/absent.
    pub scheme: String,
    /// Canonical host (punycode/lowercase/no trailing dot; canonical IPv4
    /// dotted octets or canonical unbracketed IPv6).
    pub host: String,
    /// The real connection port. `None` when unknown (no scheme, no
    /// explicit port).
    pub port: Option<u16>,
    pub is_ipv4: bool,
    pub ip: Option<[u8; 4]>,
}

impl RequestTarget {
    /// Build a target from the pieces of a *parsed* URL (host/port pulled
    /// from the URL object, never from re-splitting strings). Unknown
    /// scheme → `""`; known scheme with no explicit port → the scheme's
    /// default port.
    pub fn from_parts(
        scheme: Option<&str>,
        host: &str,
        explicit_port: Option<u16>,
        is_ipv4: bool,
        ip: Option<[u8; 4]>,
    ) -> Result<RequestTarget, String> {
        let scheme = scheme.unwrap_or("").to_ascii_lowercase();
        if !scheme.is_empty() && Scheme::parse(&scheme).is_none() {
            return Err(format!("unsupported scheme {scheme:?}"));
        }
        let canonical = canonicalize_request_host(host, is_ipv4, ip)?;
        let port = match explicit_port {
            Some(p) => Some(p),
            None => Scheme::parse(&scheme).map(|s| s.default_port()),
        };
        Ok(RequestTarget {
            scheme,
            host: canonical,
            port,
            is_ipv4,
            ip,
        })
    }

    /// Parse a textual destination (a capability `destination` string or a
    /// fetch-tool URL argument). Accepts `scheme://host[:port][/path]` for
    /// http/https/ws/wss and bare `host[:port]`. Paths are ignored for the
    /// decision (they never influence the gate). A bare host without a
    /// scheme parses with an *unknown* scheme (`""`) and an unknown port:
    /// such an ambiguous target can never match an allowlist rule (which
    /// requires a known wire scheme). Anything outside the grammar is an
    /// error — the caller must deny unparseable input.
    pub fn parse(destination: &str) -> Result<RequestTarget, String> {
        let text = destination.trim();
        if text.is_empty() {
            return Err("empty destination".into());
        }
        if text.chars().count() > MAX_ENTRY_CHARS {
            return Err("destination too long".into());
        }
        let (scheme, rest) = match text.find("://") {
            Some(i) => {
                let pre = &text[..i];
                let Some(scheme) = Scheme::parse(pre) else {
                    return Err(format!("unsupported scheme {pre:?}"));
                };
                (Some(scheme), &text[i + 3..])
            }
            None => (None, text),
        };

        // With a scheme, split host[:port] from the (ignored) path; without
        // one the whole string must be a bare host[:port].
        let authority = match scheme {
            Some(_) => match rest.find('/') {
                Some(i) => &rest[..i],
                None => rest,
            },
            None => {
                if rest.contains('/') {
                    return Err("a destination without a scheme must be bare host[:port]".into());
                }
                rest
            }
        };
        if authority.is_empty() {
            return Err("empty host".into());
        }
        if authority.contains('?') || authority.contains('#') {
            // No query/fragment in the authority of a plain URL string.
            return Err("destination must not carry a query or fragment".into());
        }

        let (host_raw, explicit_port) = if let Some(bracket_end) = authority.find(']') {
            if !authority.starts_with('[') {
                return Err("invalid authority".into());
            }
            let inner = &authority[1..bracket_end];
            let tail = &authority[bracket_end + 1..];
            let v6: Ipv6Addr = inner
                .parse()
                .map_err(|_| format!("invalid IPv6 literal {inner:?}"))?;
            let explicit_port = match tail {
                "" => None,
                t if t.starts_with(':') => Some(parse_u16_port(&t[1..])?),
                _ => return Err("unexpected characters after ']'".into()),
            };
            let port = match (scheme, explicit_port) {
                (Some(sc), None) => Some(sc.default_port()),
                (_, p) => p,
            };
            let canon = v6.to_string();
            return Ok(RequestTarget {
                scheme: scheme.map(|s| s.as_str().to_string()).unwrap_or_default(),
                host: canon,
                port,
                is_ipv4: false,
                ip: None,
            });
        } else {
            match authority.split_once(':') {
                Some((h, p)) => {
                    if p.contains(':') {
                        return Err("IPv6 literals must be bracketed".into());
                    }
                    (h, Some(parse_u16_port(p)?))
                }
                None => (authority, None),
            }
        };
        if host_raw.is_empty() {
            return Err("empty host".into());
        }

        // With a known scheme an absent port resolves to the scheme
        // default; a bare host without scheme keeps port unknown.
        let port = match (scheme, explicit_port) {
            (Some(sc), None) => Some(sc.default_port()),
            (_, p) => p,
        };
        let canonical = canonicalize_host_text(host_raw).map_err(|e| e.to_string())?;
        let (host, is_ipv4, ip) = match canonical {
            CanonHost::Name(n) => (n, false, None),
            CanonHost::V4(o) => (Ipv4Addr::from(o).to_string(), true, Some(o)),
        };
        Ok(RequestTarget {
            scheme: scheme.map(|s| s.as_str().to_string()).unwrap_or_default(),
            host,
            port,
            is_ipv4,
            ip,
        })
    }

    /// Human-readable description of the parsed target for logs and deny
    /// messages (`https://host:443`, or `host (scheme/port unknown)`).
    pub fn describe(&self) -> String {
        match (self.scheme.is_empty(), self.port) {
            (false, Some(p)) => format!("{}://{}:{}", self.scheme, self.host, p),
            (false, None) => format!("{}://{}", self.scheme, self.host),
            (true, Some(p)) => format!("{}:{} (scheme unknown)", self.host, p),
            (true, None) => format!("{} (scheme/port unknown)", self.host),
        }
    }

    /// Check this target against a policy (already parsed; see
    /// [`DestinationPolicy::check`]).
    pub fn check_against(&self, policy: &DestinationPolicy) -> Decision {
        policy.check(
            &self.scheme,
            &self.host,
            self.port.unwrap_or(0),
            self.is_ipv4,
            self.ip,
        )
    }
}

fn parse_u16_port(text: &str) -> Result<u16, String> {
    if text.is_empty() {
        return Err("empty port".into());
    }
    if !text.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("port must be numeric, got {text:?}"));
    }
    match text.parse::<u16>() {
        Ok(0) => Err("port 0 is not valid".into()),
        Ok(p) => Ok(p),
        Err(_) => Err(format!("port out of range: {text:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------ parsing

    #[test]
    fn every_supported_text_form_parses() {
        let cases: &[(&str, Option<Scheme>, HostMatcher, Option<u16>)] = &[
            (
                "example.com",
                None,
                HostMatcher::Exact {
                    host: "example.com".into(),
                },
                None,
            ),
            (
                "example.com:443",
                None,
                HostMatcher::Exact {
                    host: "example.com".into(),
                },
                Some(443),
            ),
            (
                "https://example.com",
                Some(Scheme::Https),
                HostMatcher::Exact {
                    host: "example.com".into(),
                },
                None,
            ),
            (
                "HTTPS://example.com",
                Some(Scheme::Https),
                HostMatcher::Exact {
                    host: "example.com".into(),
                },
                None,
            ),
            (
                "https://example.com:443",
                Some(Scheme::Https),
                HostMatcher::Exact {
                    host: "example.com".into(),
                },
                Some(443),
            ),
            (
                "ws://example.com:8080",
                Some(Scheme::Ws),
                HostMatcher::Exact {
                    host: "example.com".into(),
                },
                Some(8080),
            ),
            (
                "wss://example.com",
                Some(Scheme::Wss),
                HostMatcher::Exact {
                    host: "example.com".into(),
                },
                None,
            ),
            (
                "*.example.com",
                None,
                HostMatcher::WildcardSubdomain {
                    suffix: "example.com".into(),
                },
                None,
            ),
            (
                "https://*.example.com:443",
                Some(Scheme::Https),
                HostMatcher::WildcardSubdomain {
                    suffix: "example.com".into(),
                },
                Some(443),
            ),
            (
                "127.0.0.1",
                None,
                HostMatcher::Exact {
                    host: "127.0.0.1".into(),
                },
                None,
            ),
            (
                "http://127.0.0.1:9911",
                Some(Scheme::Http),
                HostMatcher::Exact {
                    host: "127.0.0.1".into(),
                },
                Some(9911),
            ),
            (
                "[::1]",
                None,
                HostMatcher::Exact { host: "::1".into() },
                None,
            ),
            (
                "[::1]:8080",
                None,
                HostMatcher::Exact { host: "::1".into() },
                Some(8080),
            ),
            (
                "http://[2001:db8::1]:80",
                Some(Scheme::Http),
                HostMatcher::Exact {
                    host: "2001:db8::1".into(),
                },
                Some(80),
            ),
        ];
        for (text, scheme, host, port) in cases {
            let rule = DestinationRule::parse(text).unwrap_or_else(|e| panic!("{text}: {e}"));
            assert_eq!(&rule.scheme, scheme, "{text}");
            assert_eq!(&rule.host, host, "{text}");
            assert_eq!(&rule.port, port, "{text}");
        }
    }

    #[test]
    fn rule_text_is_canonicalized_at_parse_time() {
        let r = DestinationRule::parse("EXAMPLE.COM.").unwrap();
        assert_eq!(
            r.host,
            HostMatcher::Exact {
                host: "example.com".into()
            }
        );
        // IDN entry punycodes to its allowlist form.
        let r = DestinationRule::parse("exämple.com").unwrap();
        assert_eq!(
            r.host,
            HostMatcher::Exact {
                host: "xn--exmple-cua.com".into()
            }
        );
        // IPv4 with leading zeros canonicalizes to dotted octets.
        let r = DestinationRule::parse("127.000.0.001:80").unwrap();
        assert_eq!(
            r.host,
            HostMatcher::Exact {
                host: "127.0.0.1".into()
            }
        );
        // Wildcard suffix normalizes the same way.
        let r = DestinationRule::parse("*.EXAMPLE.com").unwrap();
        assert_eq!(
            r.host,
            HostMatcher::WildcardSubdomain {
                suffix: "example.com".into()
            }
        );
        // IPv6 text canonicalizes (compressed, lowercase, unbracketed).
        let r = DestinationRule::parse("[2001:0DB8:0:0:0:0:0:1]").unwrap();
        assert_eq!(
            r.host,
            HostMatcher::Exact {
                host: "2001:db8::1".into()
            }
        );
    }

    #[test]
    fn parse_rejections_are_strict_and_descriptive() {
        let rejections: &[(&str, &str)] = &[
            ("", "empty"),
            ("   ", "empty"),
            ("https://x/path", "path"),
            ("example.com/a", "path"),
            ("exa mple.com", "whitespace"),
            ("example.com :443", "whitespace"),
            ("http://*.com", "leading label"),
            ("*.com", "leading label"),
            ("*.", "suffix"),
            ("*", "wildcard"),
            ("https://*.127.0.0.1", "IP literal"),
            ("example.com:0", "port"),
            ("example.com:65536", "range"),
            ("example.com:99999", "range"),
            ("example.com:abc", "number"),
            ("example.com:", "empty port"),
            (":443", "empty host"),
            ("http://", "empty host"),
            ("ftp://example.com", "scheme"),
            ("example.com..", "empty label"),
            (".example.com", "empty label"),
            ("exa..mple.com", "empty label"),
            ("example..com", "empty label"),
            ("evil_example.com", "label"),
            ("-example.com", "start/end"),
            ("example-.com", "start/end"),
            ("example*", "wildcard"),
            ("999.1.1.1", "IPv4"),
            ("123", "hostname"),
            ("2001:db8::1", "bracketed"),
            ("http://user@example.com", "label"),
            ("https://example.com:80:90", "bracketed"),
            ("https://example.com:80/x", "path"),
            ("\u{FFFE}example.com", "idn"),
            ("\u{FFFE}", "idn"),
        ];
        for (text, why) in rejections {
            let err = DestinationRule::parse(text).unwrap_err();
            assert!(
                err.reason.to_lowercase().contains(&why.to_lowercase()),
                "{text:?} rejected for {why:?} but got: {}",
                err.reason
            );
        }
    }

    #[test]
    fn duplicate_exact_rules_error_never_last_wins() {
        let err = DestinationPolicy::parse_lines(["example.com", "EXAMPLE.com"]).unwrap_err();
        assert!(err.reason.contains("duplicate"));
        assert_eq!(err.line, 2);
        // The same canonical triple through different spellings is a dup.
        let err =
            DestinationPolicy::parse_lines(["http://example.com:80", "HTTP://example.com:80"])
                .unwrap_err();
        assert!(err.reason.contains("duplicate"));
        // A port-only rule and a scheme+port rule are NOT duplicates (the
        // first is any-scheme, the second http-only); both coexist.
        let p =
            DestinationPolicy::parse_lines(["example.com:80", "http://example.com:80"]).unwrap();
        assert_eq!(p.rules().len(), 2);
        // Wildcard vs exact of the same suffix is NOT a duplicate.
        let p = DestinationPolicy::parse_lines(["example.com", "*.example.com"]).unwrap();
        assert_eq!(p.rules().len(), 2);
        // IPv4 canonicalization makes these duplicates too.
        let err = DestinationPolicy::parse_lines(["127.0.0.1", "127.000.0.001"]).unwrap_err();
        assert!(err.reason.contains("duplicate"));
        // IDN + its punycode form: duplicates.
        let err =
            DestinationPolicy::parse_lines(["exämple.com", "xn--exmple-cua.com"]).unwrap_err();
        assert!(err.reason.contains("duplicate"));
    }

    #[test]
    fn parse_lines_reports_the_line_number() {
        let err = DestinationPolicy::parse_lines(["example.com", "", "evil.com"]).unwrap_err();
        assert_eq!(err.line, 2);
        assert!(err.reason.contains("empty"));
    }

    #[test]
    fn empty_policy_parses_and_denies_everything() {
        let p = DestinationPolicy::parse_lines(Vec::<&str>::new()).unwrap();
        assert!(p.is_empty());
        let d = p.check("http", "example.com", 80, false, None);
        assert!(!d.is_allowed());
    }

    // ------------------------------------------------- decision semantics

    fn policy(lines: &[&str]) -> DestinationPolicy {
        DestinationPolicy::parse_lines(lines.iter().copied()).unwrap()
    }

    fn check(p: &DestinationPolicy, scheme: &str, host: &str, port: u16) -> Decision {
        p.check(scheme, host, port, false, None)
    }

    #[test]
    fn evil_suffix_can_never_ride_an_exact_rule() {
        let p = policy(&["example.com"]);
        // Regression (audit 36): prefix/substring semantics would allow
        // these; label-exact parsed semantics must deny every one.
        assert!(!check(&p, "https", "evil-example.com", 443).is_allowed());
        assert!(!check(&p, "https", "evil-example.com.evil", 443).is_allowed());
        assert!(!check(&p, "https", "notexample.com", 443).is_allowed());
        assert!(!check(&p, "https", "example.com.evil", 443).is_allowed());
        // The exact apex itself is allowed.
        assert!(check(&p, "https", "example.com", 443).is_allowed());
    }

    #[test]
    fn subdomain_of_allowlisted_domain_is_not_allowlisted() {
        let p = policy(&["example.com"]);
        assert!(!check(&p, "https", "api.example.com", 443).is_allowed());
        let p = policy(&["api.example.com"]);
        assert!(check(&p, "https", "api.example.com", 443).is_allowed());
        assert!(!check(&p, "https", "example.com", 443).is_allowed());
    }

    #[test]
    fn example_com_evil_only_when_literally_allowlisted() {
        let p = policy(&["example.com"]);
        assert!(!check(&p, "https", "example.com.evil", 443).is_allowed());
        let p = policy(&["example.com.evil"]);
        assert!(check(&p, "https", "example.com.evil", 443).is_allowed());
        assert!(!check(&p, "https", "example.com", 443).is_allowed());
        let p = policy(&["evil-example.com.evil"]);
        assert!(check(&p, "https", "evil-example.com.evil", 443).is_allowed());
    }

    #[test]
    fn trailing_dot_rule_equals_bare_rule_and_matches_plain_requests() {
        let p = policy(&["example.com."]);
        assert!(check(&p, "https", "example.com", 443).is_allowed());
        let p = policy(&["example.com"]);
        // Request-side trailing dot (only reachable via textual targets)
        // normalizes too.
        let t = RequestTarget::parse("https://example.com.:443/path").unwrap();
        assert!(t.check_against(&p).is_allowed());
        assert_eq!(t.host, "example.com");
    }

    #[test]
    fn idn_rule_matches_punycode_request_and_vice_versa() {
        // Rule written in unicode, request host on the wire is punycode.
        let p = policy(&["exämple.com"]);
        assert!(check(&p, "https", "xn--exmple-cua.com", 443).is_allowed());
        // Rule written as punycode, request side carries the IDN text.
        let p = policy(&["xn--exmple-cua.com"]);
        assert!(check(&p, "https", "exämple.com", 443).is_allowed());
        // Same for wildcard suffixes.
        let p = policy(&["*.exämple.com"]);
        assert!(check(&p, "https", "api.xn--exmple-cua.com", 443).is_allowed());
        assert!(!check(&p, "https", "api.example.com", 443).is_allowed());
        // A lookalike IDN never equals the ascii rule.
        let p = policy(&["example.com"]);
        assert!(!check(&p, "https", "exämple.com", 443).is_allowed());
        assert!(!check(&p, "https", "еxample.com", 443).is_allowed()); // cyrillic е
    }

    #[test]
    fn wildcard_matches_full_labels_only() {
        let p = policy(&["*.example.com"]);
        for (host, want) in [
            ("api.example.com", true),
            ("a.b.example.com", true),
            ("example.com", false),      // apex is not a subdomain
            ("evil-example.com", false), // label boundary matters
            ("example.com.evil", false),
            ("evil.example.com.evil", false),
            ("example.com.mx", false),
            ("xexample.com", false),
        ] {
            assert_eq!(check(&p, "https", host, 443).is_allowed(), want, "{host}");
        }
    }

    #[test]
    fn port_rule_semantics_explicit() {
        // `example.com` allows ANY port (rule port None = all ports).
        let p = policy(&["example.com"]);
        for port in [80u16, 443, 8080, 1, 65535] {
            assert!(
                check(&p, "https", "example.com", port).is_allowed(),
                "{port}"
            );
            assert!(
                check(&p, "http", "example.com", port).is_allowed(),
                "{port}"
            );
        }
        // `example.com:80` allows only port 80, on ANY scheme (a port-only
        // rule never restricts the scheme).
        let p = policy(&["example.com:80"]);
        assert!(check(&p, "http", "example.com", 80).is_allowed());
        assert!(check(&p, "https", "example.com", 80).is_allowed());
        assert!(!check(&p, "http", "example.com", 443).is_allowed());
        assert!(!check(&p, "https", "example.com", 8080).is_allowed());
        // A scheme+port rule restricts both dimensions.
        let p = policy(&["http://example.com:80"]);
        assert!(check(&p, "http", "example.com", 80).is_allowed());
        assert!(!check(&p, "https", "example.com", 80).is_allowed());
        assert!(!check(&p, "http", "example.com", 443).is_allowed());
    }

    #[test]
    fn scheme_mismatch_denied_when_rule_lists_other_scheme_only() {
        let p = policy(&["https://api.openai.com"]);
        // http request against an https-only rule: denied, and the reason
        // names the https rule (host+port matched, scheme missed).
        let d = check(&p, "http", "api.openai.com", 80);
        let reason = d.denied_reason().unwrap();
        assert_eq!(reason.rule_fired.as_deref(), Some("https://api.openai.com"));
        assert_eq!(reason.matched, RuleMatch::HostAndPort);
        assert!(check(&p, "https", "api.openai.com", 443).is_allowed());
        assert!(check(&p, "https", "api.openai.com", 8080).is_allowed());
        // ws against an https rule is also denied.
        assert!(!check(&p, "ws", "api.openai.com", 80).is_allowed());
        // The mirror: http-only rule denies https.
        let p = policy(&["http://api.openai.com"]);
        assert!(!check(&p, "https", "api.openai.com", 443).is_allowed());
        assert!(check(&p, "http", "api.openai.com", 80).is_allowed());
    }

    #[test]
    fn scheme_mismatch_with_port_rule_reports_host_only_match() {
        let p = policy(&["https://api.openai.com:443"]);
        let d = check(&p, "http", "api.openai.com", 80);
        let reason = d.denied_reason().unwrap();
        assert_eq!(
            reason.rule_fired.as_deref(),
            Some("https://api.openai.com:443")
        );
        assert_eq!(reason.matched, RuleMatch::Host);
    }

    #[test]
    fn denied_reason_is_plain_default_deny_when_host_matches_nothing() {
        let p = policy(&["https://api.openai.com"]);
        let d = check(&p, "https", "evil.com", 443);
        let reason = d.denied_reason().unwrap();
        assert_eq!(reason.rule_fired, None);
        assert_eq!(reason.matched, RuleMatch::None);
        let msg = reason.to_string();
        assert!(msg.contains("default-deny"), "{msg}");
    }

    #[test]
    fn longest_specificity_exact_beats_wildcard() {
        let p = policy(&["*.example.com:443", "https://example.com:443"]);
        // Both rules could match example.com:443; the exact rule is the
        // one reported when a sibling request is denied.
        assert!(check(&p, "https", "example.com", 443).is_allowed());
        // http to the apex: wildcard does not cover the apex; the exact
        // rule (scheme+port) is the reported candidate.
        let d = check(&p, "http", "example.com", 80);
        assert_eq!(
            d.denied_reason().unwrap().rule_fired.as_deref(),
            Some("https://example.com:443")
        );
        // Subdomain traffic on the allowed port is allowed by the wildcard.
        assert!(check(&p, "https", "api.example.com", 443).is_allowed());
        // A subdomain on the wrong port denies with the wildcard candidate.
        let d = check(&p, "https", "api.example.com", 21);
        assert!(d
            .denied_reason()
            .unwrap()
            .rule_fired
            .as_deref()
            .unwrap()
            .contains("*.example.com"));
        // An apex request with scheme/port both missing still attributes to
        // the exact rule (most specific host match first).
        let d = check(&p, "http", "example.com", 8080);
        assert_eq!(
            d.denied_reason().unwrap().rule_fired.as_deref(),
            Some("https://example.com:443")
        );
    }

    #[test]
    fn default_deny_reports_no_rule_when_policy_installed_but_empty() {
        let p = DestinationPolicy::empty();
        let d = check(&p, "https", "example.com", 443);
        assert_eq!(
            d,
            Decision::Denied(DeniedReason {
                rule_fired: None,
                matched: RuleMatch::None,
            })
        );
    }

    #[test]
    fn ipv4_matching_uses_octets_not_spellings() {
        let p = policy(&["http://127.0.0.1:9911"]);
        // Same octets, weird textual spelling (as some URL layers emit).
        assert!(p
            .check("http", "127.000.0.01", 9911, true, Some([127, 0, 0, 1]))
            .is_allowed());
        assert!(p
            .check("http", "127.0.0.1", 9911, true, Some([127, 0, 0, 1]))
            .is_allowed());
        // Wrong octets: denied even though the text shares a prefix.
        assert!(!p
            .check("http", "127.0.0.2", 9911, true, Some([127, 0, 0, 2]))
            .is_allowed());
        assert!(!p
            .check("http", "127.0.0.1", 9912, true, Some([127, 0, 0, 1]))
            .is_allowed());
        // Rule in leading-zero spelling canonicalizes to the same octets.
        let p = policy(&["http://127.000.0.001:9911"]);
        assert!(p
            .check("http", "127.0.0.1", 9911, true, Some([127, 0, 0, 1]))
            .is_allowed());
    }

    #[test]
    fn ipv6_matching_canonicalizes_both_sides() {
        let p = policy(&["http://[::1]:8080"]);
        assert!(p.check("http", "::1", 8080, false, None).is_allowed());
        // Full-form request text only arises from textual targets; parse()
        // canonicalizes the bracketed form before matching.
        let t = RequestTarget::parse("http://[0:0:0:0:0:0:0:1]:8080/x").unwrap();
        assert_eq!(t.host, "::1");
        assert!(t.check_against(&p).is_allowed());
        assert!(!p.check("http", "::2", 8080, false, None).is_allowed());
        assert!(!p.check("http", "::1", 8081, false, None).is_allowed());
        assert!(!p.check("https", "::1", 8080, false, None).is_allowed());
    }

    #[test]
    fn unknown_scheme_never_matches_any_rule() {
        let p = policy(&["example.com"]);
        // A bare-host textual target carries no scheme and no port: it is
        // ambiguous and can never match a rule (rules require a known wire
        // scheme), so an installed policy denies it.
        let t = RequestTarget::parse("example.com").unwrap();
        assert_eq!(t.scheme, "");
        assert_eq!(t.port, None);
        let d = t.check_against(&p);
        assert!(!d.is_allowed());
        assert_eq!(d.denied_reason().unwrap().matched, RuleMatch::Host);
        assert!(
            d.denied_reason().unwrap().rule_fired.is_some(),
            "the host matched the any-scheme rule but the request scheme is unknown"
        );
        // The same host WITH a scheme is allowlisted and allowed.
        let t = RequestTarget::parse("https://example.com").unwrap();
        assert!(t.check_against(&p).is_allowed());
        // check() with junk/unknown schemes denies.
        assert!(!p.check("", "example.com", 80, false, None).is_allowed());
        assert!(!p
            .check("gopher", "example.com", 70, false, None)
            .is_allowed());
    }

    #[test]
    fn request_target_parse_defaults_ports_and_rejects_junk() {
        let t = RequestTarget::parse("https://api.openai.com/v1/chat").unwrap();
        assert_eq!(t.scheme, "https");
        assert_eq!(t.host, "api.openai.com");
        assert_eq!(t.port, Some(443));
        let t = RequestTarget::parse("http://127.0.0.1:9911/x").unwrap();
        assert_eq!(t.port, Some(9911));
        assert!(t.is_ipv4);
        assert_eq!(t.ip, Some([127, 0, 0, 1]));
        assert_eq!(t.host, "127.0.0.1");
        // ws default port is 80, wss 443.
        assert_eq!(
            RequestTarget::parse("ws://example.com").unwrap().port,
            Some(80)
        );
        assert_eq!(
            RequestTarget::parse("wss://example.com").unwrap().port,
            Some(443)
        );
        for junk in [
            "",
            "   ",
            "not a url",
            "ftp://example.com",
            "http://",
            "http://:80",
            "http://exa mple.com",
            "http://user@example.com",
            "http://example.com:0",
            "http://example.com:70000",
            "example.com/path",
            "2001:db8::1",
        ] {
            assert!(RequestTarget::parse(junk).is_err(), "{junk:?} must fail");
        }
    }

    #[test]
    fn case_and_punycode_insensitive_request_targets() {
        let p = policy(&["https://EXAMPLE.com:443"]);
        for raw in [
            "https://example.com:443",
            "https://EXAMPLE.com:443",
            "https://example.com.",
        ] {
            let t = RequestTarget::parse(raw).unwrap();
            assert!(t.check_against(&p).is_allowed(), "{raw}");
        }
    }

    #[test]
    fn specificity_order_is_deterministic() {
        // Rules are stored most-specific first: exact host outranks
        // wildcard regardless of other dimensions; among equal host forms,
        // explicit port then explicit scheme decide.
        let p = policy(&[
            "example.com",
            "*.example.com:443",
            "https://example.com:443",
        ]);
        assert_eq!(p.rules()[0].text, "https://example.com:443");
        assert_eq!(p.rules()[1].text, "example.com");
        assert_eq!(p.rules()[2].text, "*.example.com:443");
    }
}
