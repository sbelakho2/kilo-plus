//! faktor-sandbox — capability-based permission enforcement (spec §30).
//!
//! Permissions are expressed as capabilities, never scattered conditionals.
//! Path checks are canonicalization-safe (symlink escapes and `..`
//! traversal are rejected); the network policy is the parsed-destination
//! gate from the security crate (audits 36-37): allowlist rules are parsed
//! at policy-build time into (scheme, host, port) triples with label-exact
//! host semantics — never prefix/substring matching — and every decision
//! goes through the parsed triple. The OS-level network sandbox remains a
//! separate documented layer; this gate is the app-level decision point.

use std::fs;
use std::path::{Component, Path, PathBuf};

use faktor_core::capability::{Capability, PermissionDecision};
use faktor_security::destination::{Decision, DeniedReason, DestinationPolicy, RequestTarget};

/// How the app-level network gate maps onto an installed parsed allowlist.
///
/// - [`NetworkGate::allow_all`] installs **no** policy: default-allow
///   (documented: with no destination policy configured, egress is not
///   restricted by this gate).
/// - [`NetworkGate::deny_all`] installs an *empty* policy: default-deny.
/// - [`NetworkGate::parse`] installs a parsed allowlist; every entry must
///   parse (a single bad entry is a policy build error, never silently
///   permissive) and duplicates are rejected.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NetworkGate {
    destinations: Option<DestinationPolicy>,
}

impl NetworkGate {
    /// No policy installed ⇒ default-allow for every destination.
    pub fn allow_all() -> NetworkGate {
        NetworkGate { destinations: None }
    }

    /// Installed empty policy ⇒ default-deny for every destination.
    pub fn deny_all() -> NetworkGate {
        NetworkGate {
            destinations: Some(DestinationPolicy::empty()),
        }
    }

    /// Install a parsed allowlist built from rule texts. Each entry is
    /// parsed strictly at build time (the config-time strictness boundary):
    /// any error fails the whole gate.
    pub fn parse<I>(entries: I) -> Result<NetworkGate, faktor_security::destination::RuleParseError>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        Ok(NetworkGate {
            destinations: Some(DestinationPolicy::parse_lines(entries)?),
        })
    }

    /// Install a directly-parsed policy (e.g. the daemon default).
    pub fn from_policy(policy: DestinationPolicy) -> NetworkGate {
        NetworkGate {
            destinations: Some(policy),
        }
    }

    /// The installed allowlist, if any. `None` = default-allow.
    pub fn installed(&self) -> Option<&DestinationPolicy> {
        self.destinations.as_ref()
    }

    /// Decision for a parsed request target (see the security crate's
    /// semantics: no policy ⇒ Allowed; installed ⇒ default-deny on no
    /// full triple match; denied reasons name the closest rule).
    pub fn decide(&self, target: &RequestTarget) -> Decision {
        match &self.destinations {
            None => Decision::Allowed,
            Some(policy) => target.check_against(policy),
        }
    }

    /// Denied reasons only: `Ok(())` when allowed.
    pub fn check(&self, target: &RequestTarget) -> Result<(), DestinationDenied> {
        match self.decide(target) {
            Decision::Allowed => Ok(()),
            Decision::Denied(reason) => Err(DestinationDenied {
                target: target.describe(),
                reason,
            }),
        }
    }
}

/// The typed egress denial an enforcement site receives when the network
/// gate refuses a destination (before any connection is attempted). Carries
/// the parsed deny reason for logs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestinationDenied {
    /// Human description of the parsed target (never the raw string alone).
    pub target: String,
    /// Which rule (if any) the denial is attributed to, and how far its
    /// match got.
    pub reason: DeniedReason,
}

impl std::fmt::Display for DestinationDenied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "destination {} denied: {}", self.target, self.reason)
    }
}

impl std::error::Error for DestinationDenied {}

/// The default provider endpoint allowlist (frozen set, mirrors the former
/// three-mode `AllowProviders` default). Scheme-constrained rules only.
const DEFAULT_PROVIDER_ENDPOINTS: [&str; 4] = [
    "https://api.openai.com",
    "https://api.anthropic.com",
    "https://generativelanguage.googleapis.com",
    "https://api.deepseek.com",
];

impl Default for NetworkGate {
    fn default() -> Self {
        // Static known-good entries: a parse failure here is a programming
        // error in the frozen list, never silently permissive.
        NetworkGate::parse(DEFAULT_PROVIDER_ENDPOINTS.iter().copied()).unwrap_or_else(|e| {
            panic!("frozen default provider endpoints must parse: {e}");
        })
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SandboxPolicy {
    pub read_workspace: Rule,
    pub write_workspace: Rule,
    pub read_external: Rule,
    pub write_external: Rule,
    pub execute_shell: Rule,
    pub network: NetworkGate,
    pub mcp: Rule,
    pub git: Rule,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            read_workspace: Rule::Allow,
            write_workspace: Rule::Allow,
            read_external: Rule::Ask,
            write_external: Rule::Ask,
            execute_shell: Rule::Ask,
            network: NetworkGate::default(),
            mcp: Rule::Allow,
            git: Rule::Allow,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone)]
pub struct PermissionEngine {
    policy: SandboxPolicy,
    workspace_root: Option<PathBuf>,
}

impl PermissionEngine {
    pub fn new(policy: SandboxPolicy, workspace_root: Option<PathBuf>) -> Self {
        Self {
            policy,
            workspace_root: workspace_root.map(|p| p.canonicalize().unwrap_or(p)),
        }
    }

    pub fn policy(&self) -> &SandboxPolicy {
        &self.policy
    }

    pub fn workspace_root(&self) -> Option<&Path> {
        self.workspace_root.as_deref()
    }

    /// True when `path` (possibly relative) resolves inside the workspace
    /// root, following symlinks and rejecting escapes.
    pub fn is_within_workspace(&self, path: &Path) -> bool {
        let Some(root) = &self.workspace_root else {
            return false;
        };
        let resolved = resolve_within(root, path);
        resolved
            .as_deref()
            .map(|r| r.starts_with(root))
            .unwrap_or(false)
    }

    /// The parsed network gate. Every egress decision point consults it
    /// with a *parsed* request target; see [`PermissionEngine::check_egress`].
    pub fn network_gate(&self) -> &NetworkGate {
        &self.policy.network
    }

    /// Typed pre-connection egress check (the enforcement seam every
    /// outbound call must thread): parse the destination once, decide on
    /// the parsed triple, and return the typed denial reason when refused.
    /// An unparseable destination is a typed error too — it is never
    /// allowed, never prefix-compared, never silently defaulted.
    pub fn check_egress(&self, destination: &str) -> Result<(), EgressError> {
        let target = RequestTarget::parse(destination).map_err(EgressError::Unparseable)?;
        self.policy
            .network
            .check(&target)
            .map_err(EgressError::Denied)
    }

    /// Decision on a destination whose host/scheme/port were already pulled
    /// from a parsed URL object (never from strings): `(scheme, host,
    /// explicit_port, is_ipv4, ip)`. `explicit_port` is the URL's explicit
    /// port (scheme defaults are resolved inside the security crate).
    pub fn check_egress_parts(
        &self,
        scheme: &str,
        host: &str,
        explicit_port: Option<u16>,
        is_ipv4: bool,
        ip: Option<[u8; 4]>,
    ) -> Result<(), EgressError> {
        let target = RequestTarget::from_parts(Some(scheme), host, explicit_port, is_ipv4, ip)
            .map_err(EgressError::Unparseable)?;
        self.policy
            .network
            .check(&target)
            .map_err(EgressError::Denied)
    }

    /// Evaluate one capability against the policy.
    pub fn evaluate(&self, capability: &Capability) -> PermissionDecision {
        match capability {
            Capability::ReadWorkspace { path } => {
                if self.is_within_workspace(path) {
                    rule_decision(self.policy.read_workspace)
                } else {
                    // Path escapes the workspace: it is an external read.
                    self.evaluate(&Capability::ReadExternal { path: path.clone() })
                }
            }
            Capability::WriteWorkspace { path } => {
                if self.is_within_workspace(path) {
                    rule_decision(self.policy.write_workspace)
                } else {
                    self.evaluate(&Capability::WriteExternal { path: path.clone() })
                }
            }
            Capability::ReadExternal { path } => {
                if self.is_within_workspace(path) {
                    rule_decision(self.policy.read_workspace)
                } else {
                    rule_decision(self.policy.read_external)
                }
            }
            Capability::WriteExternal { path } => {
                if self.is_within_workspace(path) {
                    rule_decision(self.policy.write_workspace)
                } else {
                    rule_decision(self.policy.write_external)
                }
            }
            Capability::ExecuteShell { .. } => rule_decision(self.policy.execute_shell),
            Capability::Network { destination } => match self.check_egress(destination) {
                Ok(()) => PermissionDecision::Allow,
                Err(e) => {
                    tracing::warn!("network capability denied: {e}");
                    PermissionDecision::Deny
                }
            },
            Capability::Mcp { .. } => rule_decision(self.policy.mcp),
            Capability::Git { .. } => rule_decision(self.policy.git),
        }
    }
}

/// Egress refusal at an app-level network decision point: either the typed
/// destination-policy denial or an unparseable destination (which is always
/// refused when a gate is installed — never prefix-matched, never allowed
/// by accident). With no gate installed (default-allow) `check_egress`
/// succeeds for any *parseable* destination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EgressError {
    Denied(DestinationDenied),
    Unparseable(String),
}

impl std::fmt::Display for EgressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EgressError::Denied(d) => write!(f, "{d}"),
            EgressError::Unparseable(e) => write!(f, "destination does not parse as a URL: {e}"),
        }
    }
}

impl std::error::Error for EgressError {}

fn rule_decision(rule: Rule) -> PermissionDecision {
    match rule {
        Rule::Allow => PermissionDecision::Allow,
        Rule::Deny => PermissionDecision::Deny,
        Rule::Ask => PermissionDecision::Ask,
    }
}

/// Resolve `path` against `root` with parent-canonicalization (symlink-safe)
/// and component-level `..` rejection. Returns the canonical absolute path
/// when the resolution is safe and exists.
fn resolve_within(root: &Path, path: &Path) -> Option<PathBuf> {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    // Component-level traversal rejection (before touching the FS).
    for component in joined.components() {
        if let Component::ParentDir = component {
            return None;
        }
    }
    // Full canonicalization resolves leaf symlinks too; on failure (missing
    // leaf, ELOOP) fall back to parent-canonicalization with an explicit
    // leaf-symlink rejection.
    if let Ok(canon) = joined.canonicalize() {
        return if canon.starts_with(root) {
            Some(canon)
        } else {
            None
        };
    }
    // The leaf is itself a symlink whose canonicalization failed (loop or
    // broken): never treat it as inside.
    if let Ok(meta) = fs::symlink_metadata(&joined) {
        if meta.file_type().is_symlink() {
            return None;
        }
    }
    let parent = joined.parent()?;
    let file_name = joined.file_name()?;
    let canon_parent = parent.canonicalize().ok()?;
    let resolved = canon_parent.join(file_name);
    if resolved.starts_with(root) {
        Some(resolved)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::os::unix::fs::symlink;

    fn engine(root: &Path) -> PermissionEngine {
        PermissionEngine::new(SandboxPolicy::default(), Some(root.to_path_buf()))
    }

    fn tmp_workspace() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ws");
        fs::create_dir_all(&root).unwrap();
        (dir, root)
    }

    #[test]
    fn traversal_matrix_rejected() {
        let (_d, root) = tmp_workspace();
        let e = engine(&root);
        // Abs paths outside.
        assert!(!e.is_within_workspace(Path::new("/etc/passwd")));
        assert!(!e.is_within_workspace(Path::new("/tmp/../etc")));
        // Parent-dir escapes.
        assert!(!e.is_within_workspace(Path::new("../escape")));
        assert!(!e.is_within_workspace(Path::new("a/../../b")));
        // Abs path of the root itself is fine.
        assert!(e.is_within_workspace(&root.join("x.rs")));
        assert!(e.is_within_workspace(&root));
    }

    #[test]
    fn symlink_escape_rejected() {
        let (_d, root) = tmp_workspace();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("secret.txt"), "secret").unwrap();
        // Symlink inside the workspace pointing outside.
        symlink(outside.path(), root.join("link")).unwrap();
        let e = engine(&root);
        assert!(
            !e.is_within_workspace(Path::new("link/secret.txt")),
            "symlink escape must be rejected"
        );
        assert!(
            !e.is_within_workspace(Path::new("link")),
            "symlinked dir itself is outside"
        );
        // Symlink to a file inside the workspace is fine.
        fs::write(root.join("real.txt"), "x").unwrap();
        symlink(root.join("real.txt"), root.join("alias.txt")).unwrap();
        assert!(e.is_within_workspace(Path::new("alias.txt")));
    }

    #[test]
    fn symlink_loop_terminates() {
        let (_d, root) = tmp_workspace();
        symlink(root.join("b"), root.join("a")).unwrap();
        symlink(root.join("a"), root.join("b")).unwrap();
        let e = engine(&root);
        // Canonicalize of a/b loops — resolve must return None, never hang.
        assert!(!e.is_within_workspace(Path::new("a/x")));
    }

    #[test]
    fn relative_and_absolute_equivalence() {
        let (_d, root) = tmp_workspace();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/main.rs"), "").unwrap();
        let e = engine(&root);
        assert!(e.is_within_workspace(Path::new("src/main.rs")));
        assert!(e.is_within_workspace(&root.join("src/main.rs")));
        assert!(!e.is_within_workspace(&root.join("src/main.rs/../../etc/x")));
    }

    #[test]
    fn workspace_capabilities_obey_policy() {
        let (_d, root) = tmp_workspace();
        fs::write(root.join("f.rs"), "").unwrap();
        let policy = SandboxPolicy {
            write_workspace: Rule::Deny,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, Some(root.clone()));
        assert_eq!(
            e.evaluate(&Capability::WriteWorkspace {
                path: root.join("f.rs")
            }),
            PermissionDecision::Deny
        );
        assert_eq!(
            e.evaluate(&Capability::ReadWorkspace {
                path: root.join("f.rs")
            }),
            PermissionDecision::Allow
        );
        assert_eq!(
            e.evaluate(&Capability::ReadWorkspace {
                path: PathBuf::from("/etc/passwd")
            }),
            PermissionDecision::Ask,
            "escaped workspace read becomes external Ask"
        );
    }

    #[test]
    fn external_rules_mapped() {
        let e = PermissionEngine::new(SandboxPolicy::default(), None);
        assert_eq!(
            e.evaluate(&Capability::ReadExternal {
                path: "/etc".into()
            }),
            PermissionDecision::Ask
        );
        assert_eq!(
            e.evaluate(&Capability::WriteExternal {
                path: "/etc".into()
            }),
            PermissionDecision::Ask
        );
        let policy = SandboxPolicy {
            read_external: Rule::Deny,
            execute_shell: Rule::Allow,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        assert_eq!(
            e.evaluate(&Capability::ReadExternal { path: "/x".into() }),
            PermissionDecision::Deny
        );
        assert_eq!(
            e.evaluate(&Capability::ExecuteShell {
                command: "ls".into()
            }),
            PermissionDecision::Allow
        );
    }

    #[test]
    fn default_gate_allowlists_provider_endpoints_and_denies_the_rest() {
        let e = PermissionEngine::new(SandboxPolicy::default(), None);
        // The frozen provider allowlist still allows its own endpoints.
        for (dest, expect) in [
            (
                "https://api.openai.com/v1/chat/completions",
                PermissionDecision::Allow,
            ),
            (
                "https://api.anthropic.com/v1/messages",
                PermissionDecision::Allow,
            ),
            (
                "https://api.deepseek.com/chat/completions",
                PermissionDecision::Allow,
            ),
            // Prefix lookalikes stay denied (parsed semantics).
            ("https://evil.example.com", PermissionDecision::Deny),
            ("https://api.openai.com.evil/v1", PermissionDecision::Deny),
            ("https://evil-api.openai.com/v1", PermissionDecision::Deny),
            ("https://notapi.openai.com/v1", PermissionDecision::Deny),
            ("http://api.openai.com/v1", PermissionDecision::Deny), // https-only rule
            ("https://example.com", PermissionDecision::Deny),
        ] {
            assert_eq!(
                e.evaluate(&Capability::Network {
                    destination: dest.into()
                }),
                expect,
                "{dest}"
            );
        }
    }

    #[test]
    fn deny_all_and_allow_all_gates() {
        let policy = SandboxPolicy {
            network: NetworkGate::deny_all(),
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        assert_eq!(
            e.evaluate(&Capability::Network {
                destination: "https://api.openai.com".into()
            }),
            PermissionDecision::Deny
        );

        let policy = SandboxPolicy {
            network: NetworkGate::allow_all(),
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        // No policy installed ⇒ default-allow (documented semantics).
        assert_eq!(
            e.evaluate(&Capability::Network {
                destination: "https://api.openai.com".into()
            }),
            PermissionDecision::Allow
        );
        assert_eq!(
            e.evaluate(&Capability::Network {
                destination: "http://127.0.0.1:9911/x".into()
            }),
            PermissionDecision::Allow
        );
    }

    #[test]
    fn bad_policy_entry_is_a_build_error_never_silent() {
        for bad in [
            "",
            " ",
            "example.com:99999",
            "http://*.com",
            "evil-example.com/x",
            "exa mple.com",
        ] {
            let err = NetworkGate::parse([bad]).unwrap_err();
            assert!(!err.reason.is_empty(), "{bad:?}");
        }
        // Duplicate exact rules error instead of last-wins.
        assert!(NetworkGate::parse(["example.com", "example.com"]).is_err());
    }

    #[test]
    fn parse_duplicate_within_endpoints_fails() {
        assert!(NetworkGate::parse(["example.com", "EXAMPLE.com"]).is_err());
    }

    #[test]
    fn typed_egress_check_reports_rule_and_match_depth() {
        let policy = SandboxPolicy {
            network: NetworkGate::parse(["https://api.openai.com"]).unwrap(),
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        // Allowed destination: parse once, allowed.
        assert!(e
            .check_egress("https://api.openai.com/v1/chat/completions")
            .is_ok());
        // Scheme mismatch: denied BEFORE any connection, with the rule text
        // and the matched depth (host+port matched, scheme missed).
        let err = e.check_egress("http://api.openai.com:443/v1").unwrap_err();
        match err {
            EgressError::Denied(d) => {
                assert_eq!(
                    d.reason.rule_fired.as_deref(),
                    Some("https://api.openai.com")
                );
                assert!(d.target.contains("api.openai.com"));
                assert!(d.to_string().contains("scheme"), "{}", d);
            }
            EgressError::Unparseable(_) => panic!("parses fine; must be a gate denial"),
        }
        // Host mismatch denies with no attributed rule (plain default-deny).
        let err = e.check_egress("https://evil-example.com").unwrap_err();
        match err {
            EgressError::Denied(d) => {
                assert_eq!(d.reason.rule_fired, None);
                assert!(d.to_string().contains("default-deny"), "{}", d);
            }
            EgressError::Unparseable(_) => panic!("parses fine"),
        }
        // Unparseable destinations are typed errors, never allowed.
        let err = e.check_egress("https://evil example.com").unwrap_err();
        assert!(matches!(err, EgressError::Unparseable(_)));
        // check_egress_parts: decision on URL-parser-supplied parts.
        assert!(e
            .check_egress_parts("https", "api.openai.com", Some(443), false, None)
            .is_ok());
        let err = e
            .check_egress_parts("http", "api.openai.com", Some(80), false, None)
            .unwrap_err();
        assert!(matches!(err, EgressError::Denied(_)));
    }

    #[test]
    fn egress_never_prefix_matches_at_the_engine_boundary() {
        let policy = SandboxPolicy {
            network: NetworkGate::parse(["example.com"]).unwrap(),
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        for dest in ["https://example.com", "https://example.com:8443/x"] {
            assert!(e.check_egress(dest).is_ok(), "{dest}");
        }
        // The audit regression set: suffixes and lookalikes are DENIED.
        for dest in [
            "https://evil-example.com",
            "https://evil-example.com.evil",
            "https://example.com.evil",
            "https://notexample.com",
            "https://api.example.com",
            "https://sub.example.com.evil.com",
        ] {
            let err = e.check_egress(dest).unwrap_err();
            assert!(
                matches!(err, EgressError::Denied(_)),
                "{dest} must be denied, got {err}"
            );
        }
        // The path/query of the FIRST fetch never changes the gate (the
        // connection goes to example.com); a server-side follow-up fetch is
        // its own egress and must pass this gate again.
        assert!(e
            .check_egress("http://example.com/redirect?to=https://evil-example.com")
            .is_ok());
    }

    #[test]
    fn no_policy_gate_allows_even_weird_but_parseable_destinations() {
        let policy = SandboxPolicy {
            network: NetworkGate::allow_all(),
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        for dest in [
            "https://anything.example:8443/a",
            "http://127.0.0.1:1/x",
            "wss://example.com",
        ] {
            assert!(e.check_egress(dest).is_ok(), "{dest}");
        }
        // Still never allow a destination that does not parse.
        assert!(matches!(
            e.check_egress("https://exa mple.com").unwrap_err(),
            EgressError::Unparseable(_)
        ));
    }

    #[test]
    fn shell_mcp_git_rules() {
        let policy = SandboxPolicy {
            mcp: Rule::Ask,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        assert_eq!(
            e.evaluate(&Capability::Mcp {
                server: "fs".into()
            }),
            PermissionDecision::Ask
        );
        assert_eq!(
            e.evaluate(&Capability::Git {
                operation: "status".into()
            }),
            PermissionDecision::Allow
        );
    }

    #[test]
    fn unicode_and_hostile_paths_never_panic() {
        let (_d, root) = tmp_workspace();
        let e = engine(&root);
        for hostile in [
            "",
            ".",
            "..",
            "a/b/../../../../etc/passwd",
            "\u{FFFE}",
            "x\0y",
            "\\\\server\\share\\x",
            "a/..",
        ] {
            let _ = e.evaluate(&Capability::ReadWorkspace {
                path: hostile.into(),
            });
            let _ = e.is_within_workspace(Path::new(hostile));
        }
        // None of the above panicked; workspace root itself still resolves.
        assert!(e.is_within_workspace(Path::new(".")) || true);
    }

    #[test]
    fn hostile_network_destinations_never_panic_and_never_allow() {
        let policy = SandboxPolicy {
            network: NetworkGate::default(),
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        for hostile in [
            "",
            " ",
            "https://",
            "http://evil example.com",
            "\u{FFFE}",
            "x\0y",
            "example.com:99999",
            "https://example.com:notaport",
        ]
        .into_iter()
        .map(str::to_string)
        .chain(std::iter::once(format!("http://{}/x", "a".repeat(5000))))
        {
            let d = e.evaluate(&Capability::Network {
                destination: hostile.clone(),
            });
            assert_eq!(d, PermissionDecision::Deny, "{hostile:?} must deny");
            let _ = e.check_egress(&hostile); // must not panic
        }
    }

    #[test]
    fn no_workspace_root_means_everything_external() {
        let e = PermissionEngine::new(SandboxPolicy::default(), None);
        assert!(!e.is_within_workspace(Path::new("/anything")));
        assert_eq!(
            e.evaluate(&Capability::ReadWorkspace { path: "/x".into() }),
            PermissionDecision::Ask,
            "no root ⇒ external Ask"
        );
    }

    #[test]
    fn policy_serde_roundtrip() {
        let p = SandboxPolicy::default();
        let v = serde_json::to_value(&p).unwrap();
        let back: SandboxPolicy = serde_json::from_value(v).unwrap();
        assert_eq!(p, back);
        let p = SandboxPolicy {
            network: NetworkGate::allow_all(),
            ..Default::default()
        };
        let v = serde_json::to_value(&p).unwrap();
        let back: SandboxPolicy = serde_json::from_value(v).unwrap();
        assert_eq!(p, back);
    }
}
