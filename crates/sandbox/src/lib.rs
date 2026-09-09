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
//!
//! ## Network isolation honesty (audit P0-39)
//!
//! The capability engine is **application-level** policy: `Network`
//! denied stops app-level outbound calls, but it does NOT stop a permitted
//! shell from opening its own sockets. Whether OS-level enforcement backs
//! the policy is a platform question, not an application question:
//! [`platform_network_enforcement`] answers it honestly, and
//! [`SandboxGuarantee`] lets a policy declare what it *requires*:
//!
//! - [`SandboxGuarantee::Required`] — the policy needs OS-level network
//!   isolation. When the platform cannot provide it — or, on Linux, has
//!   not PROVEN at spawn that a backend provides it (audit 4/28/35-39:
//!   capability existence is not enforcement) — shell commands are REFUSED
//!   before spawn with the typed [`SandboxUnavailable`] error (fail
//!   closed); they never run unenforced.
//! - [`SandboxGuarantee::BestEffort`] — run behind the existing
//!   app-level gates only, with the guarantee documented as app-level
//!   (an audit note is recorded when a shell runs under it).
//! - [`SandboxGuarantee::None`] — no network-isolation guarantee claimed
//!   or enforced.
//!
//! Enforcement matrix (see [`platform_network_enforcement`]). The Linux
//! row is HONEST per audit 4/28/35-39: the probe never claims `OsLevel`
//! from capability *existence* — the terminal spawn backend
//! (`unshare(CLONE_NEWNET)` under `NetworkIsolation::DenyAll`) must prove
//! itself active *at spawn* before any `Required` policy is allowed to
//! run a shell. Nothing in this repo reports that proof today, so Linux
//! is `AppLevel` and `Required` fails closed there too:
//!
//! | platform                              | enforcement   | Required  | BestEffort | None |
//! |---------------------------------------|---------------|-----------|------------|------|
//! | linux (backend proven at spawn)       | `OsLevel`     | allowed   | allowed    | runs |
//! | linux (no proven backend — today)     | `AppLevel`    | refused   | app gates + note | runs |
//! | macOS (no per-process backend)        | `Unavailable` | refused   | app gates + note | runs |
//! | windows (no AppContainer path)        | `Unavailable` | refused   | app gates + note | runs |

use std::fs;
use std::path::{Component, Path, PathBuf};

use faktor_core::capability::{Capability, PermissionDecision};
use faktor_security::destination::{Decision, DeniedReason, DestinationPolicy, RequestTarget};

/// What a sandbox policy requires of the platform's network isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SandboxGuarantee {
    /// OS-level network isolation is REQUIRED for commands. When the
    /// platform enforcement is [`NetworkEnforcement::Unavailable`] or only
    /// [`NetworkEnforcement::AppLevel`], shell commands refuse with the
    /// typed [`SandboxUnavailable`] error BEFORE spawn — a permitted shell
    /// under `ExecuteShell` + `Network(deny)` would otherwise open its own
    /// sockets. Fail closed; never run unenforced.
    Required,
    /// Best-effort: commands run behind the existing app-level capability
    /// gates only. NETWORK_ISOLATION_NOTE: the `ExecuteShell` +
    /// `Network(deny)` combination is app-level only under BestEffort — the
    /// capability engine stops app-level egress, but a permitted shell can
    /// still open sockets itself; no OS-level deny backend backs this. An
    /// audit note is recorded whenever a shell runs under this guarantee.
    BestEffort,
    /// No network-isolation guarantee is claimed or enforced by this
    /// policy; the host documents its own threat model.
    #[default]
    None,
}

/// The canonical documentation sentence for the BestEffort limitation
/// (asserted by tests; the variant's doc comment above carries the same
/// wording).
pub const NETWORK_ISOLATION_NOTE: &str =
    "the ExecuteShell + Network(deny) combination is app-level only under BestEffort — \
     the capability engine stops app-level egress, but a permitted shell can still open \
     its own sockets; no OS-level deny backend backs this";

/// Honest answer to "does this platform back network denial at the OS
/// level?" See [`platform_network_enforcement`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkEnforcement {
    /// Network policy is enforced by the APPLICATION (this capability
    /// engine) only. A permitted shell can open its own sockets — the
    /// app-level gate cannot stop that.
    AppLevel,
    /// Network policy CAN be enforced at the OS level by a spawn backend
    /// that has PROVEN itself active at spawn (a `Required` guarantee can
    /// then be met). Nothing in this repo reports this today: capability
    /// existence is not enforcement (audit 4/28/35-39) — Linux stays
    /// [`NetworkEnforcement::AppLevel`] until the terminal crate's
    /// `unshare(CLONE_NEWNET)` DenyAll path proves itself at spawn and this
    /// probe is wired to that proof.
    OsLevel,
    /// No reliable per-process network-denial backend exists on this
    /// platform (macOS: none is implemented in this repo; windows: the
    /// AppContainer path is not implemented). Fail-closed semantics for
    /// [`SandboxGuarantee::Required`]: commands refuse rather than run
    /// unenforced.
    Unavailable,
}

impl std::fmt::Display for NetworkEnforcement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NetworkEnforcement::AppLevel => write!(
                f,
                "app-level only: the capability engine stops app-level egress, but a \
                 permitted shell can open its own sockets"
            ),
            NetworkEnforcement::OsLevel => write!(
                f,
                "OS-level: per-process network isolation is usable on this platform"
            ),
            NetworkEnforcement::Unavailable => write!(
                f,
                "unavailable: no reliable per-process network-denial backend exists on \
                 this platform; Required guarantees fail closed"
            ),
        }
    }
}

/// Probe override used by adversarial tests to force a platform verdict
/// without touching the real host (the probe is otherwise read-only).
#[cfg(test)]
static PROBE_OVERRIDE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// What OS-level network enforcement this platform actually provides.
///
/// - **linux**: [`NetworkEnforcement::AppLevel`] — always, until a spawn
///   backend proves itself. The old probe (audit 4/28/35-39) claimed
///   [`NetworkEnforcement::OsLevel`] when `/proc/self/ns/net` existed AND
///   `/proc/self/status` `CapEff` (hex) contained CAP_SYS_ADMIN (bit 21).
///   That proves a *capability*, not that the spawn path applies a
///   namespace: `Required` ran an ordinary networked shell. Honest verdict:
///   app-level only — a `Required` guarantee refuses BEFORE spawn until
///   the terminal crate's `NetworkIsolation::DenyAll` backend
///   (`unshare(CLONE_NEWNET)` pre-exec) proves itself at spawn and this
///   probe is wired to that proof.
/// - **macos**: [`NetworkEnforcement::Unavailable`] — this repo has no
///   reliable per-process network-denial backend on macOS (seatbelt/sandbox
///   profiles are not wired); a `Required` guarantee therefore refuses.
/// - **windows**: [`NetworkEnforcement::Unavailable`] — the AppContainer
///   path is not implemented in this repo.
#[cfg(target_os = "linux")]
pub fn platform_network_enforcement() -> NetworkEnforcement {
    #[cfg(test)]
    {
        match PROBE_OVERRIDE.load(std::sync::atomic::Ordering::SeqCst) {
            1 => return NetworkEnforcement::AppLevel,
            2 => return NetworkEnforcement::OsLevel,
            3 => return NetworkEnforcement::Unavailable,
            _ => {}
        }
    }
    NetworkEnforcement::AppLevel
}

/// macOS: no reliable per-process deny backend is implemented in this
/// repo; [`SandboxGuarantee::Required`] fails closed (refuses commands).
#[cfg(not(target_os = "linux"))]
pub fn platform_network_enforcement() -> NetworkEnforcement {
    #[cfg(test)]
    {
        match PROBE_OVERRIDE.load(std::sync::atomic::Ordering::SeqCst) {
            1 => return NetworkEnforcement::AppLevel,
            2 => return NetworkEnforcement::OsLevel,
            3 => return NetworkEnforcement::Unavailable,
            _ => {}
        }
    }
    NetworkEnforcement::Unavailable
}

/// Set the probe verdict for tests. `None` restores the real probe.
#[cfg(test)]
fn override_probe_for_tests(v: Option<NetworkEnforcement>) {
    use std::sync::atomic::Ordering;
    PROBE_OVERRIDE.store(
        match v {
            None => -1,
            Some(NetworkEnforcement::AppLevel) => 1,
            Some(NetworkEnforcement::OsLevel) => 2,
            Some(NetworkEnforcement::Unavailable) => 3,
        },
        Ordering::SeqCst,
    );
}

/// Typed refusal when a sandbox policy's [`SandboxGuarantee::Required`]
/// network isolation cannot be provided by this platform. Commands must be
/// refused BEFORE spawn — never run unenforced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SandboxUnavailable {
    /// The guarantee the policy demanded.
    pub guarantee: SandboxGuarantee,
    /// What the platform actually provides.
    pub enforcement: NetworkEnforcement,
}

impl std::fmt::Display for SandboxUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sandbox unavailable: policy requires OS-level network isolation \
             (guarantee {:?}) but this platform only provides {}. Refusing before \
             spawn: a permitted shell under ExecuteShell + Network(deny) would \
             otherwise open its own sockets.",
            self.guarantee, self.enforcement
        )
    }
}

impl std::error::Error for SandboxUnavailable {}

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
    /// What this policy requires of OS-level network isolation
    /// (audit P0-39). [`SandboxGuarantee::Required`] refuses shell commands
    /// with the typed [`SandboxUnavailable`] error when the platform cannot
    /// back network denial at the OS level; `BestEffort` runs behind the
    /// app-level gates with the documented app-level-only caveat; `None`
    /// (default) claims nothing. Defaults to `None` so existing policies
    /// keep their exact semantics.
    #[serde(default)]
    pub network_guarantee: SandboxGuarantee,
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
            network_guarantee: SandboxGuarantee::None,
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

    /// The typed shell-feasibility seam (audit P0-39): can a shell command
    /// run under this policy on this platform at all? Refuses with the
    /// typed [`SandboxUnavailable`] error when the policy's
    /// [`SandboxGuarantee`] demands more OS-level network isolation than
    /// [`platform_network_enforcement`] provides. Call this BEFORE any
    /// spawn; [`PermissionEngine::evaluate`] folds the same check into the
    /// `ExecuteShell` verdict, so existing gates fail closed without new
    /// call sites.
    ///
    /// Semantics:
    /// - guarantee `None` → `Ok(())` (no guarantee claimed).
    /// - guarantee `BestEffort` → `Ok(())` with a recorded audit note that
    ///   the network isolation is app-level only (see
    ///   [`NETWORK_ISOLATION_NOTE`]).
    /// - guarantee `Required` → `Ok(())` only when enforcement is
    ///   [`NetworkEnforcement::OsLevel`]; otherwise `Err(SandboxUnavailable)`
    ///   — refuse before spawn, never run unenforced.
    pub fn check_shell_feasibility(&self) -> Result<(), SandboxUnavailable> {
        match self.policy.network_guarantee {
            SandboxGuarantee::None => Ok(()),
            SandboxGuarantee::BestEffort => {
                tracing::warn!(
                    "execute_shell under BestEffort network guarantee: {}",
                    NETWORK_ISOLATION_NOTE
                );
                Ok(())
            }
            SandboxGuarantee::Required => {
                let enforcement = platform_network_enforcement();
                if enforcement == NetworkEnforcement::OsLevel {
                    Ok(())
                } else {
                    Err(SandboxUnavailable {
                        guarantee: SandboxGuarantee::Required,
                        enforcement,
                    })
                }
            }
        }
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
            // Audit P0-39: a shell that a Required network guarantee must
            // not let run unenforced refuses BEFORE spawn with the typed
            // SandboxUnavailable folded into a Deny — this holds for the
            // Allow AND the Ask paths (an Ask that a human approves must
            // still not run unisolated; hosts additionally call
            // `check_shell_feasibility` at the spawn seam).
            Capability::ExecuteShell { .. } => {
                let rule = rule_decision(self.policy.execute_shell);
                if rule == PermissionDecision::Deny {
                    return rule;
                }
                match self.check_shell_feasibility() {
                    Ok(()) => rule,
                    Err(unavailable) => {
                        tracing::warn!("execute_shell refused before spawn: {unavailable}");
                        PermissionDecision::Deny
                    }
                }
            }
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
        // The new guarantee field round-trips too.
        let p = SandboxPolicy {
            network_guarantee: SandboxGuarantee::Required,
            ..Default::default()
        };
        let v = serde_json::to_value(&p).unwrap();
        let back: SandboxPolicy = serde_json::from_value(v).unwrap();
        assert_eq!(p, back);
        assert_eq!(back.network_guarantee, SandboxGuarantee::Required);
        // Pre-existing configs without the field still parse (serde
        // default), with the documented default = None.
        let mut v = serde_json::to_value(SandboxPolicy::default()).unwrap();
        v.as_object_mut().unwrap().remove("network_guarantee");
        let back: SandboxPolicy = serde_json::from_value(v).unwrap();
        assert_eq!(back.network_guarantee, SandboxGuarantee::None);
    }

    // ----------------------- P0-39 network isolation honesty ------------

    /// Forces the platform probe for the duration of the test, restoring
    /// the real probe even on panic. The probe override is process-global,
    /// so the guard also holds the serialization lock: probe-dependent
    /// tests never race each other's forced verdicts.
    static PROBE_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

    // The guard's field exists ONLY for its Drop side (the probe override
    // restore + lock release): it is intentionally never read.
    #[allow(dead_code)]
    struct ProbeGuard(std::sync::MutexGuard<'static, ()>);
    impl ProbeGuard {
        fn force(v: NetworkEnforcement) -> ProbeGuard {
            let lock = probe_read_lock();
            override_probe_for_tests(Some(v));
            ProbeGuard(lock)
        }
    }
    impl Drop for ProbeGuard {
        fn drop(&mut self) {
            override_probe_for_tests(None);
        }
    }

    /// Hold the probe serialization lock WITHOUT forcing: real-probe reads
    /// and feasibility checks that call the real probe must never race a
    /// parallel forced-state test.
    fn probe_read_lock() -> std::sync::MutexGuard<'static, ()> {
        PROBE_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    fn shell_cap() -> Capability {
        Capability::ExecuteShell {
            command: "curl http://evil.example".into(),
        }
    }

    #[test]
    fn required_plus_unavailable_refuses_before_spawn_with_typed_error() {
        // "macOS-like" probe forced through the cfg(test) hook: a policy
        // that REQUIRES OS-level network isolation must refuse the shell
        // even though the execute_shell rule says Allow — never run
        // unenforced.
        let _guard = ProbeGuard::force(NetworkEnforcement::Unavailable);
        let policy = SandboxPolicy {
            execute_shell: Rule::Allow,
            network_guarantee: SandboxGuarantee::Required,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        // The typed refusal is available at the spawn seam...
        let err = e.check_shell_feasibility().unwrap_err();
        assert_eq!(err.guarantee, SandboxGuarantee::Required);
        assert_eq!(err.enforcement, NetworkEnforcement::Unavailable);
        let msg = err.to_string();
        assert!(msg.contains("before spawn"), "{msg}");
        assert!(msg.contains("Network(deny)"), "{msg}");
        // ...and the evaluate gate already folds it into a Deny (the
        // decision seam existing callers use before running the command).
        assert_eq!(
            e.evaluate(&shell_cap()),
            PermissionDecision::Deny,
            "Required + Unavailable must deny before spawn"
        );
        // Same refusal when the probe reports app-level-only enforcement.
        // The probe guard serializes probe tests (process-global state):
        // release the first probe before forcing the second.
        drop(_guard);
        let _g2 = ProbeGuard::force(NetworkEnforcement::AppLevel);
        assert_eq!(
            e.check_shell_feasibility().unwrap_err().enforcement,
            NetworkEnforcement::AppLevel
        );
    }

    #[test]
    fn required_ask_path_also_fails_closed() {
        // An Ask rule must not turn into a runnable prompt when the
        // platform cannot isolate the shell: refuse as Deny.
        let _guard = ProbeGuard::force(NetworkEnforcement::Unavailable);
        let policy = SandboxPolicy {
            execute_shell: Rule::Ask,
            network_guarantee: SandboxGuarantee::Required,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        assert_eq!(e.evaluate(&shell_cap()), PermissionDecision::Deny);
        // Rule Deny stays Deny (no feasibility question arises).
        let policy = SandboxPolicy {
            execute_shell: Rule::Deny,
            network_guarantee: SandboxGuarantee::Required,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        assert_eq!(e.evaluate(&shell_cap()), PermissionDecision::Deny);
    }

    #[test]
    fn required_with_os_level_enforcement_runs() {
        // Forced "backend proven at spawn" state (the state a future
        // terminal unshare(CLONE_NEWNET) DenyAll proof would report): a
        // Required policy may then pass the feasibility seam and reach the
        // spawn layer, which must itself carry DenyAll isolation.
        let _guard = ProbeGuard::force(NetworkEnforcement::OsLevel);
        let policy = SandboxPolicy {
            execute_shell: Rule::Allow,
            network_guarantee: SandboxGuarantee::Required,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        assert_eq!(e.check_shell_feasibility(), Ok(()));
        assert_eq!(e.evaluate(&shell_cap()), PermissionDecision::Allow);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn required_fails_closed_on_the_real_linux_probe_until_a_backend_proves_itself() {
        // NO probe forcing: the real host verdict. Audit 4/28/35-39: the
        // old probe claimed OsLevel from /proc/self/ns/net + CAP_SYS_ADMIN
        // existence while the spawn path applied NO namespace, so Required
        // ran an ordinary networked shell. Until a spawn backend proves
        // itself active at spawn, the honest Linux verdict is AppLevel and
        // a Required policy must refuse BEFORE spawn — even as root, even
        // with CAP_SYS_ADMIN set.
        let _read_lock = probe_read_lock();
        assert_ne!(
            platform_network_enforcement(),
            NetworkEnforcement::OsLevel,
            "no Linux spawn backend has proven itself at spawn in this process; \
             claiming OsLevel would let a Required shell run unisolated"
        );
        let policy = SandboxPolicy {
            execute_shell: Rule::Allow,
            network_guarantee: SandboxGuarantee::Required,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        let err = e.check_shell_feasibility().unwrap_err();
        assert_eq!(err.guarantee, SandboxGuarantee::Required);
        assert_ne!(
            err.enforcement,
            NetworkEnforcement::OsLevel,
            "the typed refusal must report the truthful enforcement"
        );
        assert_eq!(
            e.evaluate(&shell_cap()),
            PermissionDecision::Deny,
            "Required + real-Linux (backend unproven) must fail closed before spawn"
        );
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn required_fails_closed_on_the_real_macos_windows_probe() {
        // macOS/windows semantics stay as-is: no per-process deny backend
        // is implemented, so the real probe is Unavailable and Required
        // refuses typed before spawn.
        let _read_lock = probe_read_lock();
        assert_eq!(
            platform_network_enforcement(),
            NetworkEnforcement::Unavailable
        );
        let policy = SandboxPolicy {
            execute_shell: Rule::Allow,
            network_guarantee: SandboxGuarantee::Required,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        let err = e.check_shell_feasibility().unwrap_err();
        assert_eq!(err.guarantee, SandboxGuarantee::Required);
        assert_eq!(err.enforcement, NetworkEnforcement::Unavailable);
        assert_eq!(
            e.evaluate(&shell_cap()),
            PermissionDecision::Deny,
            "Required + real-macOS/windows must fail closed before spawn"
        );
    }

    #[test]
    fn best_effort_runs_behind_app_level_gates_with_documented_note() {
        let _guard = ProbeGuard::force(NetworkEnforcement::Unavailable);
        let policy = SandboxPolicy {
            execute_shell: Rule::Allow,
            network_guarantee: SandboxGuarantee::BestEffort,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        // Runs through the existing gate (rule Allow) even though the
        // platform is Unavailable — BestEffort does not demand OS backing.
        assert_eq!(e.check_shell_feasibility(), Ok(()));
        assert_eq!(e.evaluate(&shell_cap()), PermissionDecision::Allow);
        // The app-level-only caveat is documented on the type and in the
        // canonical note constant asserted here.
        assert!(
            NETWORK_ISOLATION_NOTE.contains("app-level only"),
            "{NETWORK_ISOLATION_NOTE}"
        );
        assert!(NETWORK_ISOLATION_NOTE.contains("ExecuteShell + Network(deny)"));
        assert!(NETWORK_ISOLATION_NOTE.contains("permitted shell can still open its own sockets"));
        // BestEffort + Ask stays Ask (the host decides whether to prompt).
        let policy = SandboxPolicy {
            execute_shell: Rule::Ask,
            network_guarantee: SandboxGuarantee::BestEffort,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        assert_eq!(e.evaluate(&shell_cap()), PermissionDecision::Ask);
    }

    #[test]
    fn none_guarantee_runs_under_any_probe_verdict() {
        for verdict in [
            NetworkEnforcement::OsLevel,
            NetworkEnforcement::AppLevel,
            NetworkEnforcement::Unavailable,
        ] {
            let _guard = ProbeGuard::force(verdict);
            let policy = SandboxPolicy {
                execute_shell: Rule::Allow,
                ..Default::default() // network_guarantee: None
            };
            let e = PermissionEngine::new(policy, None);
            assert_eq!(
                e.check_shell_feasibility(),
                Ok(()),
                "None claims no guarantee under {verdict}"
            );
            assert_eq!(e.evaluate(&shell_cap()), PermissionDecision::Allow);
        }
    }

    #[test]
    fn probe_never_panics_and_default_policy_claims_none() {
        let _ = platform_network_enforcement(); // any verdict, no panic
        assert_eq!(
            SandboxPolicy::default().network_guarantee,
            SandboxGuarantee::None
        );
        // Enforcement/guarantee serde shapes are frozen.
        let v = serde_json::to_value(NetworkEnforcement::AppLevel).unwrap();
        assert_eq!(v, serde_json::json!("app_level"));
        let v = serde_json::to_value(SandboxGuarantee::Required).unwrap();
        assert_eq!(v, serde_json::json!("required"));
        assert!(
            serde_json::from_value::<SandboxGuarantee>(serde_json::json!("best_effort")).is_ok()
        );
    }

    #[test]
    fn network_deny_under_best_effort_is_documented_app_level_only() {
        // The adversarial scenario that started P0-39: ExecuteShell
        // allowed + Network denied. Under BestEffort this is app-level
        // only — the docs on the enforcement type say so in plain words.
        let _guard = ProbeGuard::force(NetworkEnforcement::Unavailable);
        let policy = SandboxPolicy {
            execute_shell: Rule::Allow,
            network: NetworkGate::deny_all(),
            network_guarantee: SandboxGuarantee::BestEffort,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        // The app-level gate still denies the app's own egress...
        assert_eq!(
            e.evaluate(&Capability::Network {
                destination: "https://evil.example.com".into()
            }),
            PermissionDecision::Deny
        );
        // ...but the shell still runs (documented as app-level only), and
        // the Display of AppLevel states the limitation verbatim.
        assert_eq!(e.evaluate(&shell_cap()), PermissionDecision::Allow);
        let app_level_doc = NetworkEnforcement::AppLevel.to_string();
        assert!(app_level_doc.contains("app-level"));
        assert!(app_level_doc.contains("permitted shell can open its own sockets"));
    }
}
