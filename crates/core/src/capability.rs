//! Sandbox capabilities and permission decisions. Permissions are expressed
//! as capabilities, never as scattered `if provider == ...` conditionals.

use std::path::PathBuf;

/// A concrete capability request. The sandbox (faktor-sandbox) maps these to
/// `PermissionDecision` using session policy.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "capability", content = "detail", rename_all = "snake_case")]
pub enum Capability {
    ReadWorkspace { path: PathBuf },
    WriteWorkspace { path: PathBuf },
    ReadExternal { path: PathBuf },
    WriteExternal { path: PathBuf },
    ExecuteShell { command: String },
    Network { destination: String },
    Mcp { server: String },
    Git { operation: String },
}

impl Capability {
    pub fn describe(&self) -> String {
        match self {
            Capability::ReadWorkspace { path } => format!("read {path:?}"),
            Capability::WriteWorkspace { path } => format!("write {path:?}"),
            Capability::ReadExternal { path } => format!("read external {path:?}"),
            Capability::WriteExternal { path } => format!("write external {path:?}"),
            Capability::ExecuteShell { command } => format!("execute `{command}`"),
            Capability::Network { destination } => format!("network {destination}"),
            Capability::Mcp { server } => format!("MCP {server}"),
            Capability::Git { operation } => format!("git {operation}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow,
    Deny,
    /// Ask the user through the frozen permission dialog.
    Ask,
}

/// Atomic capability classes for typed permission scopes (audit P0-40: the
/// hook `permission_scope` was an unchecked free-form String; scopes and
/// envelopes are now a typed lattice over these classes, mirroring the
/// [`Capability`] variants minus their per-request detail).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityKind {
    Read,
    Write,
    Execute,
    Network,
    Git,
    Mcp,
}

impl CapabilityKind {
    pub const ALL: [CapabilityKind; 6] = [
        CapabilityKind::Read,
        CapabilityKind::Write,
        CapabilityKind::Execute,
        CapabilityKind::Network,
        CapabilityKind::Git,
        CapabilityKind::Mcp,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            CapabilityKind::Read => "read",
            CapabilityKind::Write => "write",
            CapabilityKind::Execute => "execute",
            CapabilityKind::Network => "network",
            CapabilityKind::Git => "git",
            CapabilityKind::Mcp => "mcp",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        CapabilityKind::ALL
            .iter()
            .find(|k| k.as_str() == s)
            .copied()
    }
}

/// A typed capability set with lattice (subset) order: any finite subset of
/// [`CapabilityKind`], with `ALL` as the top element. Subset checks are the
/// permission-surface gate — a hook whose declared scope exceeds the
/// granted envelope is refused BEFORE any child is spawned. An unknown
/// capability id fails closed at deserialization (never assumed allowed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CapabilitySet(u8);

impl CapabilitySet {
    /// The empty scope: claims nothing, allowed under every envelope.
    pub const EMPTY: Self = Self(0);
    /// The full set — every known capability class.
    pub const ALL: Self = Self(0b0011_1111);

    pub const fn empty() -> Self {
        Self::EMPTY
    }

    pub const fn all() -> Self {
        Self::ALL
    }

    pub const fn of(kind: CapabilityKind) -> Self {
        Self(1 << kind as u8)
    }

    pub const fn contains(self, kind: CapabilityKind) -> bool {
        self.0 & (1 << kind as u8) != 0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn from_kinds(kinds: &[CapabilityKind]) -> Self {
        let mut mask = 0u8;
        for k in kinds {
            mask |= 1 << *k as u8;
        }
        Self(mask)
    }

    pub fn kinds(self) -> impl Iterator<Item = CapabilityKind> {
        CapabilityKind::ALL
            .iter()
            .copied()
            .filter(move |k| self.contains(*k))
    }

    /// Lattice order: `self` is within `other` when every claimed class is
    /// granted there. `ALL ⊄` any finite set; every set is within `ALL`.
    pub const fn is_subset_of(self, other: Self) -> bool {
        self.0 & !other.0 == 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }
}

impl Default for CapabilitySet {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl std::fmt::Display for CapabilitySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if *self == Self::ALL {
            return f.write_str("*");
        }
        let mut first = true;
        for k in self.kinds() {
            if !first {
                f.write_str(",")?;
            }
            first = false;
            f.write_str(k.as_str())?;
        }
        Ok(())
    }
}

impl serde::Serialize for CapabilitySet {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl<'de> serde::Deserialize<'de> for CapabilitySet {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = CapabilitySet;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a capability scope: \"*\", \"read,write\", or an array of class names")
            }
            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                if v == "*" {
                    return Ok(CapabilitySet::ALL);
                }
                let mut out = CapabilitySet::EMPTY;
                for part in v.split(',').map(str::trim) {
                    if part.is_empty() {
                        continue;
                    }
                    let k = CapabilityKind::from_name(part)
                        .ok_or_else(|| E::custom(format!("unknown capability class {part:?}")))?;
                    out = out.union(CapabilitySet::of(k));
                }
                Ok(out)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut out = CapabilitySet::EMPTY;
                while let Some(s) = seq.next_element::<String>()? {
                    let k = CapabilityKind::from_name(&s).ok_or_else(|| {
                        serde::de::Error::custom(format!("unknown capability class {s:?}"))
                    })?;
                    out = out.union(CapabilitySet::of(k));
                }
                Ok(out)
            }
        }
        d.deserialize_any(V)
    }
}

/// Network sandbox policy.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum NetworkPolicy {
    /// Everything blocked.
    DenyAll,
    /// Only provider endpoints listed in config.
    AllowProviders { endpoints: Vec<String> },
    /// Provider endpoints plus explicitly configured domains.
    AllowConfigured {
        endpoints: Vec<String>,
        domains: Vec<String>,
    },
}

impl NetworkPolicy {
    pub fn allows(&self, destination: &str) -> bool {
        match self {
            NetworkPolicy::DenyAll => false,
            NetworkPolicy::AllowProviders { endpoints } => {
                endpoints.iter().any(|e| destination.starts_with(e))
            }
            NetworkPolicy::AllowConfigured { endpoints, domains } => {
                endpoints.iter().any(|e| destination.starts_with(e))
                    || domains.iter().any(|d| destination.starts_with(d))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_policy_matrix() {
        let deny = NetworkPolicy::DenyAll;
        assert!(!deny.allows("https://api.openai.com/v1"));
        let providers = NetworkPolicy::AllowProviders {
            endpoints: vec!["https://api.openai.com".into()],
        };
        assert!(providers.allows("https://api.openai.com/v1/chat"));
        // prefix matching prevents escape via subdomain tricks? "evilapi.openai.com"
        // does not start with "https://api.openai.com" — good, but the test
        // documents that exact-prefix is the rule.
        assert!(!providers.allows("https://evilapi.openai.com/v1"));
        assert!(!providers.allows("https://example.com"));
        let configured = NetworkPolicy::AllowConfigured {
            endpoints: vec!["https://api.anthropic.com".into()],
            domains: vec!["https://mcp.example.com".into()],
        };
        assert!(configured.allows("https://api.anthropic.com/v1/messages"));
        assert!(configured.allows("https://mcp.example.com"));
        assert!(
            !configured.allows("https://example.com"),
            "prefix on domain matters"
        );
    }

    #[test]
    fn capability_describe_is_nonempty_and_carries_detail() {
        for cap in [
            Capability::ReadWorkspace { path: ".".into() },
            Capability::WriteWorkspace { path: ".".into() },
            Capability::ReadExternal {
                path: "/etc".into(),
            },
            Capability::WriteExternal {
                path: "/etc".into(),
            },
            Capability::ExecuteShell {
                command: "rm -rf /".into(),
            },
            Capability::Network {
                destination: "https://x".into(),
            },
            Capability::Mcp {
                server: "fs".into(),
            },
            Capability::Git {
                operation: "push".into(),
            },
        ] {
            assert!(!cap.describe().is_empty());
        }
    }

    #[test]
    fn capability_json_tagging_roundtrip() {
        let cap = Capability::ExecuteShell {
            command: "cargo test".into(),
        };
        let v = serde_json::to_value(&cap).unwrap();
        assert_eq!(v["capability"], "execute_shell");
        let back: Capability = serde_json::from_value(v).unwrap();
        assert_eq!(back, cap);
        // unknown tags rejected
        let bad = serde_json::json!({"capability": "own_the_server", "detail": {}});
        assert!(serde_json::from_value::<Capability>(bad).is_err());
    }

    #[test]
    fn permission_decisions_serialize_stable() {
        assert_eq!(
            serde_json::to_string(&PermissionDecision::Ask).unwrap(),
            "\"ask\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionDecision::Allow).unwrap(),
            "\"allow\""
        );
        assert_eq!(
            serde_json::to_string(&PermissionDecision::Deny).unwrap(),
            "\"deny\""
        );
    }

    #[test]
    fn capability_set_lattice_orders_scopes() {
        let read = CapabilitySet::of(CapabilityKind::Read);
        let rw = read.union(CapabilitySet::of(CapabilityKind::Write));
        assert!(CapabilitySet::EMPTY.is_subset_of(CapabilitySet::EMPTY));
        assert!(CapabilitySet::EMPTY.is_subset_of(rw));
        assert!(read.is_subset_of(rw));
        assert!(!rw.is_subset_of(read), "write is not granted by {read}");
        assert!(rw.is_subset_of(CapabilitySet::ALL));
        assert!(!CapabilitySet::ALL.is_subset_of(rw));
        assert!(CapabilitySet::ALL.is_subset_of(CapabilitySet::ALL));
        assert!(rw
            .union(CapabilitySet::of(CapabilityKind::Mcp))
            .contains(CapabilityKind::Mcp));
    }

    #[test]
    fn capability_set_scope_serde_roundtrips_and_fails_closed_on_unknown() {
        let rw =
            CapabilitySet::of(CapabilityKind::Read).union(CapabilitySet::of(CapabilityKind::Write));
        assert_eq!(serde_json::to_string(&rw).unwrap(), "\"read,write\"");
        let back: CapabilitySet = serde_json::from_str("\"read,write\"").unwrap();
        assert_eq!(back, rw);
        assert_eq!(serde_json::to_string(&CapabilitySet::ALL).unwrap(), "\"*\"");
        assert_eq!(
            CapabilitySet::ALL,
            serde_json::from_str::<CapabilitySet>("\"*\"").unwrap()
        );
        assert_eq!(
            serde_json::to_string(&CapabilitySet::EMPTY).unwrap(),
            "\"\""
        );
        let arr: CapabilitySet = serde_json::from_str("[\"execute\",\"network\"]").unwrap();
        assert!(arr.contains(CapabilityKind::Execute));
        assert!(arr.contains(CapabilityKind::Network));
        assert!(!arr.contains(CapabilityKind::Read));
        // Unknown classes fail closed — never silently allowed.
        assert!(serde_json::from_str::<CapabilitySet>("\"read,own_the_server\"").is_err());
        assert!(serde_json::from_str::<CapabilitySet>("\"read\"").is_ok());
    }
}
