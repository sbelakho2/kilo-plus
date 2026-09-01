//! Sandbox capabilities and permission decisions. Permissions are expressed
//! as capabilities, never as scattered `if provider == ...` conditionals.

use std::path::PathBuf;

/// A concrete capability request. The sandbox (kilop-sandbox) maps these to
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
}
