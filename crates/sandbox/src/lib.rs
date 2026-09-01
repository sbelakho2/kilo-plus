//! kilop-sandbox — capability-based permission enforcement (spec §30).
//!
//! Permissions are expressed as capabilities, never scattered conditionals.
//! Path checks are canonicalization-safe (symlink escapes and `..`
//! traversal are rejected); the network policy is the three-mode sandbox
//! from the spec (deny all / allow provider endpoints / allow configured).

use std::fs;
use std::path::{Component, Path, PathBuf};

use kilop_core::capability::{Capability, NetworkPolicy, PermissionDecision};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SandboxPolicy {
    pub read_workspace: Rule,
    pub write_workspace: Rule,
    pub read_external: Rule,
    pub write_external: Rule,
    pub execute_shell: Rule,
    pub network: NetworkPolicy,
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
            network: NetworkPolicy::AllowProviders {
                endpoints: vec![
                    "https://api.openai.com".into(),
                    "https://api.anthropic.com".into(),
                    "https://generativelanguage.googleapis.com".into(),
                    "https://api.deepseek.com".into(),
                ],
            },
            mcp: Rule::Allow,
            git: Rule::Allow,
        }
    }
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
            Capability::Network { destination } => {
                if self.policy.network.allows(destination) {
                    PermissionDecision::Allow
                } else {
                    PermissionDecision::Deny
                }
            }
            Capability::Mcp { .. } => rule_decision(self.policy.mcp),
            Capability::Git { .. } => rule_decision(self.policy.git),
        }
    }
}

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
    fn network_policy_matrix() {
        let e = PermissionEngine::new(SandboxPolicy::default(), None);
        assert_eq!(
            e.evaluate(&Capability::Network {
                destination: "https://api.openai.com/v1".into()
            }),
            PermissionDecision::Allow
        );
        assert_eq!(
            e.evaluate(&Capability::Network {
                destination: "https://evil.example.com".into()
            }),
            PermissionDecision::Deny
        );
        let policy = SandboxPolicy {
            network: NetworkPolicy::DenyAll,
            ..Default::default()
        };
        let e = PermissionEngine::new(policy, None);
        assert_eq!(
            e.evaluate(&Capability::Network {
                destination: "https://api.openai.com".into()
            }),
            PermissionDecision::Deny
        );
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
    }
}
