//! Tool definitions. Tools are stateless callables with declared metadata;
//! they never touch session persistence (Commandment 1) and every invocation
//! carries its workspace identity explicitly.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use kilop_core::capability::Capability;
use kilop_core::error::Error;
use kilop_core::id::{OpId, SessionId};
use kilop_core::resource::ResourceClass;
use kilop_core::WorkspaceIdentity;
use kilop_provider::ToolSpec;

use crate::tool_json::{parse_tool_calls, ToolCallMode};

/// How the runtime should recover this tool after a crash (spec §7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryHint {
    /// Deterministic write: expected hash = blake3(content from `content_arg`
    /// of the args JSON) at `path_arg`; recovery verifies the file.
    VerifyHash {
        path_arg: String,
        content_arg: String,
    },
    /// Reads / idempotent commands: safe to re-run after a crash.
    Idempotent,
    /// Commands with unknown external effects: mark `effect_status =
    /// unknown` and force verification; never blindly re-run.
    UnknownEffect,
}

#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub resource_class: ResourceClass,
    pub capability: Option<Capability>,
    pub recovery_hint: RecoveryHint,
    /// Which arg names of this tool are filesystem paths. The scheduler's
    /// ownership sets (reads/writes) are derived from these; the direction
    /// (read vs write) follows the tool's resource class (DiskWrite ⇒
    /// writes, anything else ⇒ reads). Empty for tools that touch no paths.
    pub path_args: Vec<String>,
    pub execute: ToolFn,
}

/// Filesystem ownership of one tool invocation, derived from the tool's
/// declared path arguments (spec §22: edits with overlapping writes
/// serialize; reads never block each other).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Ownership {
    pub reads: Vec<String>,
    pub writes: Vec<String>,
}

impl Default for ToolOutcome {
    fn default() -> Self {
        Self {
            text: String::new(),
            exit_code: None,
            artifact: None,
            slice_hint: None,
            effect_status: kilop_core::op::EffectStatus::Applied,
        }
    }
}

impl Tool {
    pub fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: self.description.clone(),
            input_schema: self.input_schema.clone(),
        }
    }

    /// Derive the reads/writes this invocation touches from the declared
    /// path args. Non-string arg values are skipped (never trusted).
    pub fn ownership(&self, args: &serde_json::Value) -> Ownership {
        let is_write = self.resource_class == ResourceClass::DiskWrite;
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        for arg_name in &self.path_args {
            let Some(path) = args.get(arg_name).and_then(|v| v.as_str()) else {
                continue;
            };
            if is_write {
                writes.push(path.to_string());
            } else {
                reads.push(path.to_string());
            }
        }
        Ownership { reads, writes }
    }
}

/// Execution context: explicit identity + cancellation + artifact writer.
/// The real filesystem stack is injected per invocation by the runtime:
/// `None` means "no workspace wired" and tools must error honestly.
#[derive(Clone)]
pub struct ToolRunCtx {
    pub session_id: SessionId,
    pub op_id: OpId,
    pub identity: WorkspaceIdentity,
    pub cancellation: kilop_core::cancellation::CancellationToken,
    pub artifacts: Arc<crate::ToolArtifactSink>,
    pub tool_call_mode: ToolCallMode,
    /// Resolved workspace handle for the session (canonical root, watcher).
    pub workspace: Option<Arc<kilop_fs::WorkspaceHandle>>,
    /// Transactional edit engine for optimistic writes.
    pub edit: Option<Arc<kilop_edit::EditEngine>>,
    /// CAS-backed checkpoint store (before/after hashes for undo).
    pub snapshots: Option<Arc<kilop_snapshot::CheckpointStore>>,
    /// Capability permission engine rooted at the session workspace.
    pub sandbox: Option<Arc<kilop_sandbox::PermissionEngine>>,
    /// Process supervisor for run_command (no orphans, bounded output).
    pub supervisor: Option<Arc<kilop_terminal::ProcessSupervisor>>,
    /// Remaining op deadline in ms (0 → tool default).
    pub deadline_ms: u64,
    /// The runtime resolved this tool's permission hop to Allow before the
    /// call (the Ask decision lives in the daemon's permission requester).
    /// Tools re-check hard DENY rules; an Ask-policy verdict may proceed
    /// ONLY when this is set — a direct, permission-less invocation (tests,
    /// mis-wired registries) still refuses on Ask.
    pub permission_granted: bool,
}

/// Result of one tool invocation. `text` is bounded by the tool itself
/// (ring-buffer excerpts); big outputs go to `artifacts`.
#[derive(Debug, Clone)]
pub struct ToolOutcome {
    pub text: String,
    pub exit_code: Option<i32>,
    pub artifact: Option<String>,
    pub slice_hint: Option<String>,
    pub effect_status: kilop_core::op::EffectStatus,
}

pub type ToolFn = Arc<
    dyn Fn(
            ToolRunCtx,
            serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = Result<ToolOutcome, Error>> + Send>>
        + Send
        + Sync,
>;

/// Tool registry: the agent asks the registry, tools are wired by the CLI.
#[derive(Default)]
pub struct ToolRegistry {
    tools: std::collections::HashMap<String, Arc<Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, tool: Tool) {
        let name = tool.name.clone();
        self.tools.insert(name, Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.tools.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        let mut v: Vec<ToolSpec> = self.tools.values().map(|t| t.spec()).collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Convenience for tools that need to parse their own args with bounds.
pub fn bound_args(args: &serde_json::Value, max_bytes: usize) -> Result<serde_json::Value, Error> {
    let s = serde_json::to_vec(args)
        .map_err(|e| Error::malformed(format!("args not serializable: {e}")))?;
    if s.len() > max_bytes {
        return Err(Error::oversized(format!(
            "tool args {} bytes exceed bound {max_bytes}",
            s.len()
        )));
    }
    Ok(args.clone())
}

/// Parse tool-call text extracted from a stream (StructuredFallback path).
pub fn parse_tool_text(text: &str, mode: ToolCallMode) -> Vec<crate::tool_json::ParsedToolCall> {
    parse_tool_calls(text, mode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_roundtrip_and_unknown() {
        let mut r = ToolRegistry::new();
        r.register(Tool {
            name: "read_file".into(),
            description: "d".into(),
            input_schema: serde_json::json!({"type": "object"}),
            resource_class: ResourceClass::DiskRead,
            capability: None,
            recovery_hint: RecoveryHint::Idempotent,
            path_args: vec!["path".into()],
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move {
                    Ok(ToolOutcome {
                        text: format!("{args:?}"),
                        ..Default::default()
                    })
                })
            }),
        });
        assert_eq!(r.names(), vec!["read_file"]);
        assert!(r.get("read_file").is_some());
        assert!(r.get("write_file").is_none());
        assert_eq!(r.specs().len(), 1);
    }

    #[tokio::test]
    async fn tool_gets_explicit_identity_and_cancellation() {
        let mut r = ToolRegistry::new();
        let seen = Arc::new(std::sync::Mutex::new(None));
        let seen2 = seen.clone();
        r.register(Tool {
            name: "identity_probe".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            resource_class: ResourceClass::Cpu,
            capability: None,
            recovery_hint: RecoveryHint::Idempotent,
            path_args: vec![],
            execute: Arc::new(move |ctx, args| {
                let seen = seen2.clone();
                Box::pin(async move {
                    *seen.lock().unwrap() = Some((ctx.session_id, ctx.identity.workspace_id, args));
                    Ok(ToolOutcome::default())
                })
            }),
        });
        let token = kilop_core::cancellation::CancellationToken::new();
        let ctx = ToolRunCtx {
            session_id: SessionId::new(5),
            op_id: OpId::new(6),
            identity: WorkspaceIdentity::new(
                kilop_core::WorkspaceId::new(1),
                kilop_core::WorktreeId::new(2),
                kilop_core::TaskId::new(3),
            ),
            cancellation: token.clone(),
            artifacts: Arc::new(crate::ToolArtifactSink::Null),
            tool_call_mode: ToolCallMode::Native,
            workspace: None,
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            deadline_ms: 0,
            permission_granted: false,
        };
        let tool = r.get("identity_probe").unwrap();
        (tool.execute)(ctx, serde_json::json!({"path": "/x"}))
            .await
            .unwrap();
        let got = seen.lock().unwrap().take().unwrap();
        assert_eq!(got.0, SessionId::new(5));
        assert_eq!(got.1, kilop_core::WorkspaceId::new(1));
        assert_eq!(got.2["path"], "/x");
        // Cancellation token arrives intact and functional.
        assert!(!token.is_cancelled());
    }

    #[test]
    fn bound_args_rejects_oversized() {
        let big = serde_json::json!({"payload": "x".repeat(10_000)});
        assert!(bound_args(&big, 100).is_err());
        assert!(bound_args(&big, 100_000).is_ok());
    }

    #[test]
    fn recovery_hints_are_explicit() {
        let h = RecoveryHint::VerifyHash {
            path_arg: "path".into(),
            content_arg: "content".into(),
        };
        assert!(matches!(h, RecoveryHint::VerifyHash { .. }));
        assert!(matches!(RecoveryHint::Idempotent, RecoveryHint::Idempotent));
        assert!(matches!(
            RecoveryHint::UnknownEffect,
            RecoveryHint::UnknownEffect
        ));
    }

    #[test]
    fn ownership_derived_from_path_args_and_resource_class() {
        let read = Tool {
            name: "read_file".into(),
            description: "d".into(),
            input_schema: serde_json::json!({}),
            resource_class: ResourceClass::DiskRead,
            capability: None,
            recovery_hint: RecoveryHint::Idempotent,
            path_args: vec!["path".into()],
            execute: Arc::new(|_ctx, _args| Box::pin(async move { Ok(ToolOutcome::default()) })),
        };
        let write = Tool {
            path_args: vec!["path".into()],
            resource_class: ResourceClass::DiskWrite,
            ..read.clone()
        };
        let no_paths = Tool {
            path_args: vec![],
            ..read.clone()
        };

        let o = read.ownership(&serde_json::json!({"path": "src/a.rs"}));
        assert_eq!(o.reads, vec!["src/a.rs"]);
        assert!(o.writes.is_empty());

        let o = write.ownership(&serde_json::json!({"path": "src/a.rs"}));
        assert!(o.reads.is_empty());
        assert_eq!(o.writes, vec!["src/a.rs"]);

        assert_eq!(
            read.ownership(&serde_json::json!({"path": 42})),
            Ownership::default(),
            "non-string path args are never trusted"
        );
        assert_eq!(
            no_paths.ownership(&serde_json::json!({"path": "x"})),
            Ownership::default(),
            "tools with no declared path args own nothing"
        );
    }
}
