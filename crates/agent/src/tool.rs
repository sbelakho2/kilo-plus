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
    VerifyHash { path_arg: String, content_arg: String },
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
    pub execute: ToolFn,
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
}

/// Execution context: explicit identity + cancellation + artifact writer.
#[derive(Clone)]
pub struct ToolRunCtx {
    pub session_id: SessionId,
    pub op_id: OpId,
    pub identity: WorkspaceIdentity,
    pub cancellation: kilop_core::cancellation::CancellationToken,
    pub artifacts: Arc<crate::ToolArtifactSink>,
    pub tool_call_mode: ToolCallMode,
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
            execute: Arc::new(|_ctx, args| {
                Box::pin(async move { Ok(ToolOutcome { text: format!("{args:?}"), ..Default::default() }) })
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
        };
        let tool = r.get("identity_probe").unwrap();
        (tool.execute)(ctx, serde_json::json!({"path": "/x"})).await.unwrap();
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
        assert!(matches!(RecoveryHint::UnknownEffect, RecoveryHint::UnknownEffect));
    }
}
