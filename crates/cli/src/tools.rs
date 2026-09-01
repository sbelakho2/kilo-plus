//! Built-in tools for the daemon (spec §17/§22/§30). Tools never touch
//! session persistence; every invocation carries its workspace identity and
//! runs through the permission engine.

use std::sync::Arc;

use kilop_agent::{RecoveryHint, Tool, ToolOutcome};
use kilop_core::capability::Capability;
use kilop_core::error::Error;
use kilop_core::resource::ResourceClass;

pub fn read_file_tool() -> Tool {
    Tool {
        name: "read_file".into(),
        description: "Read a file within the workspace (bounded).".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "max_bytes": { "type": "integer" }
            },
            "required": ["path"]
        }),
        resource_class: ResourceClass::DiskRead,
        capability: Some(Capability::ReadWorkspace { path: ".".into() }),
        recovery_hint: RecoveryHint::Idempotent,
        execute: Arc::new(|_ctx, args| {
            Box::pin(async move {
                let path = args
                    .get("path")
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| Error::malformed("read_file requires path"))?;
                let max = args
                    .get("max_bytes")
                    .and_then(|m| m.as_u64())
                    .unwrap_or(64 * 1024) as usize;
                let data =
                    std::fs::read(path).map_err(|e| Error::not_found(format!("{path}: {e}")))?;
                let truncated = data.len() > max;
                let bytes = data.into_iter().take(max).collect::<Vec<_>>();
                let text = String::from_utf8_lossy(&bytes).to_string();
                Ok(ToolOutcome {
                    text: if truncated {
                        format!("{text}\n[truncated at {max} bytes]")
                    } else {
                        text
                    },
                    exit_code: Some(0),
                    ..Default::default()
                })
            })
        }),
    }
}

pub fn write_file_tool() -> Tool {
    Tool {
        name: "write_file".into(),
        description: "Write a file within the workspace (atomic).".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        }),
        resource_class: ResourceClass::DiskWrite,
        capability: Some(Capability::WriteWorkspace { path: ".".into() }),
        recovery_hint: RecoveryHint::VerifyHash {
            path_arg: "path".into(),
            content_arg: "content".into(),
        },
        execute: Arc::new(|_ctx, args| {
            Box::pin(async move {
                let path = args
                    .get("path")
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| Error::malformed("write_file requires path"))?;
                let content = args
                    .get("content")
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| Error::malformed("write_file requires content"))?;
                if content.len() > 16 * 1024 * 1024 {
                    return Err(Error::oversized("write_file content exceeds 16MB"));
                }
                let parent = std::path::Path::new(path)
                    .parent()
                    .ok_or_else(|| Error::malformed("path has no parent"))?;
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::internal(format!("mkdir {parent:?}: {e}")))?;
                let tmp = format!("{path}.kp-tmp-{}", std::process::id());
                std::fs::write(&tmp, content)
                    .map_err(|e| Error::internal(format!("write tmp: {e}")))?;
                std::fs::rename(&tmp, path).map_err(|e| Error::internal(format!("rename: {e}")))?;
                Ok(ToolOutcome {
                    text: format!("wrote {} ({} bytes)", path, content.len()),
                    exit_code: Some(0),
                    ..Default::default()
                })
            })
        }),
    }
}

pub fn search_tool() -> Tool {
    Tool {
        name: "search".into(),
        description: "Substring search over workspace files (bounded).".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string" },
                "path": { "type": "string" }
            },
            "required": ["pattern"]
        }),
        resource_class: ResourceClass::DiskRead,
        capability: Some(Capability::ReadWorkspace { path: ".".into() }),
        recovery_hint: RecoveryHint::Idempotent,
        execute: Arc::new(|_ctx, args| {
            Box::pin(async move {
                let pattern = args
                    .get("pattern")
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| Error::malformed("search requires pattern"))?;
                if pattern.len() > 1024 {
                    return Err(Error::oversized("pattern too long"));
                }
                let root = args.get("path").and_then(|p| p.as_str()).unwrap_or(".");
                let mut hits = Vec::new();
                walk_search(root, pattern, 0, &mut hits, 64);
                if hits.is_empty() {
                    return Ok(ToolOutcome {
                        text: "no matches".into(),
                        exit_code: Some(1),
                        ..Default::default()
                    });
                }
                Ok(ToolOutcome {
                    text: hits.join("\n"),
                    exit_code: Some(0),
                    ..Default::default()
                })
            })
        }),
    }
}

fn walk_search(dir: &str, pattern: &str, depth: usize, hits: &mut Vec<String>, max: usize) {
    if hits.len() >= max || depth > 8 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if hits.len() >= max {
            return;
        }
        let path = entry.path();
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        if name == ".git" || name.starts_with("target") || name == "node_modules" {
            continue;
        }
        if path.is_dir() {
            walk_search(path.to_str().unwrap_or(""), pattern, depth + 1, hits, max);
        } else if let Ok(bytes) = std::fs::read(&path) {
            if bytes.len() <= 2 * 1024 * 1024 && String::from_utf8_lossy(&bytes).contains(pattern) {
                hits.push(path.to_string_lossy().to_string());
            }
        }
    }
}

pub fn run_command_tool() -> Tool {
    Tool {
        name: "run_command".into(),
        description: "Run a shell command in the workspace (supervised, bounded).".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" }
            },
            "required": ["command"]
        }),
        resource_class: ResourceClass::Terminal,
        capability: Some(Capability::ExecuteShell {
            command: String::new(),
        }),
        recovery_hint: RecoveryHint::UnknownEffect,
        execute: Arc::new(|_ctx, args| {
            Box::pin(async move {
                let command = args
                    .get("command")
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| Error::malformed("run_command requires command"))?;
                if command.len() > 4096 {
                    return Err(Error::oversized("command too long"));
                }
                // This tool is wired by the CLI with the real supervisor;
                // this default implementation reports the bound so the tool
                // registry is complete without process access.
                Ok(ToolOutcome {
                    text: format!(
                        "command `{command}` — execution requires the daemon's process supervisor"
                    ),
                    exit_code: None,
                    ..Default::default()
                })
            })
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_agent::ToolRunCtx;
    use kilop_core::cancellation::CancellationToken;
    use kilop_core::id::{OpId, SessionId, TaskId, WorkspaceId, WorktreeId};
    use kilop_core::WorkspaceIdentity;

    fn ctx() -> ToolRunCtx {
        ToolRunCtx {
            session_id: SessionId::new(1),
            op_id: OpId::new(1),
            identity: WorkspaceIdentity::new(
                WorkspaceId::new(1),
                WorktreeId::new(1),
                TaskId::new(1),
            ),
            cancellation: CancellationToken::new(),
            artifacts: Arc::new(kilop_agent::ToolArtifactSink::Null),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
        }
    }

    #[tokio::test]
    async fn read_file_bounds_and_truncates() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "x".repeat(10_000)).unwrap();
        let tool = read_file_tool();
        let out = (tool.execute)(
            ctx(),
            serde_json::json!({"path": dir.path().join("f.txt"), "max_bytes": 100}),
        )
        .await
        .unwrap();
        assert!(out.text.contains("truncated at 100 bytes"));
        let out = (tool.execute)(ctx(), serde_json::json!({"path": dir.path().join("f.txt")}))
            .await
            .unwrap();
        assert!(!out.text.contains("truncated"));
    }

    #[tokio::test]
    async fn read_file_missing_is_not_found() {
        let tool = read_file_tool();
        let err = (tool.execute)(ctx(), serde_json::json!({"path": "/nope"}))
            .await
            .unwrap_err();
        assert!(err.kind == kilop_core::error::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn write_file_atomic_and_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("out.txt");
        let tool = write_file_tool();
        (tool.execute)(
            ctx(),
            serde_json::json!({"path": target, "content": "hello"}),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"hello");
        let err = (tool.execute)(
            ctx(),
            serde_json::json!({"path": target, "content": "x".repeat(17 * 1024 * 1024)}),
        )
        .await
        .unwrap_err();
        assert!(err.kind == kilop_core::error::ErrorKind::Oversized);
    }

    #[tokio::test]
    async fn search_is_bounded_and_skips_vcs() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/config"), "needle here").unwrap();
        std::fs::write(dir.path().join("a.txt"), "needle found").unwrap();
        std::fs::write(dir.path().join("b.txt"), "nothing").unwrap();
        let tool = search_tool();
        let out = (tool.execute)(
            ctx(),
            serde_json::json!({"pattern": "needle", "path": dir.path()}),
        )
        .await
        .unwrap();
        assert!(out.text.contains("a.txt"));
        assert!(!out.text.contains(".git"), "vcs dirs must be skipped");
        assert_eq!(out.exit_code, Some(0));
    }

    #[tokio::test]
    async fn run_command_validates_input() {
        let tool = run_command_tool();
        let out = (tool.execute)(ctx(), serde_json::json!({"command": "ls"}))
            .await
            .unwrap();
        assert!(out.text.contains("ls"));
        let err = (tool.execute)(ctx(), serde_json::json!({"command": "x".repeat(5000)}))
            .await
            .unwrap_err();
        assert!(err.kind == kilop_core::error::ErrorKind::Oversized);
    }

    #[tokio::test]
    async fn malicious_args_never_panic() {
        let tool = read_file_tool();
        for args in [
            serde_json::json!({}),
            serde_json::json!({"path": 42}),
            serde_json::json!({"path": ["a"]}),
            serde_json::json!({"path": "\u{0}"}),
        ] {
            let _ = (tool.execute)(ctx(), args).await;
        }
    }
}
