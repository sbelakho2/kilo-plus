//! Production MCP bridge (spec §31): supervised stdio servers spawn at
//! daemon start and their dynamic tools are surfaced into the agent's tool
//! registry as ordinary `Tool`s. Everything rides the runtime's op envelope
//! (deadline, cancellation, recovery); a broken server fails its tool calls
//! loudly and never destabilizes the daemon.

use std::sync::Arc;

use faktor_agent::{RecoveryHint, Tool, ToolOutcome, ToolRunCtx};
use faktor_core::capability::Capability;
use faktor_core::error::{Error, ErrorKind};
use faktor_core::resource::ResourceClass;
use faktor_mcp::{McpServer, McpTool};

/// Bounds for MCP tool-call results folded into one tool outcome.
const MCP_RESULT_TEXT_MAX: usize = 64 * 1024;

/// Convert one dynamic MCP tool into an agent tool. Invocations carry the
/// runtime's deadline; `is_error` results surface as non-zero exit codes;
/// external effects are always unknown (crash recovery never replays an MCP
/// call blindly — the journal marks the row for verification).
pub fn mcp_tool(server: Arc<McpServer>, tool: &McpTool) -> Tool {
    let name = tool.name.clone();
    let server_name = server.name().to_string();
    let desc = if tool.description.is_empty() {
        format!("MCP tool from {server_name}")
    } else {
        format!("[MCP {server_name}] {}", tool.description)
    };
    let input_schema = tool.input_schema.clone();
    Tool {
        name: name.clone(),
        description: desc,
        input_schema,
        resource_class: ResourceClass::Mcp,
        capability: Some(Capability::Mcp {
            server: server_name,
        }),
        recovery_hint: RecoveryHint::UnknownEffect,
        path_args: vec![],
        execute: Arc::new(move |ctx: ToolRunCtx, args: serde_json::Value| {
            let server = server.clone();
            let name = name.clone();
            Box::pin(async move {
                if serde_json::to_vec(&args).map(|b| b.len()).unwrap_or(0) > 128 * 1024 {
                    return Err(Error::new(
                        ErrorKind::Oversized,
                        "mcp tool arguments exceed 128 KiB",
                    ));
                }
                let deadline_ms = if ctx.deadline_ms > 0 {
                    ctx.deadline_ms
                } else {
                    30_000
                };
                let result = tokio::time::timeout(
                    std::time::Duration::from_millis(deadline_ms),
                    server.call_tool(&name, args, std::time::Duration::from_millis(deadline_ms)),
                )
                .await
                .map_err(|_| {
                    Error::new(
                        ErrorKind::Timeout,
                        format!("mcp tool {name} exceeded its {deadline_ms} ms deadline"),
                    )
                })?
                .map_err(|e| Error::new(ErrorKind::Internal, format!("mcp tool {name}: {e}")))?;
                let mut text = String::new();
                for item in result.content.iter().take(64) {
                    let piece = match item {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    if text.len() + piece.len() > MCP_RESULT_TEXT_MAX {
                        text.push_str("\\n[mcp result truncated]");
                        break;
                    }
                    text.push_str(&piece);
                    text.push('\n');
                }
                Ok(ToolOutcome {
                    text,
                    exit_code: if result.is_error { Some(1) } else { Some(0) },
                    artifact: None,
                    slice_hint: None,
                    // MCP calls have unknown external effects: never replay.
                    effect_status: faktor_core::op::EffectStatus::Unknown,
                    postcondition: None,
                })
            })
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_agent::ToolRunCtx;
    use faktor_core::cancellation::CancellationToken;
    use faktor_core::id::{OpId, SessionId, TaskId, WorkspaceId, WorktreeId};
    use faktor_core::WorkspaceIdentity;

    fn python_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .output()
            .is_ok()
    }

    /// A minimal Content-Length framed JSON-RPC MCP server: initialize,
    /// tools/list (one echo tool), tools/call. Written to a temp file so
    /// spawns stay deterministic.
    /// Path of the checked-in JSON-RPC mock server fixture.
    fn mock_server_path() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mcp_mock.py")
    }

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
            artifacts: Arc::new(faktor_agent::ToolArtifactSink::Null),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            workspace: None,
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            deadline_ms: 10_000,
            permission_granted: true,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn mcp_tool_invocation_end_to_end() {
        if !python_available() {
            eprintln!("python3 missing; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = faktor_terminal::ProcessSupervisor::new(cas);
        let cfg = faktor_mcp::McpConfig {
            name: "mock".into(),
            command: "python3".into(),
            args: vec![mock_server_path().to_str().unwrap().into()],
            env: vec![],
        };
        let server = tokio::time::timeout(
            std::time::Duration::from_secs(15),
            faktor_mcp::McpServer::connect(cfg, sup),
        )
        .await
        .expect("connect timeout")
        .expect("connect failed");
        let tools = server.list_tools().await.unwrap();
        assert!(tools.iter().any(|t| t.name == "mock_echo"));
        let tool = mcp_tool(
            server.clone(),
            tools.iter().find(|t| t.name == "mock_echo").unwrap(),
        );
        // Successful call.
        let out = (tool.execute)(ctx(), serde_json::json!({"text": "hello world"}))
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.text.contains("echo:hello world"));
        // Error result surfaces as a non-zero exit.
        let out = (tool.execute)(ctx(), serde_json::json!({"text": "boom"}))
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(1));
        assert!(out.text.contains("exploded"));
        let _ = server.close().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn daemon_build_spawns_configured_mcp_servers() {
        // Production wiring (spec §31): a kilo-plus.json mcp entry spawns
        // the supervised server at daemon build and its dynamic tool lands
        // in the agent registry next to the builtins.
        if !python_available() {
            eprintln!("python3 missing; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::config::Config {
            mcp: vec![crate::config::McpEntry {
                name: "mock".into(),
                command: "python3".into(),
                args: vec![mock_server_path().to_str().unwrap().into()],
            }],
            ..Default::default()
        };
        let graph = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            crate::build_daemon_with_mcp(dir.path(), Some(cfg)),
        )
        .await
        .expect("daemon build timeout")
        .expect("daemon build failed");
        let (_session, agent, _permissions, servers) = graph;
        assert_eq!(servers.len(), 1, "configured MCP server spawned");
        assert!(
            agent.deps().tools.names().iter().any(|n| n == "mock_echo"),
            "dynamic MCP tool must ride the agent registry: {:?}",
            agent.deps().tools.names()
        );
        // Builtins still present and authoritative.
        for builtin in ["read_file", "write_file", "search", "run_command"] {
            assert!(agent.deps().tools.names().iter().any(|n| n == builtin));
        }
        for srv in servers {
            let _ = srv.close().await;
        }
    }
}
