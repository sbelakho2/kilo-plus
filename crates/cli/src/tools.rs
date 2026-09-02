//! Built-in tools for the daemon (spec §17/§22/§30). Tools never touch
//! session persistence; every invocation carries its workspace identity and
//! runs through the permission engine.
//!
//! The real stack: paths are resolved canonical/symlink-safe against the
//! session workspace, every call is gated by a per-call capability through
//! the sandbox, reads are bounded before any byte enters RAM, writes are
//! transactional (optimistic hash + atomic replace) and checkpointed into
//! the CAS, and commands run under the process supervisor (no orphans,
//! ring-buffer output, CAS spill). A ctx missing any of these components
//! errors honestly — no tool silently falls back to raw std::fs.

use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use kilop_agent::{RecoveryHint, Tool, ToolOutcome, ToolRunCtx};
use kilop_core::capability::{Capability, PermissionDecision};
use kilop_core::error::{Error, ErrorKind};
use kilop_core::op::EffectStatus;
use kilop_core::resource::ResourceClass;
use kilop_edit::{EditOp, EditRequest, RepairMode};
use kilop_fs::WorkspaceHandle;
use kilop_sandbox::PermissionEngine;
use kilop_terminal::{ProcessOwner, SpawnConfig};

const READ_DEFAULT_MAX: usize = 64 * 1024;
const READ_HARD_MAX: usize = 4 * 1024 * 1024;
const WRITE_MAX_BYTES: usize = 16 * 1024 * 1024;
const SEARCH_PER_FILE: usize = 2 * 1024 * 1024;
const SEARCH_MAX_HITS: usize = 64;
const SEARCH_MAX_DEPTH: usize = 16;
const COMMAND_MAX_LEN: usize = 4096;
const COMMAND_DEFAULT_DEADLINE_MS: u64 = 30_000;
const COMMAND_ARTIFACT_MAX: usize = 1024 * 1024;

/// The file tools REQUIRE a workspace + sandbox: a ctx without them (tests,
/// mis-wired daemons) errors honestly instead of trusting the model path.
fn require_workspace(
    ctx: &ToolRunCtx,
) -> Result<(Arc<WorkspaceHandle>, Arc<PermissionEngine>), Error> {
    let ws = ctx.workspace.clone().ok_or_else(|| {
        Error::permission("tool requires a workspace context (no workspace wired)")
    })?;
    let sandbox = ctx.sandbox.clone().ok_or_else(|| {
        Error::permission("tool requires the permission engine (no sandbox wired)")
    })?;
    Ok((ws, sandbox))
}

/// Evaluate one capability. Hard DENY always refuses (workspace
/// containment + explicit rules). An Ask-policy verdict refuses ONLY when
/// the runtime did not already resolve the interactive hop to Allow —
/// `ctx.permission_granted` is set by the agent runtime after its
/// permission request came back Allow, so the daemon's UI approval reaches
/// the tool. A direct, permission-less invocation never silently continues
/// on Ask.
fn sandbox_gate(
    ctx: &ToolRunCtx,
    sandbox: &PermissionEngine,
    capability: &Capability,
    what: &str,
) -> Result<(), Error> {
    match sandbox.evaluate(capability) {
        PermissionDecision::Allow => Ok(()),
        PermissionDecision::Deny => Err(Error::permission(format!("{what} denied by sandbox"))),
        PermissionDecision::Ask => {
            if ctx.permission_granted {
                Ok(())
            } else {
                Err(Error::permission(format!("permission required: {what}")))
            }
        }
    }
}

/// Bounded read: metadata FIRST, then at most `max + 1` bytes. A 30GB file
/// never enters RAM — the size check happens before any byte is read.
fn bounded_read(path: &Path, max: usize) -> Result<Vec<u8>, Error> {
    let meta = std::fs::metadata(path).map_err(|e| err_not_found(path, e))?;
    let mut f = std::fs::File::open(path).map_err(|e| err_not_found(path, e))?;
    let mut bytes = Vec::new();
    if meta.len() > max as u64 {
        bytes.resize(max + 1, 0);
        f.read_exact(&mut bytes)
            .map_err(|e| Error::internal(format!("read {}: {e}", path.display())))?;
    } else {
        f.read_to_end(&mut bytes)
            .map_err(|e| Error::internal(format!("read {}: {e}", path.display())))?;
    }
    Ok(bytes)
}

fn err_not_found(path: &Path, e: std::io::Error) -> Error {
    if e.kind() == std::io::ErrorKind::NotFound {
        Error::not_found(format!("{}", path.display()))
    } else {
        Error::new(ErrorKind::Internal, format!("{}: {e}", path.display()))
    }
}

pub fn read_file_tool() -> Tool {
    Tool {
        name: "read_file".into(),
        description: "Read a file within the workspace (bounded, sandboxed).".into(),
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
        path_args: vec!["path".into()],
        execute: Arc::new(|ctx, args| {
            Box::pin(async move {
                let (ws, sandbox) = require_workspace(&ctx)?;
                let path = args
                    .get("path")
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| Error::malformed("read_file requires path"))?;
                let max = args
                    .get("max_bytes")
                    .and_then(|m| m.as_u64())
                    .unwrap_or(READ_DEFAULT_MAX as u64) as usize;
                let max = max.clamp(1, READ_HARD_MAX);
                let rel = Path::new(path);
                // Canonical/symlink-safe resolution against the workspace
                // root; the capability is derived from the RESOLVED path.
                let resolved = ws.resolve(rel)?;
                sandbox_gate(
                    &ctx,
                    &sandbox,
                    &Capability::ReadWorkspace {
                        path: resolved.clone(),
                    },
                    "read_file",
                )?;
                let data = bounded_read(&resolved, max)?;
                let truncated = data.len() > max;
                let bytes = if truncated { &data[..max] } else { &data[..] };
                let text = String::from_utf8_lossy(bytes).to_string();
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
        description: "Write a file within the workspace (transactional, checkpointed).".into(),
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
        path_args: vec!["path".into()],
        execute: Arc::new(|ctx, args| {
            Box::pin(async move {
                let (ws, sandbox) = require_workspace(&ctx)?;
                let edit = ctx.edit.clone().ok_or_else(|| {
                    Error::permission("write_file requires the edit engine (none wired)")
                })?;
                let snapshots = ctx.snapshots.clone().ok_or_else(|| {
                    Error::permission("write_file requires the checkpoint store (none wired)")
                })?;
                let path = args
                    .get("path")
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| Error::malformed("write_file requires path"))?;
                let content = args
                    .get("content")
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| Error::malformed("write_file requires content"))?;
                if content.len() > WRITE_MAX_BYTES {
                    return Err(Error::oversized("write_file content exceeds 16MB"));
                }
                let rel = Path::new(path);
                let resolved = ws.resolve(rel)?;
                sandbox_gate(
                    &ctx,
                    &sandbox,
                    &Capability::WriteWorkspace {
                        path: resolved.clone(),
                    },
                    "write_file",
                )?;

                // Optimistic base: what the model would have read. A file
                // changed between this read and the edit-engine apply is a
                // Conflict (never a blind overwrite).
                let current = match ws.read(rel, WRITE_MAX_BYTES) {
                    Ok(data) => Some(data),
                    Err(e) if e.kind == ErrorKind::NotFound => None,
                    Err(e) => return Err(e),
                };
                if let Some(cur) = &current {
                    if cur.bytes == content.as_bytes() {
                        return Ok(ToolOutcome {
                            text: format!("{path} unchanged ({} bytes)", content.len()),
                            exit_code: Some(0),
                            ..Default::default()
                        });
                    }
                }
                if current.is_none() && content.is_empty() {
                    // Creating an empty file: before == after (empty), which
                    // the checkpoint store refuses as a no-op — there is
                    // nothing to undo, so write without a checkpoint row.
                    ws.write_atomic(rel, b"")?;
                    return Ok(ToolOutcome {
                        text: format!("created {path} (0 bytes)"),
                        exit_code: Some(0),
                        ..Default::default()
                    });
                }

                // Checkpoint the ORIGINAL content into the CAS (deduped)
                // BEFORE the write; after_write records before/after hashes.
                let before_bytes: &[u8] =
                    current.as_ref().map(|c| c.bytes.as_slice()).unwrap_or(b"");
                let before = snapshots.before_write(ctx.session_id, path, before_bytes)?;

                let after = match &current {
                    Some(cur) => {
                        // Transactional full-file replace: validates the
                        // expected hash, parse-checks, and writes atomically
                        // (the engine's temp name carries a uuid nonce, so
                        // parallel writers never collide on temp files).
                        let req = EditRequest {
                            path: path.to_string(),
                            expected_hash: cur.hash,
                            ops: vec![EditOp::Range {
                                start: 0,
                                end: cur.bytes.len(),
                                replacement: content.to_string(),
                            }],
                        };
                        edit.apply(&ws, &ctx.identity, &req, RepairMode::AllowModelRepair)?
                            .new_hash
                    }
                    None => ws.write_atomic(rel, content.as_bytes())?,
                };

                let sequence = snapshots.checkpoints(ctx.session_id)?.len() as i64 + 1;
                snapshots.after_write(
                    ctx.session_id,
                    path,
                    before,
                    after,
                    sequence,
                    content.as_bytes(),
                )?;
                Ok(ToolOutcome {
                    text: format!("wrote {path} ({} bytes)", content.len()),
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
        description: "Substring search over workspace files (bounded, sandboxed).".into(),
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
        path_args: vec!["path".into()],
        execute: Arc::new(|ctx, args| {
            Box::pin(async move {
                let (ws, sandbox) = require_workspace(&ctx)?;
                let pattern = args
                    .get("pattern")
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| Error::malformed("search requires pattern"))?;
                if pattern.len() > 1024 {
                    return Err(Error::oversized("pattern too long"));
                }
                let root_rel = args.get("path").and_then(|p| p.as_str()).unwrap_or(".");
                let root = ws.resolve(Path::new(root_rel))?;
                sandbox_gate(
                    &ctx,
                    &sandbox,
                    &Capability::ReadWorkspace { path: root.clone() },
                    "search",
                )?;
                // Traversal can be large: bounded on a blocking thread.
                let pattern = pattern.to_string();
                let hits = tokio::task::spawn_blocking(move || {
                    walk_search(&root, &pattern, &sandbox, 0, SEARCH_MAX_HITS)
                })
                .await
                .map_err(|e| Error::internal(format!("search task panicked: {e}")))?;
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

/// Bounded workspace walk: skips vcs/build dirs, never follows symlinks
/// (an in-workspace link pointing outside must not leak files), checks each
/// file's size BEFORE reading (2MiB cap), and stops at `max` hits.
fn walk_search(
    dir: &Path,
    pattern: &str,
    sandbox: &PermissionEngine,
    depth: usize,
    max: usize,
) -> Vec<String> {
    let mut hits = Vec::new();
    if depth > SEARCH_MAX_DEPTH {
        return hits;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return hits;
    };
    for entry in entries.flatten() {
        if hits.len() >= max {
            return hits;
        }
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name == ".git" || name.starts_with("target") || name == "node_modules" {
            continue;
        }
        let Ok(ft) = entry.file_type() else {
            continue;
        };
        if ft.is_symlink() {
            continue;
        }
        if ft.is_dir() {
            hits.extend(walk_search(&path, pattern, sandbox, depth + 1, max));
            continue;
        }
        if !sandbox.is_within_workspace(&path) {
            continue;
        }
        if let Ok(bytes) = bounded_read(&path, SEARCH_PER_FILE) {
            if bytes.len() <= SEARCH_PER_FILE && String::from_utf8_lossy(&bytes).contains(pattern) {
                hits.push(path.to_string_lossy().to_string());
            }
        }
    }
    hits
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
        path_args: vec![],
        execute: Arc::new(|ctx, args| {
            Box::pin(async move {
                let (ws, sandbox) = require_workspace(&ctx)?;
                let supervisor = ctx.supervisor.clone().ok_or_else(|| {
                    Error::permission("run_command requires the process supervisor (none wired)")
                })?;
                let command = args
                    .get("command")
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| Error::malformed("run_command requires command"))?;
                if command.len() > COMMAND_MAX_LEN {
                    return Err(Error::oversized("command too long"));
                }
                sandbox_gate(
                    &ctx,
                    &sandbox,
                    &Capability::ExecuteShell {
                        command: command.to_string(),
                    },
                    "run_command",
                )?;
                let deadline_ms = if ctx.deadline_ms > 0 {
                    ctx.deadline_ms
                } else {
                    COMMAND_DEFAULT_DEADLINE_MS
                };
                let cfg = SpawnConfig {
                    cmd: "sh".into(),
                    args: vec!["-c".into(), command.to_string()],
                    cwd: ws.root().to_path_buf(),
                    env: vec![],
                    owner: ProcessOwner::Session(ctx.session_id),
                    capture: true,
                    artifact_max: COMMAND_ARTIFACT_MAX,
                };
                let out = supervisor
                    .run(
                        cfg,
                        std::time::Duration::from_millis(deadline_ms),
                        ctx.cancellation.clone(),
                    )
                    .await?;
                Ok(ToolOutcome {
                    text: out.excerpt,
                    exit_code: out.exit_code,
                    artifact: out.artifact,
                    slice_hint: out.slice_hint,
                    // A shell command's external effects are never known:
                    // mark unknown so crash recovery forces verification
                    // (commandment 6).
                    effect_status: EffectStatus::Unknown,
                })
            })
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_agent::{ToolArtifactSink, ToolCallMode};
    use kilop_core::cancellation::CancellationToken;
    use kilop_core::hash::FileHash;
    use kilop_core::id::{OpId, SessionId, TaskId, WorkspaceId, WorktreeId};
    use kilop_core::WorkspaceIdentity;
    use kilop_sandbox::{Rule, SandboxPolicy};
    use kilop_session::SessionManager;
    use kilop_terminal::ProcessSupervisor;
    use std::path::PathBuf;

    struct ToolFixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        session: SessionId,
        identity: WorkspaceIdentity,
        sandbox: Arc<PermissionEngine>,
        snapshots: Arc<kilop_snapshot::CheckpointStore>,
        cas: Arc<kilop_cas::Cas>,
    }

    fn fixture(policy: SandboxPolicy) -> ToolFixture {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ws");
        std::fs::create_dir_all(&root).unwrap();
        let manager =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let ws_id = manager.create_workspace(root.to_str().unwrap()).unwrap();
        let row = manager
            .create_session(ws_id, "tools test", "fake", "m")
            .unwrap();
        let fs_service = kilop_fs::WorkspaceFileService::new();
        let _opened = fs_service.open(ws_id, root.clone()).unwrap();
        let identity = WorkspaceIdentity::new(ws_id, WorktreeId::new(1), TaskId::new(1));
        let cas = manager.cas();
        ToolFixture {
            _dir: dir,
            root: root.clone(),
            session: row.id(),
            identity,
            sandbox: Arc::new(PermissionEngine::new(policy, Some(root))),
            snapshots: Arc::new(kilop_snapshot::CheckpointStore::new(
                cas.clone(),
                manager.store(),
            )),
            cas,
        }
    }

    fn ctx(f: &ToolFixture) -> ToolRunCtx {
        ctx_granted(f, false)
    }

    fn ctx_granted(f: &ToolFixture, granted: bool) -> ToolRunCtx {
        let fs_service = kilop_fs::WorkspaceFileService::new();
        let workspace = fs_service
            .open(f.identity.workspace_id, f.root.clone())
            .unwrap();
        ToolRunCtx {
            session_id: f.session,
            op_id: OpId::new(1),
            identity: f.identity,
            cancellation: CancellationToken::new(),
            artifacts: Arc::new(ToolArtifactSink::Null),
            tool_call_mode: ToolCallMode::Native,
            workspace: Some(Arc::new(workspace)),
            edit: Some(Arc::new(kilop_edit::EditEngine::new(fs_service.clone()))),
            snapshots: Some(f.snapshots.clone()),
            sandbox: Some(f.sandbox.clone()),
            supervisor: Some(ProcessSupervisor::new(f.cas.clone())),
            deadline_ms: 0,
            permission_granted: granted,
        }
    }

    fn bare_ctx() -> ToolRunCtx {
        ToolRunCtx {
            session_id: SessionId::new(1),
            op_id: OpId::new(1),
            identity: WorkspaceIdentity::new(
                WorkspaceId::new(1),
                WorktreeId::new(1),
                TaskId::new(1),
            ),
            cancellation: CancellationToken::new(),
            artifacts: Arc::new(ToolArtifactSink::Null),
            tool_call_mode: ToolCallMode::Native,
            workspace: None,
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            deadline_ms: 0,
            permission_granted: false,
        }
    }

    #[tokio::test]
    async fn read_file_bounds_and_truncates() {
        let f = fixture(SandboxPolicy::default());
        std::fs::write(f.root.join("f.txt"), "x".repeat(10_000)).unwrap();
        let tool = read_file_tool();
        let out = (tool.execute)(
            ctx(&f),
            serde_json::json!({"path": "f.txt", "max_bytes": 100}),
        )
        .await
        .unwrap();
        assert!(out.text.contains("truncated at 100 bytes"));
        let out = (tool.execute)(ctx(&f), serde_json::json!({"path": "f.txt"}))
            .await
            .unwrap();
        assert!(!out.text.contains("truncated"));
    }

    #[tokio::test]
    async fn read_bounded_truncates_never_reads_whole_file() {
        // A 10MB file with max_bytes=1KB: the tool must return only the
        // bounded prefix (the old tool read the whole 10MB then sliced).
        let f = fixture(SandboxPolicy::default());
        std::fs::write(f.root.join("big.bin"), vec![7u8; 10 * 1024 * 1024]).unwrap();
        let tool = read_file_tool();
        let out = (tool.execute)(
            ctx(&f),
            serde_json::json!({"path": "big.bin", "max_bytes": 1024}),
        )
        .await
        .unwrap();
        assert!(out.text.contains("[truncated at 1024 bytes]"));
        assert!(
            out.text.len() < 2 * 1024,
            "only the bounded prefix may be returned, got {} bytes",
            out.text.len()
        );
    }

    #[tokio::test]
    async fn read_file_missing_is_not_found() {
        let f = fixture(SandboxPolicy::default());
        let tool = read_file_tool();
        let err = (tool.execute)(ctx(&f), serde_json::json!({"path": "nope.rs"}))
            .await
            .unwrap_err();
        assert!(err.kind == ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn path_traversal_and_symlink_escape_rejected() {
        // These assertions FAIL on the old tools: they read with
        // std::fs::read(path) and walked with std::fs::read_dir, trusting
        // the model-supplied path — an in-workspace symlink pointing
        // outside was followed and "../" traversals escaped the workspace.
        let f = fixture(SandboxPolicy::default());
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "needle-in-secret").unwrap();
        std::os::unix::fs::symlink(outside.path(), f.root.join("link")).unwrap();

        let read = read_file_tool();
        let err = (read.execute)(ctx(&f), serde_json::json!({"path": "link/secret.txt"}))
            .await
            .unwrap_err();
        assert_eq!(
            err.kind,
            ErrorKind::Permission,
            "read must reject symlink escape"
        );

        let write = write_file_tool();
        let err = (write.execute)(
            ctx(&f),
            serde_json::json!({"path": "link/secret.txt", "content": "pwned"}),
        )
        .await
        .unwrap_err();
        assert_eq!(
            err.kind,
            ErrorKind::Permission,
            "write must reject symlink escape"
        );

        let err = (read.execute)(ctx(&f), serde_json::json!({"path": "../etc/passwd"}))
            .await
            .unwrap_err();
        assert_eq!(
            err.kind,
            ErrorKind::Permission,
            "traversal must be rejected"
        );

        let search = search_tool();
        let out = (search.execute)(ctx(&f), serde_json::json!({"pattern": "needle-in-secret"}))
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(1));
        assert!(
            !out.text.contains("needle-in-secret"),
            "search must never follow the escaping symlink"
        );
    }

    #[tokio::test]
    async fn write_denied_by_sandbox_policy() {
        let f = fixture(SandboxPolicy {
            write_workspace: Rule::Deny,
            ..Default::default()
        });
        let tool = write_file_tool();
        let err = (tool.execute)(
            ctx(&f),
            serde_json::json!({"path": "blocked.txt", "content": "x"}),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
        assert!(
            !f.root.join("blocked.txt").exists(),
            "denied write must not create the file"
        );
    }

    #[tokio::test]
    async fn write_produces_checkpoint_with_before_after_hashes() {
        let f = fixture(SandboxPolicy::default());
        std::fs::write(f.root.join("a.txt"), "original").unwrap();
        let tool = write_file_tool();
        (tool.execute)(
            ctx(&f),
            serde_json::json!({"path": "a.txt", "content": "new content"}),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read(f.root.join("a.txt")).unwrap(), b"new content");

        let rows = f.snapshots.checkpoints(f.session).unwrap();
        assert_eq!(rows.len(), 1, "a write must record exactly one checkpoint");
        // The CAS is the hash source of truth: putting the bytes returns
        // the exact FileHash the checkpoint must record.
        let before = f.cas.put(b"original").unwrap();
        let after = f.cas.put(b"new content").unwrap();
        assert_eq!(FileHash::from_hex(&rows[0].before_hash).unwrap(), before);
        assert_eq!(FileHash::from_hex(&rows[0].after_hash).unwrap(), after);
        assert_eq!(f.cas.get(before).unwrap(), b"original");
        assert_eq!(f.cas.get(after).unwrap(), b"new content");

        // An unchanged rewrite must NOT record a second checkpoint (the
        // store rejects no-op checkpoints as malformed).
        (tool.execute)(
            ctx(&f),
            serde_json::json!({"path": "a.txt", "content": "new content"}),
        )
        .await
        .unwrap();
        assert_eq!(f.snapshots.checkpoints(f.session).unwrap().len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn write_conflict_when_file_changed_between_read_and_write() {
        // The tool's expected hash is the current content at ITS read; when
        // an external write lands between that read and the edit-engine
        // apply, the optimistic check must surface Conflict and the file
        // must be untouched by the agent. Two tool writers can never race
        // here (both would re-read fresh content), so the adversarial actor
        // is an external writer. The current file is 16MB of incompressible
        // bytes: the tool's before-write CAS store (which sits between read
        // and apply) then takes tens of ms on any machine, so the external
        // write at +15ms lands deterministically inside the window.
        let f = fixture(SandboxPolicy::default());
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let noise: Vec<u8> = (0..16 * 1024 * 1024)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (seed >> 33) as u8
            })
            .collect();
        std::fs::write(f.root.join("race.txt"), &noise).unwrap();
        let tool = Arc::new(write_file_tool());
        let writer = tokio::spawn({
            let tool = tool.clone();
            let c = ctx(&f);
            async move {
                (tool.execute)(
                    c,
                    serde_json::json!({"path": "race.txt", "content": "agent write"}),
                )
                .await
            }
        });
        tokio::time::sleep(std::time::Duration::from_millis(15)).await;
        std::fs::write(f.root.join("race.txt"), b"external edit").unwrap();
        let err = writer.await.unwrap().unwrap_err();
        assert_eq!(
            err.kind,
            ErrorKind::Conflict,
            "the stale expected hash must surface as Conflict: {err:?}"
        );
        assert_eq!(
            std::fs::read(f.root.join("race.txt")).unwrap(),
            b"external edit",
            "a conflicted write must leave the file untouched"
        );
        assert_eq!(
            f.snapshots.checkpoints(f.session).unwrap().len(),
            0,
            "a conflicted write must record no checkpoint"
        );
    }

    #[tokio::test]
    async fn parallel_writes_new_file_both_succeed_no_temp_collisions() {
        // Two parallel writes to the SAME new path: the old temp name
        // (target + pid) collided; the engine's uuid temp nonce makes both
        // atomic writes safe. Final content is one complete variant.
        let f = fixture(SandboxPolicy::default());
        let tool = Arc::new(write_file_tool());
        let variant_a = format!("AAAA-{}", "a".repeat(5000));
        let variant_b = format!("BBBB-{}", "b".repeat(5000));
        let (r1, r2) = tokio::join!(
            (tool.clone().execute)(
                ctx(&f),
                serde_json::json!({"path": "new.txt", "content": variant_a}),
            ),
            (tool.execute)(
                ctx(&f),
                serde_json::json!({"path": "new.txt", "content": variant_b}),
            ),
        );
        r1.unwrap();
        r2.unwrap();
        let final_bytes = std::fs::read(f.root.join("new.txt")).unwrap();
        assert!(
            final_bytes == variant_a.as_bytes() || final_bytes == variant_b.as_bytes(),
            "final content must be one complete variant, got {} bytes",
            final_bytes.len()
        );
        for entry in std::fs::read_dir(f.root).unwrap().flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            assert!(!name.contains("kp-tmp-"), "no temp file may leak: {name}");
        }
    }

    #[tokio::test]
    async fn search_skips_vcs_and_caps_results() {
        let f = fixture(SandboxPolicy::default());
        std::fs::create_dir_all(f.root.join(".git")).unwrap();
        std::fs::write(f.root.join(".git/config"), "needle here").unwrap();
        for i in 0..100 {
            std::fs::write(f.root.join(format!("f{i:03}.txt")), "needle found").unwrap();
        }
        let tool = search_tool();
        let out = (tool.execute)(ctx(&f), serde_json::json!({"pattern": "needle"}))
            .await
            .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(!out.text.contains(".git"), "vcs dirs must be skipped");
        assert!(
            out.text.lines().count() <= 64,
            "search must cap at 64 hits, got {}",
            out.text.lines().count()
        );
        // The old tool's depth cap and size cap held; a too-big file is
        // never read whole.
        std::fs::write(f.root.join("huge.txt"), vec![b'n'; 5 * 1024 * 1024]).unwrap();
        let out = (tool.execute)(ctx(&f), serde_json::json!({"pattern": "n"}))
            .await
            .unwrap();
        assert!(
            !out.text.contains("huge.txt"),
            "oversized files are skipped"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_command_executes_via_sh() {
        let f = fixture(SandboxPolicy {
            execute_shell: Rule::Allow,
            ..Default::default()
        });
        let tool = run_command_tool();
        let out = (tool.execute)(
            ctx(&f),
            serde_json::json!({"command": "echo hello-from-tool"}),
        )
        .await
        .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.text.contains("hello-from-tool"), "{:?}", out.text);
        assert_eq!(out.effect_status, EffectStatus::Unknown);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_command_respects_cancellation() {
        let f = fixture(SandboxPolicy {
            execute_shell: Rule::Allow,
            ..Default::default()
        });
        let mut c = ctx(&f);
        c.deadline_ms = 60_000;
        let tool = run_command_tool();
        let task = tokio::spawn({
            let c = c.clone();
            let tool = tool.clone();
            async move { (tool.execute)(c, serde_json::json!({"command": "sleep 30"})).await }
        });
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        c.cancellation.cancel();
        let err = task.await.unwrap().unwrap_err();
        assert_eq!(err.kind, ErrorKind::Cancelled);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_command_large_output_spills_to_cas() {
        let f = fixture(SandboxPolicy {
            execute_shell: Rule::Allow,
            ..Default::default()
        });
        let tool = run_command_tool();
        let out = (tool.execute)(
            ctx(&f),
            serde_json::json!({
                "command": "dd if=/dev/zero bs=1048576 count=2 2>/dev/null | tr '\\0' 'a' | fold -w 100"
            }),
        )
        .await
        .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.text.len() < 64 * 1024, "excerpt must stay bounded");
        let artifact = out
            .artifact
            .as_ref()
            .expect("overflow must spill to the CAS");
        let hash = artifact
            .strip_prefix("artifact://")
            .and_then(FileHash::from_hex)
            .expect("artifact ref must carry a CAS hash");
        let blob = f.cas.get(hash).unwrap();
        assert!(
            blob.len() > 1024 * 1024,
            "the spill must hold the full overflow, got {} bytes",
            blob.len()
        );
        assert!(
            blob.iter().all(|b| *b == b'a' || *b == b'\n'),
            "spill content must be complete fold lines"
        );
    }

    #[tokio::test]
    async fn run_command_ask_policy_errors_without_runtime_grant() {
        // Tool-level (no runtime hop): Ask must still refuse — a direct
        // invocation never silently continues.
        let f = fixture(SandboxPolicy::default()); // execute_shell: Ask
        let tool = run_command_tool();
        let err = (tool.execute)(ctx(&f), serde_json::json!({"command": "echo x"}))
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
    }

    #[tokio::test]
    async fn run_command_ask_policy_runs_after_runtime_grant() {
        // The daemon flow: the runtime resolved the interactive permission
        // hop to Allow (permission_granted). An Ask-policy verdict must NOT
        // hard-error the tool after the user approved in the UI — this was
        // the audit-round bug (UI approval could never reach the tool).
        let f = fixture(SandboxPolicy::default());
        let tool = run_command_tool();
        let outcome = (tool.execute)(
            ctx_granted(&f, true),
            serde_json::json!({"command": "echo granted"}),
        )
        .await
        .unwrap();
        assert_eq!(outcome.exit_code, Some(0));
        assert!(outcome.text.contains("granted"));
    }

    #[tokio::test]
    async fn hard_deny_never_yields_to_runtime_grant() {
        // A policy DENY is a hard sandbox invariant: even an approved
        // runtime hop cannot read outside the workspace.
        let f = fixture(SandboxPolicy {
            read_external: Rule::Deny,
            write_external: Rule::Deny,
            execute_shell: Rule::Allow,
            ..Default::default()
        });
        let outside = f._dir.path().join("outside.txt");
        std::fs::write(&outside, "x").unwrap();
        let tool = read_file_tool();
        let args = serde_json::json!({"path": outside.to_str().unwrap()});
        let err = (tool.execute)(ctx_granted(&f, true), args)
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
    }

    #[tokio::test]
    async fn run_command_ask_policy_errors_before_spawn() {
        let f = fixture(SandboxPolicy::default()); // execute_shell: Ask
        let tool = run_command_tool();
        let err = (tool.execute)(ctx(&f), serde_json::json!({"command": "echo x"}))
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
    }

    #[tokio::test]
    async fn no_workspace_tools_error_honestly() {
        for (tool, args) in [
            (read_file_tool(), serde_json::json!({"path": "a.txt"})),
            (
                write_file_tool(),
                serde_json::json!({"path": "a.txt", "content": "x"}),
            ),
            (search_tool(), serde_json::json!({"pattern": "x"})),
            (run_command_tool(), serde_json::json!({"command": "echo x"})),
        ] {
            let err = (tool.execute)(bare_ctx(), args).await.unwrap_err();
            assert_eq!(err.kind, ErrorKind::Permission);
        }
    }

    #[tokio::test]
    async fn run_command_validates_input() {
        let f = fixture(SandboxPolicy {
            execute_shell: Rule::Allow,
            ..Default::default()
        });
        let tool = run_command_tool();
        let err = (tool.execute)(ctx(&f), serde_json::json!({"command": "x".repeat(5000)}))
            .await
            .unwrap_err();
        assert!(err.kind == ErrorKind::Oversized);
    }

    #[tokio::test]
    async fn malicious_args_never_panic() {
        let f = fixture(SandboxPolicy::default());
        for tool in [
            read_file_tool(),
            write_file_tool(),
            search_tool(),
            run_command_tool(),
        ] {
            for args in [
                serde_json::json!({}),
                serde_json::json!({"path": 42}),
                serde_json::json!({"path": ["a"]}),
                serde_json::json!({"path": "\u{0}"}),
                serde_json::json!({"path": "x", "content": 7}),
                serde_json::json!({"command": 7}),
            ] {
                let _ = (tool.clone().execute)(ctx(&f), args).await;
            }
        }
    }
}
