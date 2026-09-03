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

use faktor_agent::{FilePostcondition, RecoveryHint, Tool, ToolOutcome, ToolRunCtx};
use faktor_core::capability::{Capability, PermissionDecision};
use faktor_core::error::{Error, ErrorKind};
use faktor_core::hash::FileHash;
use faktor_core::op::EffectStatus;
use faktor_core::resource::ResourceClass;
use faktor_edit::{EditOp, EditRequest, RepairMode};
use faktor_fs::WorkspaceHandle;
use faktor_sandbox::PermissionEngine;
use faktor_terminal::{ProcessOwner, SpawnConfig};

const READ_DEFAULT_MAX: usize = 64 * 1024;
const READ_HARD_MAX: usize = 4 * 1024 * 1024;
const WRITE_MAX_BYTES: usize = 16 * 1024 * 1024;
const SEARCH_PER_FILE: usize = 2 * 1024 * 1024;
const SEARCH_MAX_HITS: usize = 64;
const SEARCH_MAX_DEPTH: usize = 16;
const COMMAND_MAX_LEN: usize = 4096;
const COMMAND_DEFAULT_DEADLINE_MS: u64 = 30_000;
const COMMAND_ARTIFACT_MAX: usize = 1024 * 1024;
/// Maximum operations in one `edit_file` call.
const EDIT_MAX_OPS: usize = 8;
/// Maximum bytes of one operation's text payload (search/replace/anchor/text).
const EDIT_MAX_OP_TEXT_BYTES: usize = 8 * 1024;
/// Maximum total payload bytes across all operations of one call.
const EDIT_MAX_TOTAL_TEXT_BYTES: usize = 32 * 1024;

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
        recovery_hint: RecoveryHint::WorkspaceWrite,
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
                // The postcondition recovery verifies: BLAKE3 of the ACTUAL
                // bytes as written, relative to the workspace root — never a
                // hash of JSON-encoded args and never a daemon-cwd path.
                let postcondition = |after_hash: FileHash| FilePostcondition {
                    workspace_id: ctx.identity.workspace_id,
                    worktree_id: ctx.identity.worktree_id,
                    relative_path: path.to_string(),
                    expected_hash: after_hash,
                };

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
                            postcondition: Some(postcondition(cur.hash)),
                            ..Default::default()
                        });
                    }
                }
                if current.is_none() && content.is_empty() {
                    // Creating an empty file: before == after (empty), which
                    // the checkpoint store refuses as a no-op — there is
                    // nothing to undo, so write without a checkpoint row.
                    let hash = ws.write_atomic(rel, b"")?;
                    return Ok(ToolOutcome {
                        text: format!("created {path} (0 bytes)"),
                        exit_code: Some(0),
                        postcondition: Some(postcondition(hash)),
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
                    postcondition: Some(postcondition(after)),
                    ..Default::default()
                })
            })
        }),
    }
}

/// One parsed `edit_file` operation. All matching is literal; nothing here
/// is a regex.
#[derive(Debug, Clone, PartialEq, Eq)]
enum EditPos {
    Before,
    After,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ToolEditOp {
    /// Replace the FIRST literal occurrence of `search` with `replace`.
    ReplaceExact { search: String, replace: String },
    /// Literal replace; `unique` (default true) requires exactly one
    /// occurrence, `unique: false` replaces the first occurrence.
    SearchReplace {
        search: String,
        replace: String,
        unique: bool,
    },
    /// Insert `text` verbatim before/after the anchor (must match once).
    Insert {
        anchor: String,
        position: EditPos,
        text: String,
    },
    /// Replace whole lines `start_line..=end_line` (1-based, inclusive) with
    /// the line block `text`.
    RegionReplace {
        start_line: usize,
        end_line: usize,
        text: String,
    },
}

impl ToolEditOp {
    fn name(&self) -> &'static str {
        match self {
            ToolEditOp::ReplaceExact { .. } => "replace_exact",
            ToolEditOp::SearchReplace { .. } => "search_replace",
            ToolEditOp::Insert { .. } => "insert",
            ToolEditOp::RegionReplace { .. } => "region_replace",
        }
    }
}

/// Parse + bound the `operations` array of an `edit_file` call: at most
/// [`EDIT_MAX_OPS`] ops, every text payload bounded and the total bounded,
/// all before any filesystem access.
fn parse_edit_ops(args: &serde_json::Value) -> Result<Vec<ToolEditOp>, Error> {
    let arr = args
        .get("operations")
        .and_then(|o| o.as_array())
        .ok_or_else(|| Error::malformed("edit_file requires operations: [ ... ]"))?;
    if arr.is_empty() {
        return Err(Error::malformed(
            "edit_file requires at least one operation",
        ));
    }
    if arr.len() > EDIT_MAX_OPS {
        return Err(Error::oversized(format!(
            "edit_file accepts at most {EDIT_MAX_OPS} operations, got {}",
            arr.len()
        )));
    }
    let mut ops = Vec::with_capacity(arr.len());
    let mut total = 0usize;
    for (i, item) in arr.iter().enumerate() {
        let opn = i + 1;
        let no_name = format!("op {opn}: ");
        let prefix = |name: &str| format!("op {opn} ({name}): ");
        let typ = item
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| Error::malformed(format!("{no_name}missing string `type`")))?;
        let str_field = |name: &str| -> Result<String, Error> {
            item.get(name)
                .and_then(|v| v.as_str())
                .map(String::from)
                .ok_or_else(|| {
                    Error::malformed(format!("{}requires string field `{name}`", prefix(typ)))
                })
        };
        let op = match typ {
            "replace_exact" => {
                let search = str_field("search")?;
                let replace = str_field("replace")?;
                if search.is_empty() {
                    return Err(Error::malformed(format!(
                        "{}`search` must be non-empty",
                        prefix("replace_exact")
                    )));
                }
                ToolEditOp::ReplaceExact { search, replace }
            }
            "search_replace" => {
                let search = str_field("search")?;
                let replace = str_field("replace")?;
                if search.is_empty() {
                    return Err(Error::malformed(format!(
                        "{}`search` must be non-empty",
                        prefix("search_replace")
                    )));
                }
                let unique = match item.get("unique") {
                    None => true,
                    Some(v) => v.as_bool().ok_or_else(|| {
                        Error::malformed(format!(
                            "{}`unique` must be a boolean",
                            prefix("search_replace")
                        ))
                    })?,
                };
                ToolEditOp::SearchReplace {
                    search,
                    replace,
                    unique,
                }
            }
            "insert" => {
                let anchor = str_field("anchor")?;
                let text = str_field("text")?;
                if anchor.is_empty() {
                    return Err(Error::malformed(format!(
                        "{}`anchor` must be non-empty",
                        prefix("insert")
                    )));
                }
                let position = match item.get("position").and_then(|p| p.as_str()) {
                    Some("before") => EditPos::Before,
                    Some("after") => EditPos::After,
                    _ => {
                        return Err(Error::malformed(format!(
                            "{}`position` must be \"before\" or \"after\"",
                            prefix("insert")
                        )))
                    }
                };
                ToolEditOp::Insert {
                    anchor,
                    position,
                    text,
                }
            }
            "region_replace" => {
                let num = |name: &str| -> Result<usize, Error> {
                    item.get(name)
                        .and_then(|v| v.as_u64())
                        .map(|n| n as usize)
                        .ok_or_else(|| {
                            Error::malformed(format!(
                                "{}requires positive integer `{name}`",
                                prefix("region_replace")
                            ))
                        })
                };
                let start_line = num("start_line")?;
                let end_line = num("end_line")?;
                if start_line == 0 {
                    return Err(Error::malformed(format!(
                        "{}line numbers are 1-based",
                        prefix("region_replace")
                    )));
                }
                if end_line < start_line {
                    return Err(Error::malformed(format!(
                        "{}end_line {end_line} < start_line {start_line}",
                        prefix("region_replace")
                    )));
                }
                ToolEditOp::RegionReplace {
                    start_line,
                    end_line,
                    text: str_field("text")?,
                }
            }
            other => {
                return Err(Error::malformed(format!(
                    "{no_name}unknown operation type {other:?}"
                )))
            }
        };
        // Per-op payload bound, then the whole-call total.
        let mut op_bytes = 0usize;
        match &op {
            ToolEditOp::ReplaceExact { search, replace }
            | ToolEditOp::SearchReplace {
                search, replace, ..
            } => {
                op_bytes += search.len();
                op_bytes += replace.len();
            }
            ToolEditOp::Insert { anchor, text, .. } => {
                op_bytes += anchor.len();
                op_bytes += text.len();
            }
            ToolEditOp::RegionReplace { text, .. } => {
                op_bytes += text.len();
            }
        }
        if op_bytes > EDIT_MAX_OP_TEXT_BYTES {
            return Err(Error::oversized(format!(
                "op {opn} ({}): payload of {op_bytes} bytes exceeds the {EDIT_MAX_OP_TEXT_BYTES} byte per-op bound",
                op.name()
            )));
        }
        total += op_bytes;
        if total > EDIT_MAX_TOTAL_TEXT_BYTES {
            return Err(Error::oversized(format!(
                "edit_file total payload of {total} bytes exceeds the {EDIT_MAX_TOTAL_TEXT_BYTES} byte bound"
            )));
        }
        ops.push(op);
    }
    Ok(ops)
}

/// Split a text into its line contents: split on `\n` and drop ONE trailing
/// empty piece (a trailing newline terminates the last line). `""` is zero
/// lines, `"X\nY\n"` is `[X, Y]`, `"\n"` is one blank line.
fn split_text_lines(text: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = text.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}

/// Line contents of the buffer (the trailing newline terminates the last
/// line and is not itself a line). The buffer is always valid UTF-8.
fn buffer_lines(buf: &str) -> Vec<&str> {
    let mut lines: Vec<&str> = buf.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    lines
}

/// Apply ONE operation to the in-memory buffer. Errors are LOUD: they name
/// the 1-based operation index and its type. Validation happens against the
/// evolving buffer, so an anchor that an earlier op moved is "missing",
/// never silently re-located.
fn apply_edit_op(buf: &mut String, op: &ToolEditOp, idx: usize) -> Result<(), Error> {
    let fail = |kind: ErrorKind, msg: String| -> Error {
        Error::new(kind, format!("op {} ({}): {msg}", idx + 1, op.name()))
    };
    match op {
        ToolEditOp::ReplaceExact { search, replace } => match buf.find(search.as_str()) {
            Some(start) => {
                buf.replace_range(start..start + search.len(), replace);
                Ok(())
            }
            None => Err(fail(
                ErrorKind::Malformed,
                "search text not found (0 matches)".into(),
            )),
        },
        ToolEditOp::SearchReplace {
            search,
            replace,
            unique,
        } => {
            let matches = buf.match_indices(search.as_str()).count();
            if matches == 0 {
                return Err(fail(
                    ErrorKind::Malformed,
                    "search text not found (0 matches)".into(),
                ));
            }
            if *unique && matches > 1 {
                return Err(fail(
                    ErrorKind::Conflict,
                    format!("search text is ambiguous ({matches} matches); widen the context or set unique: false"),
                ));
            }
            let start = buf.find(search.as_str()).expect("non-zero matches");
            buf.replace_range(start..start + search.len(), replace);
            Ok(())
        }
        ToolEditOp::Insert {
            anchor,
            position,
            text,
        } => {
            let matches: Vec<usize> = buf.match_indices(anchor.as_str()).map(|(i, _)| i).collect();
            match matches.len() {
                0 => Err(fail(
                    ErrorKind::Malformed,
                    "anchor not found (0 matches)".into(),
                )),
                1 => {
                    let at = matches[0];
                    match position {
                        EditPos::Before => buf.insert_str(at, text),
                        EditPos::After => buf.insert_str(at + anchor.len(), text),
                    }
                    Ok(())
                }
                n => Err(fail(
                    ErrorKind::Conflict,
                    format!(
                        "anchor must match uniquely ({n} matches); include surrounding context"
                    ),
                )),
            }
        }
        ToolEditOp::RegionReplace {
            start_line,
            end_line,
            text,
        } => {
            let lines = buffer_lines(buf);
            if *start_line > lines.len() || *end_line > lines.len() {
                return Err(fail(
                    ErrorKind::Malformed,
                    format!(
                        "line range {start_line}..={end_line} exceeds the file's {} lines",
                        lines.len()
                    ),
                ));
            }
            if *start_line == 1 && *end_line == lines.len() {
                return Err(fail(
                    ErrorKind::Malformed,
                    "whole-file replacement is not offered by edit_file; use write_file".into(),
                ));
            }
            // Whole-line block semantics: the region consumes the line
            // CONTENTS; every surviving line is re-terminated, so the only
            // adjustment is a file that did NOT end with a newline (its last
            // line must stay unterminated).
            let mut out = String::new();
            for l in &lines[..*start_line - 1] {
                out.push_str(l);
                out.push('\n');
            }
            for l in split_text_lines(text) {
                out.push_str(l);
                out.push('\n');
            }
            for l in &lines[*end_line..] {
                out.push_str(l);
                out.push('\n');
            }
            if !buf.ends_with('\n') && !out.is_empty() {
                out.pop();
            }
            *buf = out;
            Ok(())
        }
    }
}

/// `edit_file`: precise, bounded, transactional edits (P1 "agent-exposed
/// editing is too coarse"). All operations run against ONE file read with an
/// optional `expected_hash` staleness preimage check; they apply IN ORDER to
/// an in-memory copy and the result is written ONCE through the edit
/// engine's atomic write (expected-hash + parse validation) — a failing
/// operation N leaves the file byte-identical and the error names
/// operation N.
///
/// Escalation chain: `replace_exact` (first literal occurrence) → unique
/// `search_replace` (refuses ambiguity) → `insert` (unique anchor) →
/// bounded `region_replace` (whole lines by number). Whole-file replacement
/// is NOT offered by this tool — use `write_file` for that.
///
/// The outcome reports the new `expected_hash`; pass it as the next
/// `expected_hash` so a stale follow-up edit is rejected instead of
/// overwriting a concurrent change.
pub fn edit_file_tool() -> Tool {
    Tool {
        name: "edit_file".into(),
        description: "Edit a file with precise bounded operations (atomic; nothing is written unless EVERY operation succeeds). Input: {path, expected_hash? (BLAKE3 hex from the last read_file/edit_file/write_file outcome), operations: [op, ...]} (1..=8 ops). Op types: (1) replace_exact {search, replace} — replace the FIRST literal occurrence; (2) search_replace {search, replace, unique?} — unique (default true) requires exactly one match, ambiguous matches REFUSE with no write; unique: false replaces the first occurrence; (3) insert {anchor, position: before|after, text} — anchor must match uniquely; (4) region_replace {start_line, end_line, text} — replace whole lines (1-based inclusive) with the given line block; empty text deletes the lines. Escalation: exact -> unique search/replace -> bounded region; whole-file replacement is NOT offered (use write_file). Staleness: when expected_hash is supplied and the file changed since, the edit REFUSES (conflict) without writing. When a later operation fails, the error names the operation and the file is left UNCHANGED. The reply reports the file's new expected_hash — send it back as the next expected_hash. Paths resolve against the session workspace (sandboxed, symlink-safe).".into(),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "expected_hash": { "type": "string" },
                "operations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 8,
                    "items": {
                        "oneOf": [
                            { "type": "object", "properties": { "type": {"const": "replace_exact"}, "search": {"type": "string"}, "replace": {"type": "string"} }, "required": ["type", "search", "replace"] },
                            { "type": "object", "properties": { "type": {"const": "search_replace"}, "search": {"type": "string"}, "replace": {"type": "string"}, "unique": {"type": "boolean"} }, "required": ["type", "search", "replace"] },
                            { "type": "object", "properties": { "type": {"const": "insert"}, "anchor": {"type": "string"}, "position": {"enum": ["before", "after"]}, "text": {"type": "string"} }, "required": ["type", "anchor", "position", "text"] },
                            { "type": "object", "properties": { "type": {"const": "region_replace"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}, "text": {"type": "string"} }, "required": ["type", "start_line", "end_line", "text"] }
                        ]
                    }
                }
            },
            "required": ["path", "operations"]
        }),
        resource_class: ResourceClass::DiskWrite,
        capability: Some(Capability::WriteWorkspace { path: ".".into() }),
        recovery_hint: RecoveryHint::WorkspaceWrite,
        path_args: vec!["path".into()],
        execute: Arc::new(|ctx, args| {
            Box::pin(async move {
                let (ws, sandbox) = require_workspace(&ctx)?;
                let edit = ctx.edit.clone().ok_or_else(|| {
                    Error::permission("edit_file requires the edit engine (none wired)")
                })?;
                let snapshots = ctx.snapshots.clone().ok_or_else(|| {
                    Error::permission("edit_file requires the checkpoint store (none wired)")
                })?;
                // Parse + bounds FIRST: a hostile payload never touches disk.
                let ops = parse_edit_ops(&args)?;
                let path = args
                    .get("path")
                    .and_then(|p| p.as_str())
                    .ok_or_else(|| Error::malformed("edit_file requires path"))?;
                let expected = match args.get("expected_hash").and_then(|h| h.as_str()) {
                    None => None,
                    Some(raw) => Some(
                        FileHash::from_hex(raw).ok_or_else(|| {
                            Error::malformed("expected_hash must be the 64-char hex BLAKE3 of the file")
                        })?,
                    ),
                };
                let rel = Path::new(path);
                let resolved = ws.resolve(rel)?;
                sandbox_gate(
                    &ctx,
                    &sandbox,
                    &Capability::WriteWorkspace {
                        path: resolved.clone(),
                    },
                    "edit_file",
                )?;
                let postcondition = |after_hash: FileHash| FilePostcondition {
                    workspace_id: ctx.identity.workspace_id,
                    worktree_id: ctx.identity.worktree_id,
                    relative_path: path.to_string(),
                    expected_hash: after_hash,
                };
                // One read: the buffer every operation runs against.
                let current = ws
                    .read(rel, WRITE_MAX_BYTES)
                    .map_err(|e| match e.kind {
                        ErrorKind::NotFound => Error::new(
                            ErrorKind::NotFound,
                            format!("{path} does not exist; use write_file to create files"),
                        ),
                        _ => e,
                    })?;
                if current.truncated {
                    return Err(Error::oversized(format!(
                        "{path} exceeds the {} byte edit bound",
                        WRITE_MAX_BYTES
                    )));
                }
                // Optional staleness preimage check (adversarial: refuse a
                // stale edit loudly instead of overwriting).
                if let Some(expected) = expected {
                    if current.hash != expected {
                        return Err(Error::conflict(format!(
                            "{path} changed since it was read (expected {}, found {}); re-read and retry",
                            expected.to_hex(),
                            current.hash.to_hex()
                        )));
                    }
                }
                let original = String::from_utf8(current.bytes.clone())
                    .map_err(|_| Error::malformed(format!("{path} is not valid UTF-8")))?;
                // Apply ALL operations to the in-memory copy; a failing op N
                // aborts with nothing written (atomicity).
                let mut edited = original.clone();
                for (i, op) in ops.iter().enumerate() {
                    apply_edit_op(&mut edited, op, i)?;
                }
                let applied: Vec<String> = ops.iter().map(|o| o.name().to_string()).collect();
                if edited == original {
                    return Ok(ToolOutcome {
                        text: format!(
                            "{path} unchanged ({} operations were no-ops)",
                            ops.len()
                        ),
                        exit_code: Some(0),
                        postcondition: Some(postcondition(current.hash)),
                        ..Default::default()
                    });
                }
                // Checkpoint the ORIGINAL content into the CAS (deduped)
                // BEFORE the write, exactly like write_file.
                let before = snapshots.before_write(ctx.session_id, path, &current.bytes)?;
                // One atomic write through the engine: expected-hash
                // validation against the CURRENT read + parse-before-accept
                // (a parse-breaking edit rolls back with a loud error).
                let req = EditRequest {
                    path: path.to_string(),
                    expected_hash: current.hash,
                    ops: vec![EditOp::Range {
                        start: 0,
                        end: original.len(),
                        replacement: edited.clone(),
                    }],
                };
                let outcome = edit.apply(&ws, &ctx.identity, &req, RepairMode::Rollback)?;
                let sequence = snapshots.checkpoints(ctx.session_id)?.len() as i64 + 1;
                snapshots.after_write(
                    ctx.session_id,
                    path,
                    before,
                    outcome.new_hash,
                    sequence,
                    edited.as_bytes(),
                )?;
                Ok(ToolOutcome {
                    text: format!(
                        "applied {} of {} operations to {path}: {}. new expected_hash: {}",
                        outcome.ops_applied,
                        ops.len(),
                        applied.join(", "),
                        outcome.new_hash.to_hex()
                    ),
                    exit_code: Some(0),
                    postcondition: Some(postcondition(outcome.new_hash)),
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

/// Marker appended to a run_command excerpt when the durable artifact was
/// truncated at its cap (audit round 10) — truncation is never silent.
const ARTIFACT_TRUNCATED_MARK: &str = "\n[artifact truncated at its cap]";

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
                // Audit round 10: artifact truncation is EXPLICIT — when
                // the output overflowed artifact_max, the excerpt carries a
                // marker so no caller mistakes a capped artifact for the
                // full stream.
                let mut text = out.excerpt;
                if out.artifact_truncated && !text.contains(ARTIFACT_TRUNCATED_MARK) {
                    text.push_str(ARTIFACT_TRUNCATED_MARK);
                }
                Ok(ToolOutcome {
                    text,
                    exit_code: out.exit_code,
                    artifact: out.artifact,
                    slice_hint: out.slice_hint,
                    // A shell command's external effects are never known:
                    // mark unknown so crash recovery forces verification
                    // (commandment 6).
                    effect_status: EffectStatus::Unknown,
                    postcondition: None,
                })
            })
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_agent::{ToolArtifactSink, ToolCallMode};
    use faktor_core::cancellation::CancellationToken;
    use faktor_core::hash::FileHash;
    use faktor_core::id::{OpId, SessionId, TaskId, WorkspaceId, WorktreeId};
    use faktor_core::WorkspaceIdentity;
    use faktor_sandbox::{Rule, SandboxPolicy};
    use faktor_session::SessionManager;
    use faktor_terminal::ProcessSupervisor;
    use std::path::PathBuf;

    struct ToolFixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        session: SessionId,
        identity: WorkspaceIdentity,
        sandbox: Arc<PermissionEngine>,
        snapshots: Arc<faktor_snapshot::CheckpointStore>,
        cas: Arc<faktor_cas::Cas>,
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
        let fs_service = faktor_fs::WorkspaceFileService::new();
        let _opened = fs_service.open(ws_id, root.clone()).unwrap();
        let identity = WorkspaceIdentity::new(ws_id, WorktreeId::new(1), TaskId::new(1));
        let cas = manager.cas();
        ToolFixture {
            _dir: dir,
            root: root.clone(),
            session: row.id(),
            identity,
            sandbox: Arc::new(PermissionEngine::new(policy, Some(root))),
            snapshots: Arc::new(faktor_snapshot::CheckpointStore::new(
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
        let fs_service = faktor_fs::WorkspaceFileService::new();
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
            edit: Some(Arc::new(faktor_edit::EditEngine::new(fs_service.clone()))),
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

    // ---------------------------------------------------------- edit_file

    fn edit_args(path: &str, ops: serde_json::Value) -> serde_json::Value {
        serde_json::json!({ "path": path, "operations": ops })
    }

    #[tokio::test]
    async fn edit_file_exact_replace_roundtrip() {
        let f = fixture(SandboxPolicy::default());
        std::fs::write(f.root.join("a.txt"), "hello world hello").unwrap();
        let tool = edit_file_tool();
        let out = (tool.execute)(
            ctx(&f),
            edit_args(
                "a.txt",
                serde_json::json!([
                    {"type": "replace_exact", "search": "hello", "replace": "goodbye"}
                ]),
            ),
        )
        .await
        .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(
            out.text.contains("new expected_hash"),
            "outcome must report the follow-up hash: {}",
            out.text
        );
        // replace_exact targets the FIRST literal occurrence only.
        assert_eq!(
            std::fs::read(f.root.join("a.txt")).unwrap(),
            b"goodbye world hello"
        );
        let pc = out.postcondition.unwrap();
        assert_eq!(pc.expected_hash, f.cas.put(b"goodbye world hello").unwrap());
        assert_eq!(f.snapshots.checkpoints(f.session).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn edit_file_unique_search_replace_two_matches_refuses_no_write() {
        let f = fixture(SandboxPolicy::default());
        std::fs::write(f.root.join("a.txt"), "foo foo").unwrap();
        let tool = edit_file_tool();
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "a.txt",
                serde_json::json!([
                    {"type": "search_replace", "search": "foo", "replace": "bar"}
                ]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict, "{err}");
        assert!(err.message.contains("ambiguous"), "{err}");
        assert!(
            err.message.contains("op 1"),
            "the error names the operation: {err}"
        );
        assert_eq!(
            std::fs::read(f.root.join("a.txt")).unwrap(),
            b"foo foo",
            "an ambiguous edit must not write"
        );
        // unique: false explicitly replaces the first occurrence.
        let out = (tool.execute)(
            ctx(&f),
            edit_args(
                "a.txt",
                serde_json::json!([
                    {"type": "search_replace", "search": "foo", "replace": "bar", "unique": false}
                ]),
            ),
        )
        .await
        .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert_eq!(std::fs::read(f.root.join("a.txt")).unwrap(), b"bar foo");
    }

    #[tokio::test]
    async fn edit_file_insert_before_and_after_unique_anchor() {
        let f = fixture(SandboxPolicy::default());
        std::fs::write(f.root.join("a.txt"), "a\nb\nc\n").unwrap();
        let tool = edit_file_tool();
        (tool.execute)(
            ctx(&f),
            edit_args(
                "a.txt",
                serde_json::json!([
                    {"type": "insert", "anchor": "b", "position": "before", "text": "B0\n"},
                    {"type": "insert", "anchor": "c\n", "position": "after", "text": "C1\n"}
                ]),
            ),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read(f.root.join("a.txt")).unwrap(),
            b"a\nB0\nb\nc\nC1\n"
        );
        // Ambiguous anchor refuses with the op named; nothing written.
        std::fs::write(f.root.join("a.txt"), "x x\n").unwrap();
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "a.txt",
                serde_json::json!([{"type": "insert", "anchor": "x", "position": "after", "text": "y"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict, "{err}");
        assert!(err.message.contains("op 1 (insert)"), "{err}");
        assert_eq!(std::fs::read(f.root.join("a.txt")).unwrap(), b"x x\n");
        // Missing anchor is malformed.
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "a.txt",
                serde_json::json!([{"type": "insert", "anchor": "zzz", "position": "before", "text": "y"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
        assert!(err.message.contains("op 1 (insert)"), "{err}");
        // Bad position is malformed before any fs access.
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "a.txt",
                serde_json::json!([{"type": "insert", "anchor": "x", "position": "middle", "text": "y"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
    }

    fn numbered_file(n: usize) -> String {
        (1..=n).map(|i| format!("line {i}\n")).collect()
    }

    #[tokio::test]
    async fn edit_file_region_replace_lines_20_900_is_bounded_and_correct() {
        let f = fixture(SandboxPolicy::default());
        std::fs::write(f.root.join("big.txt"), numbered_file(1000)).unwrap();
        let tool = edit_file_tool();
        let out = (tool.execute)(
            ctx(&f),
            edit_args(
                "big.txt",
                serde_json::json!([
                    {"type": "region_replace", "start_line": 20, "end_line": 900, "text": "REPLACED\nmid\n"}
                ]),
            ),
        )
        .await
        .unwrap();
        assert_eq!(out.exit_code, Some(0));
        let text = String::from_utf8(std::fs::read(f.root.join("big.txt")).unwrap()).unwrap();
        let out_lines: Vec<&str> = text.lines().collect();
        assert_eq!(out_lines.len(), 121, "19 + 2 inserted + 100 tail");
        assert_eq!(out_lines[0], "line 1");
        assert_eq!(out_lines[18], "line 19");
        assert_eq!(out_lines[19], "REPLACED");
        assert_eq!(out_lines[20], "mid");
        assert_eq!(out_lines[21], "line 901");
        assert_eq!(out_lines[120], "line 1000");
        // The follow-up hash in the outcome matches the actual bytes.
        let hash = out.text.split("new expected_hash: ").nth(1).unwrap().trim();
        assert_eq!(
            FileHash::from_hex(hash).unwrap(),
            f.cas.put(text.as_bytes()).unwrap()
        );
    }

    #[tokio::test]
    async fn edit_file_region_line_semantics_and_guards() {
        let f = fixture(SandboxPolicy::default());
        let tool = edit_file_tool();
        // Lines keep their terminators; text "X\nY\n" is two lines.
        std::fs::write(f.root.join("r.txt"), "a\nb\nc\n").unwrap();
        (tool.execute)(
            ctx(&f),
            edit_args(
                "r.txt",
                serde_json::json!([{"type": "region_replace", "start_line": 2, "end_line": 2, "text": "X\nY\n"}]),
            ),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read(f.root.join("r.txt")).unwrap(),
            b"a\nX\nY\nc\n"
        );
        // A file WITHOUT a trailing newline keeps its last line unterminated.
        std::fs::write(f.root.join("r.txt"), "a\nb\nc").unwrap();
        (tool.execute)(
            ctx(&f),
            edit_args(
                "r.txt",
                serde_json::json!([{"type": "region_replace", "start_line": 2, "end_line": 2, "text": "X"}]),
            ),
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read(f.root.join("r.txt")).unwrap(), b"a\nX\nc");
        // Whole-file coverage is refused (write_file's job).
        std::fs::write(f.root.join("r.txt"), "one\ntwo\n").unwrap();
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "r.txt",
                serde_json::json!([{"type": "region_replace", "start_line": 1, "end_line": 2, "text": "x"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
        assert!(err.message.contains("write_file"), "{err}");
        assert_eq!(std::fs::read(f.root.join("r.txt")).unwrap(), b"one\ntwo\n");
        // Out-of-range lines name the op and refuse.
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "r.txt",
                serde_json::json!([{"type": "region_replace", "start_line": 5, "end_line": 9, "text": "x"}]),
            ),
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("op 1 (region_replace)"), "{err}");
        assert_eq!(err.kind, ErrorKind::Malformed);
        // start > end refuses.
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "r.txt",
                serde_json::json!([{"type": "region_replace", "start_line": 2, "end_line": 1, "text": "x"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
    }

    #[tokio::test]
    async fn edit_file_expected_hash_staleness_refuses_without_write() {
        let f = fixture(SandboxPolicy::default());
        let original = b"original content";
        std::fs::write(f.root.join("s.txt"), original).unwrap();
        let tool = edit_file_tool();
        // The model read the ORIGINAL content; an external writer lands.
        let stale = f.cas.put(original).unwrap();
        std::fs::write(f.root.join("s.txt"), b"external edit").unwrap();
        let err = (tool.execute)(
            ctx(&f),
            serde_json::json!({
                "path": "s.txt",
                "expected_hash": stale.to_hex(),
                "operations": [
                    {"type": "replace_exact", "search": "external", "replace": "agent"}
                ]
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict, "{err}");
        assert!(err.message.contains("changed since"), "{err}");
        assert_eq!(
            std::fs::read(f.root.join("s.txt")).unwrap(),
            b"external edit",
            "a stale edit must not write"
        );
        assert_eq!(f.snapshots.checkpoints(f.session).unwrap().len(), 0);
        // A MALFORMED expected_hash hex is refused up front.
        let err = (tool.execute)(
            ctx(&f),
            serde_json::json!({
                "path": "s.txt",
                "expected_hash": "not-hex",
                "operations": [{"type": "replace_exact", "search": "x", "replace": "y"}]
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
    }

    #[tokio::test]
    async fn edit_file_multi_op_failure_is_loud_and_atomic() {
        let f = fixture(SandboxPolicy::default());
        let original = b"alpha beta gamma";
        std::fs::write(f.root.join("m.txt"), original).unwrap();
        let tool = edit_file_tool();
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "m.txt",
                serde_json::json!([
                    {"type": "replace_exact", "search": "alpha", "replace": "ALPHA"},
                    {"type": "insert", "anchor": "beta", "position": "after", "text": " "},
                    {"type": "replace_exact", "search": "zzz-missing", "replace": "x"}
                ]),
            ),
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("op 3"), "loud error names op 3: {err}");
        assert_eq!(
            std::fs::read(f.root.join("m.txt")).unwrap(),
            original,
            "a later failing op must leave the file UNCHANGED (no half-applied write)"
        );
        assert_eq!(f.snapshots.checkpoints(f.session).unwrap().len(), 0);
    }

    #[tokio::test]
    async fn edit_file_hostile_inputs_refuse_cleanly() {
        let f = fixture(SandboxPolicy::default());
        std::fs::write(f.root.join("h.txt"), "content").unwrap();
        let tool = edit_file_tool();
        // 9 operations exceed the cap.
        let nine: Vec<serde_json::Value> = (0..9)
            .map(|i| {
                serde_json::json!({"type": "replace_exact", "search": format!("c{i}"), "replace": "x"})
            })
            .collect();
        let err = (tool.execute)(ctx(&f), edit_args("h.txt", serde_json::json!(nine)))
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized, "{err}");
        assert!(err.message.contains("8"), "{err}");
        // Oversized per-op payload.
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "h.txt",
                serde_json::json!([{"type": "replace_exact", "search": "c", "replace": "x".repeat(9 * 1024)}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized, "{err}");
        assert!(err.message.contains("op 1"), "{err}");
        // Missing path / missing operations / empty operations.
        let err = (tool.execute)(ctx(&f), serde_json::json!({"operations": []}))
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
        let err = (tool.execute)(ctx(&f), serde_json::json!({"path": "h.txt"}))
            .await
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
        let err = (tool.execute)(
            ctx(&f),
            serde_json::json!({"path": "h.txt", "operations": []}),
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("at least one"), "{err}");
        // Unknown op type.
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "h.txt",
                serde_json::json!([{"type": "rewrite_whole_file", "search": "c", "replace": "x"}]),
            ),
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("unknown operation type"), "{err}");
        assert!(err.message.contains("op 1"), "{err}");
        // Non-UTF8 file errors clearly.
        std::fs::write(f.root.join("bin.dat"), vec![0xFF, 0xFE, 0x00]).unwrap();
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "bin.dat",
                serde_json::json!([{"type": "replace_exact", "search": "a", "replace": "b"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{err}");
        assert!(err.message.contains("UTF-8"), "{err}");
        // Missing file tells the model to use write_file.
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "missing.txt",
                serde_json::json!([{"type": "replace_exact", "search": "a", "replace": "b"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::NotFound, "{err}");
        assert!(err.message.contains("write_file"), "{err}");
        // The hostile attempts never touched the healthy file.
        assert_eq!(std::fs::read(f.root.join("h.txt")).unwrap(), b"content");
    }

    #[tokio::test]
    async fn edit_file_followup_with_returned_hash_works_then_stale_refuses() {
        let f = fixture(SandboxPolicy::default());
        std::fs::write(f.root.join("c.txt"), "step zero").unwrap();
        let tool = edit_file_tool();
        let out = (tool.execute)(
            ctx(&f),
            edit_args(
                "c.txt",
                serde_json::json!([{"type": "replace_exact", "search": "zero", "replace": "one"}]),
            ),
        )
        .await
        .unwrap();
        let h1 = out
            .text
            .split("new expected_hash: ")
            .nth(1)
            .unwrap()
            .trim()
            .to_string();
        assert_eq!(f.cas.put(b"step one").unwrap().to_hex(), h1);
        // Follow-up edit carries the returned hash → succeeds.
        let out = (tool.execute)(
            ctx(&f),
            serde_json::json!({
                "path": "c.txt",
                "expected_hash": h1,
                "operations": [{"type": "replace_exact", "search": "one", "replace": "two"}]
            }),
        )
        .await
        .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert_eq!(std::fs::read(f.root.join("c.txt")).unwrap(), b"step two");
        // The SAME hash again is now stale → refused, file untouched.
        let err = (tool.execute)(
            ctx(&f),
            serde_json::json!({
                "path": "c.txt",
                "expected_hash": h1,
                "operations": [{"type": "replace_exact", "search": "two", "replace": "three"}]
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict, "{err}");
        assert_eq!(std::fs::read(f.root.join("c.txt")).unwrap(), b"step two");
    }

    #[tokio::test]
    async fn edit_file_sandbox_deny_and_symlink_escape_refuse() {
        let f = fixture(SandboxPolicy {
            write_workspace: Rule::Deny,
            ..Default::default()
        });
        std::fs::write(f.root.join("d.txt"), "x").unwrap();
        let tool = edit_file_tool();
        let err = (tool.execute)(
            ctx(&f),
            edit_args(
                "d.txt",
                serde_json::json!([{"type": "replace_exact", "search": "x", "replace": "y"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission, "{err}");
        assert_eq!(std::fs::read(f.root.join("d.txt")).unwrap(), b"x");
        // Symlink escape is refused by the workspace resolution.
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "s").unwrap();
        std::os::unix::fs::symlink(outside.path(), f.root.join("link")).unwrap();
        let f2 = fixture(SandboxPolicy::default());
        std::os::unix::fs::symlink(outside.path(), f2.root.join("link")).unwrap();
        let err = (tool.execute)(
            ctx(&f2),
            edit_args(
                "link/secret.txt",
                serde_json::json!([{"type": "replace_exact", "search": "s", "replace": "pwned"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission, "{err}");
        assert_eq!(
            std::fs::read(outside.path().join("secret.txt")).unwrap(),
            b"s"
        );
    }

    #[tokio::test]
    async fn edit_file_no_workspace_and_malicious_args_never_panic() {
        let f = fixture(SandboxPolicy::default());
        let tool = edit_file_tool();
        let err = (tool.execute)(
            bare_ctx(),
            edit_args(
                "a.txt",
                serde_json::json!([{"type": "replace_exact", "search": "a", "replace": "b"}]),
            ),
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
        for args in [
            serde_json::json!({}),
            serde_json::json!({"path": 42, "operations": []}),
            serde_json::json!({"path": ["a"], "operations": []}),
            serde_json::json!({"path": "x", "operations": "nope"}),
            serde_json::json!({"path": "x", "operations": [{}]}),
            serde_json::json!({"path": "x", "operations": [{"type": 7}]}),
            serde_json::json!({"path": "x", "operations": [{"type": "insert", "anchor": 3, "position": "before", "text": 4}]}),
            serde_json::json!({"path": "x", "expected_hash": 7, "operations": []}),
            serde_json::json!({"path": "x", "operations": [{"type": "region_replace", "start_line": 1.5, "end_line": "a", "text": null}]}),
        ] {
            let _ = (tool.clone().execute)(ctx(&f), args).await;
        }
        // A successful edit whose result equals the input is a no-op:
        // exit 0, unchanged-hash outcome, and NO checkpoint row.
        std::fs::write(f.root.join("noop.txt"), "same text").unwrap();
        let out = (tool.execute)(
            ctx(&f),
            edit_args(
                "noop.txt",
                serde_json::json!([{"type": "replace_exact", "search": "same", "replace": "same"}]),
            ),
        )
        .await
        .unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.text.contains("unchanged"), "{}", out.text);
        assert_eq!(
            out.postcondition.unwrap().expected_hash,
            f.cas.put(b"same text").unwrap()
        );
        assert_eq!(f.snapshots.checkpoints(f.session).unwrap().len(), 0);
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

    #[tokio::test]
    async fn write_file_reports_postcondition_of_raw_bytes() {
        // P0 (requirement 3): the real write_file tool computes its
        // FilePostcondition — workspace id + worktree id, the RELATIVE path
        // as written, and BLAKE3 of the ACTUAL bytes written (never a hash
        // of the JSON-encoded content argument, never a daemon-cwd path).
        let f = fixture(SandboxPolicy::default());
        let tool = write_file_tool();
        let out = (tool.execute)(
            ctx(&f),
            serde_json::json!({"path": "a.txt", "content": "hello"}),
        )
        .await
        .unwrap();
        let pc = out
            .postcondition
            .expect("write_file reports a postcondition");
        assert_eq!(pc.workspace_id, f.identity.workspace_id);
        assert_eq!(pc.worktree_id, f.identity.worktree_id);
        assert_eq!(pc.relative_path, "a.txt", "the RELATIVE path as written");
        // BLAKE3 of the raw bytes ("hello") — NOT of the JSON encoding
        // ("\"hello\"", which the old generic runtime hashed).
        assert_eq!(pc.expected_hash, f.cas.put(b"hello").unwrap());
        assert_ne!(pc.expected_hash, f.cas.put(b"\"hello\"").unwrap());
        assert_eq!(std::fs::read(f.root.join("a.txt")).unwrap(), b"hello");
        // An unchanged rewrite still reports the same postcondition (the
        // file state matches the would-be write).
        let out = (tool.execute)(
            ctx(&f),
            serde_json::json!({"path": "a.txt", "content": "hello"}),
        )
        .await
        .unwrap();
        assert_eq!(out.postcondition.unwrap().expected_hash, pc.expected_hash);
        // A new empty file reports blake3("") (the empty bytes as written).
        let out = (tool.execute)(
            ctx(&f),
            serde_json::json!({"path": "empty.txt", "content": ""}),
        )
        .await
        .unwrap();
        let pc = out.postcondition.unwrap();
        assert_eq!(pc.expected_hash, f.cas.put(b"").unwrap());
        assert_eq!(pc.relative_path, "empty.txt");
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
        // Artifact cap semantics: the fixture's artifact_max (1 MiB) bounds
        // the spool; the artifact holds the cap's first bytes and the
        // excerpt carries an explicit truncation marker.
        assert_eq!(
            blob.len(),
            1024 * 1024,
            "the spill must respect artifact_max, got {} bytes",
            blob.len()
        );
        assert!(
            out.text.contains("artifact truncated at its cap"),
            "truncation must be explicit in the outcome: {}",
            out.text
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
