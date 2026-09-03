# faktor-mcp + faktor-lsp + faktor-sandbox specs (spec §30, §31, §32)

## Part A — faktor-sandbox (first: no deps beyond faktor-core)

Crate: crates/sandbox (faktor-core, serde, serde_json, tracing; tempfile dev).

```rust
pub struct SandboxPolicy { pub read_workspace: Rule, pub write_workspace: Rule,
    pub read_external: Rule, pub write_external: Rule, pub execute_shell: Rule,
    pub network: NetworkPolicy, pub mcp: Rule, pub git: Rule }
pub enum Rule { Allow, Deny, Ask }
impl Default for SandboxPolicy { /* workspace rw Allow, external Ask, shell Ask, network AllowProviders, mcp Allow, git Allow */ }
pub struct PermissionEngine { policy: SandboxPolicy, workspace_root: Option<PathBuf> }
impl PermissionEngine {
    pub fn new(policy: SandboxPolicy, workspace_root: Option<PathBuf>) -> Self;
    /// Evaluate a Capability → decision; path capabilities are checked
    /// against the workspace root with canonicalization (symlink-safe).
    pub fn evaluate(&self, capability: &Capability) -> PermissionDecision;
    pub fn is_within_workspace(&self, path: &Path) -> bool;
    pub fn policy(&self) -> &SandboxPolicy;
}
```
Rules: `is_within_workspace` canonicalizes the parent dir + joins the file name, rejects symlink escapes and `..`. External reads/writes evaluate against policy with the real path. Network destinations use NetworkPolicy::allows.
Adversarial tests: traversal matrix (.., abs, symlink, symlinked dir, unicode tricks), policy matrix (Allow/Deny/Ask per capability), network matrix (DenyAll/AllowProviders/AllowConfigured incl. subdomain escape), evaluate_never_panics_on_any_path, symlink_loop (a→b→a) terminates.

## Part B — faktor-mcp: JSON-RPC MCP client

Crate: crates/mcp (faktor-core, faktor-terminal, serde_json, tokio, reqwest).

```rust
pub struct McpConfig { pub name: String, pub command: String, pub args: Vec<String>, pub env: Vec<(String,String)> }
pub struct McpServer { /* JSON-RPC over stdio via ProcessSupervisor */ }
impl McpServer {
    pub async fn connect(cfg: McpConfig, supervisor: Arc<ProcessSupervisor>) -> Result<Arc<McpServer>>;
    pub async fn list_tools(&self) -> Result<Vec<McpTool>>;
    pub async fn call_tool(&self, name: &str, args: Value, deadline: Duration) -> Result<McpResult>;
    pub async fn close(&self) -> Result<()>;
    pub fn name(&self) -> &str;
    pub fn is_alive(&self) -> bool;
}
pub struct McpTool { pub name: String, pub description: String, pub input_schema: Value }
pub struct McpResult { pub content: Vec<Value>, pub is_error: bool }
```
Rules: Content-Length framed JSON-RPC 2.0; initialize handshake; every call has a deadline; bad servers (garbage output, hangs, crashes) never destabilize the caller: garbage → Malformed error, hang → timeout + kill, crash → NotFound with restart-ability. Bounded: each call's response ≤ 16MB.
Adversarial tests (write tiny fake MCP servers in the test as `sh -c` scripts or a compiled helper... simplest: use `python3 -c` if available, else `sh` with printf — prefer a tiny Rust helper binary? NO: use `sh -c 'while read l; do ...'` is painful; instead test with a mock server implemented as a thread that spawns a child process speaking JSON-RPC over a pipe? Simpler: implement the JSON-RPC framing parser tests directly (pure), and integration tests spawn `python3` if present (skip gracefully when absent)):
1. framing_roundtrip_content_length (pure parser)
2. garbage_output_is_malformed_not_hang
3. hang_killed_by_deadline (server sleeps forever; timeout error; process reaped)
4. crash_after_connect_reports_not_found
5. oversized_response_rejected (16MB cap)
6. malformed_jsonrpc_rejected (wrong version, missing id, non-object)
7. call_with_unknown_tool_errors
8. close_terminates_process (no leftover; ps check)
9. concurrent_calls_multiplexed (2 tools × 5 calls parallel, ids match responses)
10. notification_ignored (server sends unsolicited notification → ignored)

## Part C — faktor-lsp: workspace-scoped language server

Crate: crates/lsp (faktor-core, faktor-terminal, serde_json, tokio).

```rust
pub struct LspConfig { pub name: String, pub command: String, pub args: Vec<String> }
pub struct LspManager { /* workspace → server map; idle unload */ }
impl LspManager {
    pub fn new(supervisor: Arc<ProcessSupervisor>) -> Self;
    pub async fn start(&self, workspace_id: WorkspaceId, cfg: LspConfig) -> Result<Arc<LspClient>>;
    pub async fn client(&self, workspace_id: WorkspaceId) -> Result<Arc<LspClient>>; // shared between sessions (spec §32)
    pub async fn shutdown(&self, workspace_id: WorkspaceId) -> Result<()>;  // heavy servers unload on idle
    pub fn active(&self) -> Vec<WorkspaceId>;
}
pub struct LspClient { /* JSON-RPC over stdio: initialize, didOpen, documentSymbol, shutdown/exit */ }
impl LspClient {
    pub async fn initialize(&self) -> Result<Value>;
    pub async fn did_open(&self, uri: &str, text: &str);
    pub async fn document_symbols(&self, uri: &str) -> Result<Vec<Value>>;
    pub async fn shutdown(&self) -> Result<()>;
}
```
Rules: multiplexed requests with per-request ids + response correlation; initialize handshake with shutdown on idle; bad servers handled like MCP. Integration tests use `typescript-language-server`/`rust-analyzer` ONLY if present — otherwise `#[ignore]`d; the framing/multiplexing tests are pure. Keep integration optional via `which`.

Build/test each crate green, zero warnings. Do NOT modify other crates. Do NOT commit.
