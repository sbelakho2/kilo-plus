# Faktor inter-crate API contracts

This file is the frozen interface contract between crates. Sub-agents
implement crates against these signatures. Do not change foundation APIs
without updating this file and every dependant.

## faktor-core (already implemented, `crates/core`)

```rust
pub struct SessionId(u64); pub struct WorkspaceId(u64); pub struct WorktreeId(u64);
pub struct TaskId(u64); pub struct OpId(u64); pub struct ProviderCallId(u64); pub struct EventSeq(u64);
// all: ::new(raw) (panics on 0), ::raw() -> u64, Display, serde, PartialEq/Ord/Hash/Copy

pub enum ErrorKind { NotFound, Conflict, InvalidState{from,to}, Permission, Timeout, Cancelled,
    Store, Network, Provider{code,retryable}, Malformed, Oversized, RateLimited, Deadlock, Internal }
pub struct Error { pub kind: ErrorKind, pub message: String, pub retryable: bool }
// Error::new/not_found/conflict/permission/timeout/cancelled/malformed/oversized/internal
pub type Result<T> = std::result::Result<T, Error>;

pub enum AgentState { Idle, Preparing, BuildingContext, WaitingForModel, Streaming, ToolRequested,
    WaitingForPermission, ExecutingTool, Validating, UpdatingMemory, ReadyForNextTurn,
    Completed, Cancelled, FailedRecoverable, FailedPermanent, NeedsUserInput, Suspended }
// AgentState::allowed_transitions()->&'static[AgentState], is_terminal(), is_active(), label()
pub struct StateMachine(AgentState); // ::new(s), state(), transition(to)->Result<()>, force(to)

pub enum EventKind { SessionCreated, PromptReceived, ContextPrepared, ModelStarted, ModelChunkReceived,
    ToolRequested, ToolStarted, FileChanged, ToolCompleted, ToolCancelled, CheckpointCreated,
    ContextCompacted, CompactRejected, SubagentStarted, SubagentCompleted, TurnCompleted,
    PermissionGranted, PermissionDenied, CrashDetected, RecoveryApplied, SessionEnded,
    Suspended, Resumed, Failed }
pub struct Event { pub seq: EventSeq, pub session_id: SessionId, pub op_id: Option<OpId>,
    pub kind: EventKind, pub state: AgentState, pub ts_ms: i64, pub payload: Option<serde_json::Value> }

pub enum OpState { Pending, Running, Done, Failed, Cancelled }
pub enum RecoveryStrategy { VerifyHash{path:String,expected:FileHash}, MarkUnknown, Idempotent, Manual, None }
pub enum EffectStatus { Unknown, Verified, Applied, Failed }
pub struct OpMeta { pub operation_id: OpId, pub session_id: SessionId, pub state: OpState,
    pub start_time_ms: i64, pub deadline: Deadline, pub retry_policy: RetryPolicy,
    pub cancellation: CancellationToken, pub recovery: RecoveryStrategy }
// OpMeta::new(op, session, deadline, retry, token, recovery, now_ms), ensure_alive(now_ms)->Result<()>

pub struct CancellationToken; // ::new(), is_cancelled(), cancel()->bool, child(), attach(other), wait()

pub trait Clock: Send+Sync { fn now_ms(&self) -> i64; }
pub struct SystemClock; pub struct TestClock; // ::new(ms), advance(ms), set(ms)
pub struct Deadline; // ::at(ms), now_plus(&clock, ms), is_expired(now_ms), at_ms()

pub struct RetryPolicy { pub max_attempts: u32, pub base_delay_ms: u64, pub max_delay_ms: u64,
    pub jitter: f64, pub class: RetryClass }
pub enum RetryClass { Network, RateLimited, ServerError, Always }
// should_retry(attempt, retryable, rate_limited)->bool, next_delay(attempt)->Duration

pub struct ModelCapabilities { pub context: usize, pub max_output: usize, pub tools: bool,
    pub parallel_tools: bool, pub thinking: bool, pub vision: bool, pub json_schema: bool,
    pub streaming: bool, pub embeddings: bool, pub reasoning: bool }
// Default = conservative; small_local() = 32K Ollama profile; supports_tools(), supports_parallel_tools()
pub enum ReasoningMode { Off, Low, Medium, High }

pub enum ResourceClass { Model, DiskRead, DiskWrite, Cpu, Git, Network, Terminal, Mcp, Indexing }
pub struct ResourceLimits { pub limits: HashMap<ResourceClass, usize> } // Default = budgets

pub enum Capability { ReadWorkspace{path}, WriteWorkspace{path}, ReadExternal{path},
    WriteExternal{path}, ExecuteShell{command}, Network{destination}, Mcp{server}, Git{operation} }
pub enum PermissionDecision { Allow, Deny, Ask }
pub enum NetworkPolicy { DenyAll, AllowProviders{endpoints}, AllowConfigured{endpoints,domains} }

pub struct FileHash([u8;32]); // ::from([u8;32]), from_hex(&str)->Option<Self>, to_hex(), cas_path(), bytes()
pub struct WorkspaceIdentity { pub workspace_id: WorkspaceId, pub worktree_id: WorktreeId, pub task_id: TaskId }
pub const VERSION: &str; pub const PROTOCOL_V756: &str; pub const UX_BASELINE: &str;
```

## faktor-cas (already implemented)

```rust
pub struct Cas; // ::open(PathBuf)->CasResult<Cas>, root at root/
// put(&[u8])->CasResult<FileHash>, put_bounded(&[u8],max)->CasResult<FileHash>,
// has(hash)->bool, get(hash)->CasResult<Vec<u8>>, stored_size(hash), copy_to(hash,&mut W),
// verify_integrity()->Vec<FileHash>, blob_count()->usize
pub enum CasError { Io, NotFound(FileHash), SizeMismatch{..}, HashMismatch(FileHash), Zstd(String) }
```

## faktor-store (already implemented, `crates/store`)

```rust
pub struct Store; // ::open(root: impl Into<PathBuf>, integrity_check: bool)->StoreResult<Store>
// rows: SessionRow { id, workspace_id, title, provider, model, state, created_ms, updated_ms }
//       MessageRow { id, session_id, seq, role, data: Value, created_ms }
//       PartRow { id, message_id, kind, data: Value, created_ms }
//       ToolRunRow { id, session_id, op_id, tool, args: Value, status, started_ms, ended_ms,
//                    effect_status, recovery: Value, expected_hash: Option<String> }
//       CheckpointRow { id, session_id, sequence, path, before_hash, after_hash,
//                    after_cas_hash: Option<String>, created_ms, restored_ms }
//       WorktreeRow { id, workspace_id, path, branch, active }
pub fn create_workspace(&self, root: &str) -> StoreResult<WorkspaceId>
pub fn workspace_root(&self, id: WorkspaceId) -> StoreResult<Option<String>>
pub fn create_session(&self, ws: WorkspaceId, title: &str, provider: &str, model: &str) -> StoreResult<SessionRow>
pub fn get_session(&self, id) -> StoreResult<Option<SessionRow>>
pub fn list_sessions(&self, ws: Option<WorkspaceId>) -> StoreResult<Vec<SessionRow>>
pub fn set_session_state(&self, id, state: AgentState) -> StoreResult<()>
pub fn append_event(&self, session, op_id: Option<OpId>, kind: EventKind, state: AgentState, ts_ms: i64, payload: Option<Value>) -> StoreResult<EventSeq>
pub fn events_after(&self, session, after: EventSeq) -> StoreResult<Vec<Event>>
pub fn events_range(&self, session, from_seq: u64, limit: Option<u64>) -> StoreResult<Vec<Event>>
pub fn last_event_seq(&self, session) -> StoreResult<Option<EventSeq>>
pub fn messages_before(&self, session, before_seq: Option<i64>, limit: u64) -> StoreResult<Vec<MessageRow>> // newest first
pub fn put_message(&self, session, seq: i64, role: &str, data: Value) -> StoreResult<i64>
pub fn put_part(&self, message_id: i64, kind: &str, data: Value) -> StoreResult<i64>
pub fn parts_of(&self, message_id: i64) -> StoreResult<Vec<PartRow>>
pub fn message_count(&self, session) -> StoreResult<i64>
pub fn message_created_ms(&self, session, seq: i64) -> StoreResult<Option<i64>>
pub fn get_task_ledger(&self, session) -> StoreResult<Option<Value>>
pub fn put_task_ledger(&self, session, ledger: Value) -> StoreResult<()>
pub fn start_tool_run(&self, session, op: OpId, tool: &str, args: Value, recovery: Value, expected_hash: Option<String>) -> StoreResult<i64>
pub fn finish_tool_run(&self, session, op: OpId, status: &str, effect_status: &str) -> StoreResult<()>
pub fn set_tool_run_effect(&self, session, op: OpId, effect_status: &str) -> StoreResult<()>
pub fn pending_tool_runs(&self, session) -> StoreResult<Vec<ToolRunRow>>
pub fn record_provider_call(&self, session, op: OpId, provider: &str, model: &str, status: &str, tokens_in: Option<u64>, tokens_out: Option<u64>, error: Option<&str>) -> StoreResult<i64>
pub fn put_checkpoint(&self, session, sequence: i64, path: &str, before_hash: &str, after_hash: &str, after_cas_hash: Option<&str>) -> StoreResult<i64>
pub fn checkpoints_of(&self, session) -> StoreResult<Vec<CheckpointRow>>
pub fn mark_checkpoint_restored(&self, id: i64) -> StoreResult<()>
pub fn put_artifact(&self, session, kind: &str, cas_hash: &str, summary: &str, size: i64) -> StoreResult<i64>
pub fn artifact(&self, cas_hash: &str) -> StoreResult<Option<(String, String)>> // (summary, kind)
pub fn put_worktree(&self, ws: WorkspaceId, path: &str, branch: &str) -> StoreResult<i64>
pub fn worktrees_of(&self, ws) -> StoreResult<Vec<WorktreeRow>>
pub fn remove_worktree(&self, path: &str) -> StoreResult<()>
pub fn upsert_memory_fact(&self, session, kind: &str, key: &str, value: &str) -> StoreResult<()>
pub fn memory_facts(&self, session) -> StoreResult<Vec<(String,String,String)>>
pub fn record_compaction(&self, session, before: i64, after: i64, target: i64, accepted: bool, strategy: &str) -> StoreResult<()>
pub fn insert_permission(&self, session, op: OpId, capability: &str) -> StoreResult<i64>
pub fn resolve_permission(&self, id: i64, decision: &str) -> StoreResult<()>
pub fn pending_permission(&self, id: i64) -> StoreResult<Option<(SessionId, OpId, String)>>
pub fn integrity_check(&self) -> StoreResult<Vec<String>>
pub fn backup_to(&self, dest: &Path) -> StoreResult<()>
pub fn diagnostics(&self) -> StoreResult<Value>
```

## faktor-snapshot (`crates/snapshot`)

```rust
pub enum RollbackOutcome { Restored { path, hash }, Conflict { path, current, expected_after } }
pub enum DiffLine { Context(String), Removed(String), Added(String) } // render() -> " x"|"-x"|"+x"
pub struct DiffResult { pub path: String, pub diff: String }
pub const DIFF_MAX_LINES: usize; // 2000, hard bound on a wire diff

pub struct CheckpointStore; // ::new(cas: Arc<Cas>, store: Arc<Store>)
pub fn before_write(&self, session, path, content: &[u8]) -> Result<FileHash, Error>
// stores the after-content blob in the CAS (deduped); content/hash mismatch is loud
pub fn after_write(&self, session, path, before: FileHash, after: FileHash, sequence: i64, after_content: &[u8]) -> Result<i64, Error>
pub fn rollback(&self, workspace: &WorkspaceHandle, identity: &WorkspaceIdentity, session: SessionId, checkpoint_id: i64) -> Result<RollbackOutcome, Error>
pub fn redo(&self, workspace, identity, session: SessionId, checkpoint_id) -> Result<RollbackOutcome, Error> // unrevert; Conflict when current != before_hash; pre-v3 rows (no after blob) refused honestly
pub fn diff_latest(&self, workspace, identity, session: SessionId) -> Result<Option<DiffResult>, Error> // Ok(None) = no checkpoints
pub fn checkpoints(&self, session) -> Result<Vec<CheckpointRow>, Error>
pub fn diff_lines(before: &[u8], after: &[u8]) -> Vec<DiffLine> // prefix/suffix, 3 lines of context, bounded
```

## faktor-session (`crates/session`)

`SessionManager` exposes the shared durable store/CAS to snapshot consumers:

```rust
pub fn store(&self) -> Arc<Store>
pub fn cas(&self) -> Arc<Cas>
```

## faktor-protocol (already implemented, `crates/protocol`)

```rust
pub mod v756 {
    pub const HANDSHAKE_PREFIX: &str;
    pub struct Handshake { version, protocol, pid, auth_token, port } // to_line(), from_line()
    pub struct HelloRequest { client, version }
    pub struct HelloResponse { ok, version, protocol, auth_required, providers: Vec<String> }
    pub struct CreateSessionRequest { provider, model, workspace: Option<String>, title: Option<String> } // deny_unknown_fields
    pub struct CreateSessionResponse { id, title, created_ms }
    pub struct PromptRequest { prompt, files: Vec<String>, op_id: Option<String> }
    pub struct PromptResponse { op_id, accepted, queued }
    pub struct MessagesQuery { before: Option<i64>, limit: i64, events_after: Option<i64> }
    pub struct MessagesPage { session_id, messages: Vec<Message>, has_more, next_before: Option<i64> }
    pub struct SessionState { session_id, state, title, last_event_seq, agent_state: AgentStateView, task_ledger: Option<Value> }
    pub struct AgentStateView { state, label, active, terminal }
    pub struct PermissionDecisionRequest { permission_id, decision }
    pub struct PermissionDecisionResponse { ok }
    pub struct AbortRequest { op_id: Option<String> }
    pub struct AbortResponse { aborted: Vec<String> }
    pub struct ProviderList { providers: Vec<ProviderInfo> }
    pub struct ProviderInfo { id, name, kind, models: Vec<ModelInfo> }
    pub struct ModelInfo { id, name, capabilities: ModelCapabilities }
    pub struct Message { id, role, session_id, seq, created_ms, parts: Vec<Part> }
    pub enum Part { Text{text}, Reasoning{text}, ToolCall{tool_call_id,name,input,state},
                    ToolResult{tool_call_id,result: ToolResultBody}, Summary{text} } // serde tagged "type"
    pub struct ToolResultBody { excerpt, exit_code: Option<i32>, artifact: Option<String>, slice_hint: Option<String> }
    pub struct ProviderConfig { id, kind, base_url, api_key_env: Option<String>, models: Vec<ModelConfig> }
    pub struct ModelConfig { id, name, capabilities }
}
pub mod sse {
    pub enum SseEvent { SessionUpdated{..}, MessageCreated{..}, MessagePartUpdated{..},
        ToolCallState{..}, PermissionRequested{permission_id,session_id,capability,detail},
        AgentStateChanged{..}, AgentManagerUpdate{update}, Compaction{..}, Error{session_id,code,message} }
    // to_frame(seq)->String, from_frame(&str)->Option<(u64,SseEvent)>, event_type()
    pub fn project_event(&Event) -> Option<(SseEvent, EventKind)>
    pub fn state_event(session_id: &str, state: AgentState) -> SseEvent
}
pub struct ApiError { code, message, http_status, retryable } // from_core(&Error)->ApiError, to_json()
```

## faktor-context (`crates/context`)

```rust
pub struct ContextBudget { pub system: usize, pub tools: usize, pub working: usize,
    pub retrieved: usize, pub recent: usize, pub output_reserve: usize, pub safety: usize }
// ::default() = 32K local profile (5K+3K+7K+10K+5K+2K); ::for_capabilities(&ModelCapabilities)
// total(), context_max() (= total - output_reserve - safety), effective_usage(used) -> f64

pub enum MemoryClass { StaticPrefix, SemiStable, Volatile }
pub struct ContextSection { class, text, tokens }
pub struct AssembledContext { sections, total_tokens, cacheable_tokens, volatile_start } // render()
pub struct RecentTurn { role, text }
pub struct Evidence { path, snippet, score }
pub struct ContextAssembler; // ::assemble(static_prefix, system_extra, tool_schemas, project_rules,
//   ledger, repo_map, recent_turns, retrieved_evidence, errors, budget) -> Result<AssembledContext, Error>

pub struct TaskLedger { goal, constraints, completed_steps, open_steps, decisions, known_failures,
    changed_files, tests_run, tests_failed, user_preferences }
// ::record_turn(&TurnSummary), compact_render(), token_estimate(), validate_sane()
pub struct TurnSummary { steps_completed, steps_opened, decisions, failures, files_changed,
    tests_run, tests_failed }

pub struct CompactionRequest { before_tokens, target_tokens, min_reduction_ratio } // ::new(before, target)
pub enum CompactionStrategy { LlmSummary, DeterministicPruning, Rejected }
pub struct CompactionPlan { accepted, before_tokens, after_tokens, target_tokens, strategy,
    ledger, kept_recent: Vec<RecentTurn>, archived }
pub struct Compactor; // ::new(Option<Arc<dyn Summarizer>>), deterministic_only(),
//   compact(&[RecentTurn], &TaskLedger, &CompactionRequest) -> CompactionPlan
pub trait Summarizer { fn summarize(&self, &[RecentTurn], &TaskLedger) -> String }

// Audit round 5 (P0): the budget bounds the ACTUAL wire request.
pub struct WirePlan { pub system: String, pub messages: Vec<RequestMessage>,
    pub tools: Vec<ToolSpec>, pub total_tokens: usize }
pub fn plan_wire_request(instructions: &str, system_extra: &str, tool_schemas: &[ToolSpec],
    project_rules: &str, ledger: &TaskLedger, repo_map: &str, history: &[RequestMessage],
    evidence: &[Evidence], errors: &str, budget: &ContextBudget) -> Result<WirePlan, Error>
// system = static prefix + semi-stable + volatile tail (evidence, errors); tools appear ONLY in
// `tools`; history ONLY in `messages`. total_tokens = estimate(system) + Σ estimate(message) +
// estimate(tools) <= budget.context_max(), enforced before anything is sent. Trimming
// (deterministic): oldest history messages first, pairing-aware (dropping an assistant
// tool-call message also drops the following user message carrying its tool result — a result
// never dangles without its call); then evidence; then errors; still over with empty history →
// Err(Oversized). Never returns an unbudgeted plan.
```

## faktor-provider (already implemented, `crates/provider`)

```rust
pub enum Role { User, Assistant, System }
pub struct ContentPart { kind: ContentKind, tool_call_id: Option<String> } // ContentPart::text(...)
pub enum ContentKind { Text{text}, Image{url}, ToolCall{id,name,input}, ToolResult{content,is_error} }
pub struct RequestMessage { role, content: Vec<ContentPart> }
pub struct ToolSpec { name, description, input_schema: Value }
pub struct RequestMeta { operation_id: OpId, session_id: SessionId, provider, attempt: u32, deadline_ms: u64, cancellation: CancellationToken }
pub struct GenericAgentRequest { model, system, messages: Vec<RequestMessage>, tools: Vec<ToolSpec>,
    max_output: Option<usize>, reasoning: Option<ReasoningMode>, stream: bool, meta: RequestMeta }
pub enum ProviderChunk { Text{text}, Reasoning{text}, ToolCall{id,name,input,complete}, Usage{tokens_in,tokens_out}, Done }
pub enum ProviderErrorKind { Network, Timeout, RateLimited, BadRequest, Auth, Server, Cancelled, Malformed }
pub struct ProviderError { kind, message, retryable, code: Option<String> }
pub type ProviderStream = Pin<Box<dyn Stream<Item=Result<ProviderChunk, ProviderError>> + Send>>;
pub trait Provider: Send+Sync { fn id(&self)->&str; fn identity(&self)->ProviderIdentity; fn capabilities(&self, model:&str)->ModelCapabilities; fn stream(&self, req: GenericAgentRequest)->ProviderStream; }
pub struct ProviderIdentity { pub instance_id: String, pub family: String } // ::new(instance, family), from_family(family)
// identity() defaults to instance_id == family (one instance per family);
// the registry keys by identity().instance_id. InstanceProvider::wrap(inner, instance_id)
// overrides the registry id while keeping the adapter family id for capability queries.
pub struct CapabilityValidator; // validate(&req, &caps)->Result<(),Error>
pub struct RequestNormalizer; // normalize(&req)->NormalizedRequest (internal fields never on wire)
pub struct ProviderRegistry; // register(Arc<dyn Provider>) keys by identity().instance_id; get(id)->Option<Arc<dyn Provider>>, ids(), capabilities(provider,model)->Option<ModelCapabilities>
pub struct FakeProvider; // with_script(id, caps, script: Vec<ScriptedResponse>), die_mid_stream(...), inject_rate_limit()
pub enum ScriptedResponse { Text(String), ToolCall{id,name,input}, End, Die(ProviderError) }
```

## Contract notes

- `faktor-core` is std-only (no tokio). Everything else may use tokio.
- Store/Cas are sync; async callers wrap heavy ops in `tokio::task::spawn_blocking`
  or call directly for short ops.
- IDs are never zero; `EventSeq` per session starts at 1 (`SessionCreated`).
- SSE resume cursor = event sequence; `events_after(session, seq)` is the source.
- No crate may import `faktor-store`/`faktor-cas` from provider adapters.
  Provider code cannot touch session persistence (Commandment 1).
