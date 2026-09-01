# Kilo+ architecture (frozen specification)

This document is the normative spec the implementation must satisfy.
It is stored verbatim; the code is the derived artifact.

Section numbers cited in crate comments (e.g. "spec §9") refer to the
sections below.

---

## 1. Product definition

Kilo+ is a native Rust engine that reproduces the Kilo Code v7.5.6
experience byte-for-byte. The UI is a frozen compatibility fixture; the
engine under it is a durable, journaled, bounded runtime. The LLM is used
only where reasoning is actually needed — indexing, retrieval, compaction
of historical turns, and deterministic bookkeeping are local.

**Frozen baselines (never merged wholesale from later releases):**

- **VS Code UI:** Kilo Code v7.5.6 webview, CSS, images, message layout —
  byte-for-byte fixture in `apps/vscode/`.
- **JetBrains shell:** JetBrains 7.1.2 Kotlin frontend in `apps/jetbrains/`;
  only the process manager is modified, and only to launch the Kilo+ binary.
- **Protocol:** the implemented v7.5.6 server contract subset (§16),
  frozen as golden fixtures in `compat/kilo-v756/`. The Rust daemon must
  pass this compatibility suite before the old backend is removed.

**Non-goals:** reimplementing the old TypeScript/Bun engine, improving the
wire protocol, or merging newer UI releases wholesale.

---

## 2. High-level architecture

```
┌────────────────────────── IDE clients (frozen) ──────────────────────────┐
│  VS Code v7.5.6 webview (apps/vscode)   JetBrains 7.1.2 shell (apps/jetbrains) │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ HTTP + SSE (frozen v7.5.6 protocol)
                                     ▼
                    ┌───────────────────────────────────┐
                    │  kilop-server  (HTTP/SSE surface) │  auth: KILO_SERVER_PASSWORD
                    │  startup line, /global/event bus  │  (Basic | Bearer | x-kilo-server-password)
                    └────────────────┬──────────────────┘
                                     │ commands (kilop-session API, synchronous)
                                     ▼
        ┌────────────────────────────────────────────────────────────┐
        │  Agent runtime  (kilop-agent: the durable turn loop)       │
        │  drives session state, streams providers, schedules tools, │
        │  keeps context bounded (kilop-context)                     │
        └─────────┬──────────────────────────────┬───────────────────┘
                  ▼                              ▼
     ┌───────────────────────┐      ┌──────────────────────────────────────┐
     │  Provider layer       │      │  Tool runtime                         │
     │  kilop-provider       │      │  kilop-scheduler  (DAG + budgets)     │
     │  ollama openai        │      │  kilop-terminal  (supervised child)   │
     │  anthropic google     │      │  kilop-edit      (transactional)      │
     │  deepseek gateway     │      │  kilop-snapshot  (CAS checkpoints)    │
     │  (normalized chunks)  │      │  kilop-fs / git / mcp / lsp / sandbox │
     └───────────────────────┘      │  kilop-index / search / memory        │
                                    └──────────────────────────────────────┘
                                     all state exits through commands/events
                                     ▼
                    ┌──────────────────────────────────────┐
                    │  kilop-session (durable runtime)     │
                    │  journal, state machines, tool-run   │
                    │  ledger, permissions, recovery       │
                    └──────────────┬───────────────────────┘
                                   ▼
              ┌───────────────────────────────┐
              │  kilop-store (SQLite, WAL)    │
              │  kilop-cas (BLAKE3+Zstd blobs)│
              └───────────────────────────────┘
```

Rules that shape the diagram (Commandments):

1. **Dependencies point inward toward pure core types.** `kilop-core` has
   no workspace dependencies and no I/O. Provider code never touches
   session persistence; tools never mutate session state directly.
   Everything enters the runtime through commands/events.
2. **Every async operation is explicit.** No bare `tokio::spawn` defines
   application state. Operations carry `operation_id, session_id, state,
   start_time, deadline, retry_policy, cancellation_token,
   recovery_strategy` (see §7).
3. **No `if provider == "..."` in the agent.** Provider behavior is decided
   by `ModelCapabilities` and provider profiles; quirks stay inside
   adapters (see §9).
4. **Bounded everything.** No unlimited command output in RAM or context
   (ring buffer + compressed artifact, §10), no unbounded transcripts
   (paging is fundamental), no unbounded resource lifetimes (explicit
   scopes, idle unload).
5. **Zero orphans.** Every child process has a runtime owner; if the
   session dies, ownership transfers deliberately or the process dies (§10).
6. **Wire compatibility is a frozen contract** (§16).

---

## 3. Workspace layout

```
apps/        frozen UI compatibility fixtures
  vscode/       Kilo Code v7.5.6 webview (byte-for-byte UI)
  jetbrains/    JetBrains 7.1.2 Kotlin shell (process manager launches the Kilo+ binary)
compat/      permanent protocol fixtures
  kilo-v756/    frozen v7.5.6 wire contract (golden JSON fixtures)
  jetbrains-712/ reserved JetBrains split-mode fixture corpus
fixtures/    protocol/, providers/, screenshots/, repositories/ test data
tests/       integration/, soak/, performance/, fault/, visual/ (adversarial-only)
crates/      the Rust engine workspace
docs/        architecture.md (this spec), api-contracts.md, specs/*.md (frozen sub-specs)
```

Crate responsibilities (each crate's module doc is authoritative):

| Crate | Responsibility |
|---|---|
| `core` | Pure types, std-only, no workspace deps: IDs, `Error`/`ErrorKind`, `AgentState` + `SessionLifecycle` machines, `EventKind`/`Event`, `OpMeta`/`RecoveryStrategy`/`EffectStatus`, `CancellationToken`, `Clock`/`Deadline`, `RetryPolicy`, `ModelCapabilities`, `ResourceClass`/`ResourceLimits`, `Capability`/`PermissionDecision`/`NetworkPolicy`, `FileHash`, `WorkspaceIdentity`. |
| `cas` | Content-addressed blob storage: BLAKE3 identity + Zstd compression, sharded layout (`ab/cdef…`), atomic writes (temp + fsync + rename), reads verify the hash. |
| `store` | SQLite persistence: WAL, single logical writer + bounded reader pool, busy timeout, explicit transactional migrations, integrity checks, automatic backups. Large blobs live in the CAS; SQLite stores hashes. Message/part rows store JSON so the store stays protocol-agnostic. |
| `protocol` | The frozen v7.5.6 wire contract (`v756` shapes, `sse` frames, `ApiError` mapping, `fixtures`). Golden tests lock request/response/SSE/JSON-field-presence/null-behavior/error-code behavior against `compat/kilo-v756/`. |
| `session` | The durable half of a session: journaled state machine (commands append events through `StateMachine`-validated transitions), conversation view, tool-run ledger, permission requests, checkpoints, memory facts, compaction records, crash recovery, owned processes. Synchronous `Send + Sync` API on `kilop-store` + `kilop-cas`. |
| `agent` | The durable agent reasoning loop: drives the session with commands, streams providers, schedules tools through `kilop-scheduler`, keeps context bounded via `kilop-context`. No provider-name conditionals; state-aware continuation; repair once, never five times; loop detection. |
| `context` | Bounded context construction (five memory classes, §8), durable task ledger, compaction engine that cannot death-spiral, artifact writer, budget, estimator. |
| `provider` | The common provider interface hub: `Provider` trait, `GenericAgentRequest` pipeline (capability validation → normalization → adapter wire serializer → adapter transport), `ProviderRegistry`, `FakeProvider` scripted test harness, `testing` mock HTTP server. |
| `ollama` | Native Ollama adapter: discovery via `GET /api/tags`, capability probing via `GET /api/show`, native `/api/chat` streaming with tools/thinking/keep_alive, `/api/embed` embeddings; OpenAI-compatible mode is a fallback only. |
| `openai` | OpenAI Chat Completions, Responses, and OpenAI-compatible endpoints. Wire serializer produces exactly the frozen OpenAI shapes; internal option names can never leak onto the wire (locked by tests). |
| `anthropic` | Anthropic Messages adapter (stream events, `tool_use` accumulation, `input_json_delta`). |
| `google` | Gemini streaming adapter (candidates → parts → functionCall). |
| `deepseek` | First-class DeepSeek profiles (direct, OpenRouter, Kilo Gateway, arbitrary compatible, local derivatives); capability normalization after discovery. |
| `gateway` | Kilo/OpenRouter-style gateway adapters: OpenAI-compatible endpoint with model routing and extra headers; BYOK preserved, gateway key never persisted. |
| `scheduler` | Tool/subagent concurrency as a dependency DAG with resource-class budgets, state-aware retries with jitter, circuit breakers. Independent reads/subagents run concurrently; edits touching overlapping ownership sets do not. |
| `terminal` | Process supervision: no orphans, process groups (Unix) / Job Objects (Windows), 200-line ring buffer live with CAS-artifact spill, dedicated reader threads. |
| `edit` | Transactional patch engine: `expected_hash` versioning, validate-against-copy then ONE atomic write, parse-before-accept via tree-sitter for supported languages. |
| `snapshot` | Native content-addressed checkpoints: before-content stored once in CAS (dedup free), rollback verifies current == recorded after-hash then atomically writes before; an independently changed file is a `Conflict`, never silently overwritten. |
| `fs` | File service and watcher: explicit workspace identity on every call, traversal/symlink-safe path resolution, bounded reads, atomic writes, idle unload of heavyweight resources. |
| `git` | Worktree management and per-repository mutation locks. Every invocation is a supervised child with an explicit owner; reads run concurrently, mutations serialize per repository (never a global Git lock). |
| `mcp` | JSON-RPC Model Context Protocol client: supervised like terminals, every invocation has a deadline, responses bounded, Content-Length framing. |
| `lsp` | Workspace-scoped language server integration: one daemon shares servers across sessions on a workspace, per-request id multiplexing, heavy servers unloaded after workspace inactivity. |
| `sandbox` | Capability-based permission enforcement: canonicalization-safe path checks (symlink escapes and `..` rejected), three-mode network policy. |
| `index` | Hybrid repository index: lexical inverted index plus tree-sitter symbol index (Rust/Python), incrementally updated, bounded by caps, workspace-isolated. |
| `search` | Hybrid retrieval with rank fusion: exact + lexical + symbol (+ optional semantic) fused by reciprocal rank weighted by symbol relevance, lexical score, semantic score, file recency, task affinity; evidence packages retrieved before serious reasoning turns. |
| `memory` | Long-term structured session memory: durable task state and structured facts (the transcript is *not* memory), compact context render for the semi-stable memory class. |
| `server` | The HTTP/SSE surface of the daemon, speaking the frozen v7.5.6 protocol. The UI connection is disposable: turns run detached from any SSE connection and resume from the journal. |
| `cli` | `serve` (prints the frozen startup line), `run` (headless one-prompt), `doctor` (self-check: store, CAS, integrity, permissions, providers), `sessions` (list). Logging goes to stderr; stdout is the startup-line contract. |

---

## 4. Durable state machine

There is **no generic `await Promise` that determines application state**.
Every session is an explicit state machine; every transition is validated
before any write, and an illegal transition leaves **no trace** in the
journal or the session row.

### 4.1 The turn machine (`AgentState`, 17 states)

`kilop-core` defines `AgentState` (wire tags are `snake_case`):

```
Idle, Preparing, BuildingContext, WaitingForModel, Streaming,
ToolRequested, WaitingForPermission, ExecutingTool, Validating,
UpdatingMemory, ReadyForNextTurn, Completed, Cancelled,
FailedRecoverable, FailedPermanent, NeedsUserInput, Suspended
```

**Legal transitions** (`AgentState::allowed_transitions`, verified
exhaustively by tests):

| From | May transition to |
|---|---|
| `Idle` | `Preparing`, `Suspended`, `Completed`, `Cancelled` |
| `Preparing` | `BuildingContext`, `FailedRecoverable`, `Cancelled`, `Suspended` |
| `BuildingContext` | `WaitingForModel`, `FailedRecoverable`, `Cancelled`, `Suspended` |
| `WaitingForModel` | `Streaming`, `FailedRecoverable`, `Cancelled`, `Suspended`, `NeedsUserInput` |
| `Streaming` | `ToolRequested`, `Validating`, `WaitingForModel`, `FailedRecoverable`, `Cancelled`, `Suspended` |
| `ToolRequested` | `WaitingForPermission`, `ExecutingTool`, `Validating`, `Cancelled`, `Suspended` |
| `WaitingForPermission` | `ExecutingTool`, `ReadyForNextTurn`, `Cancelled`, `Suspended`, `NeedsUserInput` |
| `ExecutingTool` | `Validating`, `ToolRequested`, `FailedRecoverable`, `Cancelled`, `Suspended` |
| `Validating` | `UpdatingMemory`, `ToolRequested`, `FailedRecoverable`, `Completed`, `Cancelled`, `Suspended` |
| `UpdatingMemory` | `ReadyForNextTurn`, `WaitingForModel`, `Completed`, `FailedRecoverable`, `Cancelled`, `Suspended` |
| `ReadyForNextTurn` | `Preparing`, `Completed`, `Cancelled`, `Suspended`, `NeedsUserInput` |
| `Completed` | *(none — terminal)* |
| `Cancelled` | `Preparing`, `ReadyForNextTurn`, `Suspended` |
| `FailedPermanent` | *(none — terminal)* |
| `FailedRecoverable` | `Preparing`, `Idle`, `Cancelled`, `Suspended`, `NeedsUserInput` |
| `NeedsUserInput` | `ReadyForNextTurn`, `Preparing`, `Cancelled`, `Suspended` |
| `Suspended` | `Idle`, `Preparing`, `Cancelled`, `Completed` |

Contract rules:

- **Self-transitions are legal and idempotent** — replay of re-emitted
  events must not fail.
- **Skipping states is rejected** (e.g. `Streaming → UpdatingMemory`
  without `Validating` is illegal).
- **`Cancelled` is a turn outcome, not a session outcome.** Stop in Kilo
  cancels the turn; the chat stays promptable (`Cancelled → Preparing` and
  `Cancelled → ReadyForNextTurn` are legal). `is_terminal()` is true only
  for `Completed` and `FailedPermanent`.
- `is_active()` = not terminal and not `Idle`/`Suspended`.
- `StateMachine::force` exists for recovery/replay only, never normal
  flow; setting a terminal state that has recorded later events is a
  corruption sign — callers must validate against the journal before
  forcing.

### 4.2 The session LIFETIME machine (`SessionLifecycle`, 5 states)

The per-turn machine alone cannot express session lifetime — a session is
Open for days across many turns. `SessionLifecycle` is **orthogonal** to
`AgentState` and persists on the session row:

```
Open, Suspended, Closing, Closed, FailedPermanent
```

**Legal transitions** (`SessionLifecycle::allowed_transitions`):

| From | May transition to |
|---|---|
| `Open` | `Suspended`, `Closing`, `Closed`, `FailedPermanent` |
| `Suspended` | `Open`, `Closing`, `Closed`, `FailedPermanent` |
| `Closing` | `Closed`, `FailedPermanent`, `Open` |
| `Closed` | *(none — terminal)* |
| `FailedPermanent` | *(none — terminal)* |

Contract rules:

- **Prompts are gated on `Open`** (`can_accept_prompts()`). A prompt on a
  `Suspended` session auto-resumes it (`Suspended → Open`); prompts on
  `Closing`/`Closed`/`FailedPermanent` are rejected with `Conflict`.
- **`end_session()` is the only normal terminal route.** It journals
  `SessionEnded` and moves the lifecycle `→ Closed`. Registered child
  processes block `end_session` until released or deliberately
  transferred (zero orphans).
- **`abort` cancels the turn, never the session.** It cancels every
  tracked operation's token, journals `ToolCancelled` (tool ops) or
  `Failed {"error":"aborted"}` (turn ops) landing state `Cancelled`, then
  journals `TurnCompleted` landing **`ReadyForNextTurn`**. The session
  stays promptable.
- `FailedPermanent` is reachable only as a documented escalation from
  `FailedRecoverable` (journal validation requires the two-step hop).

---

## 5. The append-only event journal

Every significant transition becomes a journal event. The session database
therefore knows exactly what happened; the rendered conversation is a
*view derived from the journal*, never the source of truth.

### 5.1 Event kinds (24, frozen)

```
SessionCreated, PromptReceived, ContextPrepared, ModelStarted,
ModelChunkReceived, ToolRequested, ToolStarted, FileChanged,
ToolCompleted, ToolCancelled, CheckpointCreated, ContextCompacted,
CompactRejected, SubagentStarted, SubagentCompleted, TurnCompleted,
PermissionGranted, PermissionDenied, CrashDetected, RecoveryApplied,
SessionEnded, Suspended, Resumed, Failed
```

### 5.2 Sequencing and structure

- Each session has a **gapless per-session sequence** (`EventSeq`, starts
  at 1 with `SessionCreated`); the sequence is the SSE resume cursor.
- `Event { seq, session_id, op_id: Option<OpId>, kind, state, ts_ms,
  payload: Option<Value> }`. The `state` column is the machine state the
  event *lands on*; every event's state must be a legal transition from
  the previous state (`validate_transition` in the session journal).
- Two kinds carry documented sub-chains: `ToolRequested` events are
  recorded with state `WaitingForPermission` (hop `ToolRequested` then
  `WaitingForPermission`, both must be legal); `Failed` recorded with
  state `FailedPermanent` is legal only via the two-step
  `FailedRecoverable`-then-force escalation.
- `StateMachine::transition` is validated **before** any write; illegal
  transitions return `ErrorKind::InvalidState` and leave no trace.

### 5.3 Durable vs ephemeral split

- **Durable (journaled):** model lifecycle (`ModelStarted`,
  `ModelChunkReceived` with `message_id` + `text_len` only — never the
  text), tool lifecycle (`ToolRequested/Started/Completed/Cancelled`),
  `FileChanged`, checkpoints (`CheckpointCreated`), compaction
  (`ContextCompacted`/`CompactRejected`), turn (`PromptReceived`,
  `TurnCompleted`), permissions, crash recovery, session lifecycle.
- **The text itself is not journaled per chunk.** Chunk text accumulates
  in message *parts* (SQLite JSON rows); the journal records lengths so
  replay can re-derive shape without unbounded journal growth.
- **Ephemeral/coalesced on the wire:** text deltas, reasoning deltas, and
  terminal output are reconstructed/coalesced by the SSE pipeline (§11):
  the `GlobalEventBus` re-diffs stored message parts (`recover_text`) and
  runs consecutive text deltas through a 50 ms / 8 KiB `DeltaCoalescer`
  before framing. Terminal output streams as
  `interactive_terminal_data` frames and is never journaled.

### 5.4 Journal validation

`replay_journal()` replays the full journal through the state machine and
fails with `Internal` on corruption (skipped states, illegal sequences).
Recovery uses it to detect that a crash never left a corrupt trace.

---

## 6. Crash recovery

**Never blindly re-run a command after a crash.** Unfinished operations
are reconstructed from durable state; deterministic FS ops verify the
expected resulting hash; unknown external effects are marked
`effect_status = unknown` and forced to verification.

### 6.1 The five `RecoveryStrategy` variants (core)

| Variant | Meaning | Recovery action |
|---|---|---|
| `VerifyHash { path, expected: FileHash }` | Deterministic FS op | Hash the file (through the CAS, bounded read). Matches → the op completed before the crash (`completed` / `verified`); mismatch or missing → the op truly never ran (`failed` / `failed`). |
| `MarkUnknown` | Command with unknown external effects | Record `effect_status = unknown` (`interrupted`) and force verification instead of re-running. |
| `Idempotent` | Safe to re-run (reads, idempotent calls) | Mark failed so the scheduler may re-run. |
| `Manual` | Never re-run automatically | Require a human (`interrupted`, `needs_human`). |
| `None` | No recovery action | Record `interrupted`, no action. |

### 6.2 How recovery decides

`SessionHandle::recover_all` (and the agent's `recover_session` before a
turn):

1. Scans durable `tool_run` rows still `running` (the only durable record
   that survives a crash) and applies each row's recorded strategy.
2. Journals `CrashDetected`, then one `RecoveryApplied` per recovered op
   with `status`, `effect`, and `action` tags
   (`verified`/`not_applied`/`unknown_effect`/`rerun_allowed`/
   `needs_human`/`no_action`).
3. **Idempotent:** a second sweep finds nothing pending and appends
   nothing (test-locked).
4. **Interrupted turn detection:** an op-active session with *no* pending
   tool rows means the crash hit the model stream itself — journal
   `CrashDetected` and land on the honest target: `FailedRecoverable` →
   `WaitingForPermission` (durable permission requests are resumable only
   from that state) → `NeedsUserInput` → stay put.
5. **Contradiction detection:** pending tool rows while the journal says
   idle/suspended/terminal → fix the rows, keep the state, set the
   `contradiction` flag.
6. **Tamper detection:** the `expected_hash` column must agree with the
   strategy's `expected` hash, or recovery fails `Malformed`; an unknown
   strategy tag in a row is `Malformed`, never silently defaulted.
7. **Orphans:** children owned at restart are reported and cleared — the
   runtime must never pretend to own zombies (parent-death handling is the
   OS's job; the runtime's is to notice and record).
8. Hash verification is performed **through the CAS** (`SystemFileHasher`
   puts the verified content into the CAS as a deduplicated audit
   artifact, bounded by `MAX_VERIFY_BYTES`).

### 6.3 `EffectStatus` semantics

`Unknown | Verified | Applied | Failed`, persisted on `tool_run` rows:

- `Unknown` — effects not determinable (crash mid-command); the op is
  forced to verification before its output may be reused.
- `Verified` — a deterministic check confirmed the effect (hash match).
- `Applied` — the op completed normally during the turn.
- `Failed` — the op failed (or hash mismatch proves it never ran).

---

## 7. Operation metadata and the one-operation model

### 7.1 `OpMeta` (core)

Every async operation carries the full envelope from the spec:

```rust
pub struct OpMeta {
    pub operation_id: OpId,
    pub session_id: SessionId,
    pub state: OpState,          // Pending | Running | Done | Failed | Cancelled
    pub start_time_ms: i64,
    pub deadline: Deadline,      // ensure_alive(): cancelled → Cancelled, expired → Timeout
    pub retry_policy: RetryPolicy,
    pub cancellation: CancellationToken,
    pub recovery: RecoveryStrategy,
}
```

`ensure_alive(now_ms)` fails fast on cancellation or deadline expiry
*before any work*.

### 7.2 The one-operation-model rule

A single operation has **exactly one identity — one `OpId`** — shared by
every component that touches it:

- the **scheduler** `ScheduledOp { meta: OpMeta, ... }` (same id, same deadline,
  same retry policy, child cancellation token);
- the **durable tool-run row** (`start_tool_run(op_meta.clone(), ...)` in
  `kilop-session`, which also registers the op + token in the session's
  `OpRegistry` so abort fans cancellation out);
- the **provider request** (`RequestMeta { operation_id, session_id,
  provider, attempt, deadline_ms, cancellation }` — never serialized to
  the wire);
- the **journal** events carrying `op_id`.

There is no second identity space: cancel the `OpId`, and scheduler task,
tool row, provider stream, and journal all observe the same token.
Turn ops are `OpKind::Turn`; tool ops are `OpKind::Tool` (drives abort's
event kind). Tools are never blindly retried: their `OpMeta` retry policy
has `max_attempts: 1`; retries are a scheduler concern with explicit
policies.

---

## 8. Context engine

### 8.1 Five memory classes (spec §8)

| Class | Content | Treatment |
|---|---|---|
| 1. Immutable instructions | system prompt, tool schemas, project rules | static, cacheable prefix |
| 2. Durable task state | the task ledger (decisions, constraints, files touched) | compact, kept whole |
| 3. Repository knowledge | retrieved evidence (exact/lexical/symbol/semantic) | bounded by budget |
| 4. Recent conversation | recent turns from the message view | paged, bounded, oldest archived |
| 5. Historical artifacts | summarized/compacted history | CAS artifacts + summaries |

The budget is enforced **BEFORE anything is sent to a provider** — the
engine never discovers the limit from a provider error.

### 8.2 `ContextBudget` — the 32K local profile math

Default (`ModelCapabilities::small_local()` profile — context ≤ 32,768):

```
system         5,000
tools          0
working        3,000
retrieved      7,000
recent        10,000
output_reserve 5,000
safety         2,000
────────────── ──────
total         32,000   (exact, test-locked)
context_max   25,000   (total − output_reserve − safety; test-locked)
```

Larger contexts scale proportionally (working 4K; recent = clamp
(context/8, 10K..60K); retrieved = clamp(context/12, 7K..24K); output
reserve = max_output + 1K capped at context/4; safety = max(context/32,
2K); system = clamp(context/16, 5K..12K); tools = remainder capped 8K).
Hostile capability metadata (usize::MAX, zero context) saturates, never
panics (test-locked).

### 8.3 Compaction hard invariant

`kilop-context` + `kilop-session` enforce, with **no exceptions**:

- A successful compaction must land **at or below the target** (`after ≤
  target`) **and** achieve the configured minimum reduction: `reduction =
  1 − after/before ≥ min_reduction_ratio` (**default 0.25**).
- A "summary" that reduces context by ~1% is **rejected** with a
  `CompactRejected` journal event, and deterministic pruning takes over.
- `hard_cap = min(target, before × (1 − ratio))` — the compactor can never
  accept a plan above it.
- **Death-spiral convergence:** the compactor retries with the *plan's*
  after-tokens as the new before only while it is still converging toward
  the hard cap; a compaction that cannot shrink is terminal. Compaction is
  an **interior event**: the session state does not move.
- Accepted strategies: `LlmSummary` (weak-but-honest summarizer output is
  rejected when it does not shrink enough) or `DeterministicPruning`
  (task ledger preserved whole; only old recent turns archived).
- Proactive trigger: effective usage (`used / context_max`) ≥
  `compact_at_usage` (default 0.65) before a provider call.

### 8.4 Prefix caching ordering

The assembler orders memory classes for prefix caching:

```
StaticPrefix     (instructions + tools + project rules)   — fully cacheable
SemiStable       (task ledger/durable state)              — cacheable with prefix
Volatile         (recent turns, evidence)                 — always after
```

The static prefix plus semi-stable tokens form the provider-cacheable
prefix; volatile content is appended last so provider prompt caching is
not invalidated by every new turn.

---

## 9. Provider architecture

### 9.1 The request pipeline

Every provider call passes the same pipeline:

```
GenericAgentRequest
        ↓
CapabilityValidation   (CapabilityValidator: tools/reasoning/max_output vs ModelCapabilities;
                         violations are loud errors, never silent truncation)
        ↓
Provider Normalizer    (RequestNormalizer → NormalizedRequest: whitelisted fields only;
                         internal meta — operation_id/session_id/attempt/deadline/cancellation —
                         structurally cannot leak onto the wire; 7-field frozen shape, test-locked)
        ↓
Wire Serializer        (inside each adapter: exactly the frozen vendor shapes)
        ↓
HTTP Transport         (inside each adapter; state-aware retries, circuit breakers)
```

`GenericAgentRequest { model, system, messages, tools, max_output,
reasoning, stream, meta }`; chunks are normalized
(`Text | Reasoning | ToolCall{id,name,input,complete} | Usage | Done`);
errors are `ProviderError { kind, message, retryable, code }` with
retryability decided by kind (`Network | Timeout | RateLimited | Server`
retryable; `BadRequest | Auth | Cancelled | Malformed` not).

### 9.2 `ModelCapabilities` is the only behavior switch

`context, max_output, tools, parallel_tools, thinking, vision,
json_schema, streaming, embeddings, reasoning`. Defaults are conservative
(no tools, no thinking, small context) so an unprobed model **fails safe,
not loud**. There is **no `if provider == "…"` anywhere in the agent** —
the agent reads capabilities; adapters set them; provider quirks stay
inside adapters. Capabilities come from discovery/probing, never
hard-coded lists.

### 9.3 Adapter families

| Family | Wire | Notes |
|---|---|---|
| `ollama` | native `/api/chat` | Discovery via `GET /api/tags` (never a hard-coded list — `ollama pull` makes models appear), capability probing via `GET /api/show`, native streaming with tools/thinking/keep_alive, `/api/embed` embeddings; OpenAI-compatible mode is a fallback only. |
| `openai` | Chat Completions, Responses, OpenAI-compatible | Wire serializer locked by tests so internal option names can never leak. |
| `anthropic` | Messages API | Owns stream events, `tool_use` accumulation, `input_json_delta`. |
| `google` | Gemini | candidates → parts → functionCall normalization. |
| `deepseek` | OpenAI-shaped | First-class profiles: `Direct` (api.deepseek.com), `OpenRouter`, `Gateway`, `Compatible` (arbitrary endpoint), `LocalDerivative`. Capability normalization after discovery; all profiles produce capability-driven behavior (test-locked matrix). |
| `gateway` | OpenAI-compatible | Kilo/OpenRouter-style model routing + extra headers; BYOK preserved, gateway key configured per provider and never persisted by the runtime. |

### 9.4 Registry and test matrix

- `ProviderRegistry` maps provider id → `Arc<dyn Provider>`; the agent
  asks the registry, never a provider string.
- `FakeProvider` is a scripted, adversarial harness (`with_script`,
  `die_mid_stream`, `inject_rate_limit`; scripts consumed exactly once —
  a replaying provider would let the loop re-execute forever).
- `provider::testing` is a mock HTTP server for wire-level adversarial
  tests.
- The capability-driven test matrix runs the same agent-level behaviors
  against every profile/family (e.g. DeepSeek's
  `all_profiles_produce_capability_driven_behavior`).

---

## 10. Tool runtime

### 10.1 Transactional editing (kilop-edit)

- Every agent edit is **optimistic and versioned**: `expected_hash` must
  match the current content or the edit is rejected **before any write**
  (`Conflict`).
- All ops are validated against a copy first, then applied with **ONE
  atomic write** (temp + fsync + rename) — no partial writes.
- **Parse-before-accept:** for supported languages (Rust, Python via
  tree-sitter), if the original parses cleanly and the edited version does
  not, the edit is suspicious: roll back (or flag, per mode).
- `FileHash` is the BLAKE3 identity used everywhere (CAS, edits,
  checkpoints, recovery).

### 10.2 Native CAS checkpoints (kilop-snapshot)

- Snapshots are **not Git repositories pretending to be undo history**.
  Before changing a file its original content is stored once in the CAS
  (BLAKE3 + Zstd; dedup is free: ten checkpoints of the same unchanged
  file = one copy). Git stays for branches/commits/worktrees/diffs only.
- `CheckpointRow { before_hash, after_hash }` records each write.
- **Rollback verification:** restore verifies the current file hash equals
  the recorded `after_hash`, then atomically writes the `before` content.
  An independently changed file is a `RollbackOutcome::Conflict`, never
  silently overwritten.

### 10.3 Process supervision (kilop-terminal)

- **Zero orphans.** Every child has a runtime owner (`ProcessOwner`:
  `Session | Workspace | Daemon`); kill targets the whole process group
  (Unix) or Job Object (Windows); `transfer` is the deliberate ownership
  handoff; `kill_all_for(owner)` runs on session death. Children
  registered on a session block `end_session` until released/transferred.
- **Bounded output.** A 200-line ring buffer is live; overflow spills to a
  CAS artifact (`artifact_max` default 100 MiB) — a 300 MB log never
  becomes a 300 MB RAM object. `CommandOutput { excerpt, exit_code,
  artifact, slice_hint, ring_lines }`; blocking pipe reads live on
  dedicated reader threads so they can never stall the async loop.
- `reap()` collects exited children (no zombies); kill is
  SIGTERM → grace → SIGKILL.

### 10.4 MCP / LSP supervision

- **MCP** (kilop-mcp): JSON-RPC over Content-Length framing; processes are
  supervised exactly like terminals (crashes, hangs, and garbage output
  never destabilize the agent runtime); every invocation has a deadline;
  responses are bounded.
- **LSP** (kilop-lsp): language servers are **workspace resources, not
  session resources** — one daemon shares `rust-analyzer` /
  `typescript-language-server` / `pyright` across sessions on the same
  workspace; requests multiplexed with per-request ids; heavy servers
  unloaded after workspace inactivity (idle unload).

### 10.5 Sandbox capabilities and network policy (kilop-sandbox)

- Permissions are expressed as `Capability` objects (`ReadWorkspace`,
  `WriteWorkspace`, `ReadExternal`, `WriteExternal`, `ExecuteShell`,
  `Network`, `Mcp`, `Git`), never scattered conditionals.
- Path checks are canonicalization-safe: symlink escapes and `..`
  traversal are rejected.
- `NetworkPolicy` is the three-mode sandbox: `DenyAll` /
  `AllowProviders { endpoints }` / `AllowConfigured { endpoints, domains }`.
- Tool-call JSON parsing has three modes (`ToolCallMode`): `Native`,
  `NativeWithRepair` (ONE deterministic repair pass on almost-valid JSON —
  repair happens once, never five times), `StructuredFallback` (extract
  from text); repeated identical failures trip the loop detector.
- **Loop detection** (threshold 3): the same normalized tool call
  (name + sorted args), same failure, or same patch key stops the turn
  (`loop_stopped`, lands `FailedRecoverable`) instead of repeating for 40
  turns.

---

## 11. Networking and streaming

### 11.1 Single async stack

- `kilop-core` is std-only (no tokio). Everything else uses one tokio
  runtime; the server is axum; `kilop-store`/`kilop-cas` are synchronous
  and heavy calls are wrapped in `tokio::task::spawn_blocking`.
- No bare `tokio::spawn` defines application state (§4); the SSE bus is
  polled lazily by connected streams — no background task, no unbounded
  lifetime.

### 11.2 State-aware retries and circuit breakers

- `RetryPolicy { max_attempts, base_delay_ms, max_delay_ms, jitter,
  class: RetryClass }` with `RetryClass { Network, RateLimited,
  ServerError, Always }`; `next_delay` adds jitter (full jitter),
  `should_retry` respects the attempt cap.
- The scheduler's execution loop combines retry-with-jitter with a
  **circuit breaker** per task name (`Closed → Open → HalfOpen` probe);
  budget-busy is a distinct `BudgetBusy` signal, never an error.

### 11.3 SSE surfaces

- **Per-session:** journal events projected to frozen `SseEvent` frames
  (`SessionUpdated, MessageCreated, MessagePartUpdated, ToolCallState,
  PermissionRequested, AgentStateChanged, AgentManagerUpdate,
  Compaction, Error`); the SSE `id:` cursor **is the event sequence**;
  resume = `events_after(seq)`.
- **Global:** `/global/event?after=<n>` serves the `GlobalEvent` envelope
  over a **bounded ring (4096 frames)**:
  `{ directory, project, workspace, payload }` where payload is a tagged
  union (`type` discriminator, snake_case): `session_created,
  session_turn_open, session_turn_close, session_queue_changed,
  background_process_updated, interactive_terminal_data,
  sandbox_status_changed, indexing_status, message_part_updated,
  session_next_text_delta, session_next_reasoning_delta,
  session_next_tool_called, session_state_changed, error`.
- Sessions project in **deterministic session-id order** so the global
  sequence is append-only; cursors make re-polling idempotent.
- **Coalescing:** consecutive text deltas run through `DeltaCoalescer`
  (50 ms window, 8 KiB per-frame cap); text for text_len-only chunk events
  is recovered by diffing stored message parts (`recover_text`); quiet
  tails are emitted one window after their last chunk.
- **Resume semantics:** `after` replays events with id > n; oversized
  cursors (e.g. u64::MAX) are clamped — the stream stays open, never an
  error. The UI connection is disposable: turns run detached from any SSE
  connection and resume from the journal.

---

## 12. Scheduler

`kilop-scheduler` schedules tool/subagent work as a **dependency DAG** of
`ScheduledOp` values — the whole `OpMeta` envelope travels with the op, so
the scheduler never builds a second identity:

```rust
pub struct ScheduledOp {
    pub meta: OpMeta,                                    // identity, deadline, retry, cancellation, recovery
    pub resources: ResourceRequest,                      // which ResourceClass budget it draws from
    pub reads: OwnershipSet,                             // files read (dependency overlap analysis)
    pub writes: OwnershipSet,                            // files written (overlapping writes serialize)
    pub dependencies: Vec<(OpId, DependencyPolicy)>,     // per-edge gate policy
    pub run: OpFn,
}
```

Each dependency edge carries a `DependencyPolicy`:

- `Success` (default) — the dependent runs only if the upstream ended
  `Done`; an upstream that `Failed`, was `Cancelled`, or was itself
  `Blocked` leaves the edge dead and the dependent can never run
  (`TaskStatus::Blocked`, propagated transitively).
- `Terminal` — the dependent runs after any terminal execution state
  (`Done | Failed | Cancelled`); a `Blocked` upstream does not satisfy it.
- `Always` — cleanup/finalizer edge: satisfied by any terminal upstream
  state, including `Blocked`.

- **Resource classes and budgets** (`ResourceClass`, 9 classes; default
  in-flight limits): `Model 1`, `DiskRead 16`, `DiskWrite 4`, `Cpu 2`,
  `Git 1` (serialized per repo in kilop-git), `Network 8`, `Terminal 2`,
  `Mcp 4`, `Indexing 1` (deliberately low — indexing yields). A class
  budget exists so e.g. embedding/indexing work can never starve an
  interactive session.
- **Event-driven scheduling, permit-before-spawn, no wave barriers.**
  `run_to_completion` validates the DAG, then drains a ready FIFO into a
  tokio `JoinSet`: a task starts only when every edge is satisfied AND it
  holds a resource permit (`try_acquire` before spawning); a task that
  finds the budget full returns `BudgetBusy` and cycles back to the
  ready-queue tail for the next drain — no artificial synchronization
  across rounds. Completion of any task immediately frees its permit and
  decrements its dependents, so a long task never gates short tasks that
  became ready after it.
- **Deadlock detection:** `validate()` rejects unknown dependencies and
  cycles before any run (`ErrorKind::Deadlock`); if every ready task is
  budget-blocked and nothing else is making progress, that is a
  configuration deadlock and errors loudly.
- **Ownership overlap:** independent reads/subagents run concurrently;
  edits touching overlapping `writes` ownership sets do not. A failed or
  blocked task does not deadlock the DAG — `Terminal`/`Always` edges let
  dependents run, and `Blocked` propagates instead of hanging.
- Every task honors deadline, cancellation, retry-with-jitter, and
  circuit breaking (`CircuitBreaker`: `Closed → Open → HalfOpen` probe);
  task status is observable via `status(id)`/`statuses()`.

---

## 13. Performance budgets (§37)

Numbers are enforced by tests as gates. `cargo test --workspace` runs
them; soak/perf runs are `#[ignore]`-gated (see §14).

| Budget | Value | Enforced by |
|---|---|---|
| Context total (small/local profile) | 32,000 tokens (5K+3K+7K+10K+5K+2K) | `context::budget` exact-math tests |
| Context usable before provider call | 25,000 tokens (context_max) | same |
| Compaction minimum reduction | 25% (`min_reduction_ratio` default 0.25) | `session::compaction`, `context::compactor` |
| Compaction convergence | must land ≤ hard_cap, never grow | compactor tests |
| Terminal output ring | 200 lines live; artifact spill ≥ 100 MiB | terminal tests |
| Tool result excerpt | truncated to 2000 chars on the wire | agent runtime |
| SSE delta coalescing | 50 ms window, 8 KiB/frame cap | server coalesce tests |
| Global event ring | 4096 frames replay capacity | server global bus |
| Message page | ≤ 200 messages per page, `has_more` | session paging tests |
| Prompt bound | 512 KiB (max 64 files, 4 KiB path each) | session bounds |
| Message / part / tool-args / ledger / artifact / verify caps | 1 MiB / 4 MiB / 4 MiB / 1 MiB / 64 MiB / 64 MiB | session bounds |
| Turn deadline | 24 h (`TURN_DEADLINE_MS`) | session |
| Scheduler parallelism gate | 32 tasks × 30 ms at DiskRead 16 completes < 400 ms | scheduler perf test |
| Loop detector threshold | 3 identical normalized calls | agent loop tests |

---

## 14. Testing philosophy

**Adversarial-only.** Every test attempts to break the system: crash
mid-operation, corrupt storage, truncate streams, race writers, malicious
payloads, oversized input, path traversal, illegal state transitions,
duplicate replay, out-of-order events. Happy-path-only tests are rejected
in review. The harness is adversarial too — `FakeProvider` scripts die
mid-stream, inject rate limits, and emit malformed tool calls; recovery
tests forge crash states (running `tool_run` rows, forced journal
entries) and verify idempotent, tamper-loud behavior.

Test crates (each an independent workspace member):

- `tests/integration` — end-to-end HTTP/SSE behavior against the frozen
  protocol.
- `tests/fault` — fault injection: crash, corruption, truncation, races.
- `tests/soak` — long-run stability (memory bounds, journal growth).
- `tests/performance` — perf gates (§13) — `[perf]`.
- `tests/visual` — pixel-level UI compatibility: zero-pixel-diff against
  `fixtures/screenshots` **outside branding masks** (`BrandingMask` +
  `compose_plus_overlay` for the "+" overlay region); fixtures regenerated
  by an ignored `[visual]` test.

Ignore conventions (AGENTS.md, verified by CI):

- Long tests are `#[ignore]`-gated and named with `[soak]`, `[perf]`,
  `[visual]`, or `[fault]` prefixes; they do not run in normal CI.
- `cargo test --workspace` runs everything else; each crate's unit tests
  must pass before the crate is considered done.

---

## 15. Build and migration strategy

**Build:** a single Rust workspace (`cargo build --workspace`); no
external runtime beyond the native binary and SQLite. Native targets:
macOS, Linux, Windows (process groups / Job Objects where the platform
differs).

**Migration stages A–K** (from the original spec) — the frozen UI is
served by a compatibility shell at every stage, so the product never
regresses:

| Stage | Deliverable |
|---|---|
| A. Freeze | lock v7.5.6 UI + protocol baselines, fixture corpus (`compat/kilo-v756/`) |
| B. Compat server | HTTP/SSE shell that passes the v756 suite byte-for-byte (startup line, password auth, GlobalEvent envelope) |
| C. Persistence | `kilop-store` + `kilop-cas` + the durable session state machine (journal, tool-run ledger, recovery) |
| D. Providers | `kilop-provider` hub + adapter families + registry + capability normalization |
| E. Tools | transactional edit, native snapshots, fs/git, terminal supervision, MCP/LSP, sandbox |
| F. Agent loop | the durable turn machine driven by `kilop-agent` on top of the session |
| G. Context | budgets, five memory classes, ledger, death-spiral-proof compaction |
| H. Indexing | hybrid index + rank-fused retrieval + structured memory |
| I. Snapshots | CAS checkpoints with rollback verification wired into edits |
| J. Agent manager | daemon-owned background agents (Agent Manager cards) |
| K. Legacy removal | delete the old backend once the v756 compatibility suite passes |

---

## 16. v7.5.6 wire compatibility surface (subset)

**Frozen. Changing wire behavior requires updating fixtures first.**

This section documents the **subset** of the v7.5.6 server contract this
daemon implements: the golden fixtures, the routes actually wired, and the
auth forms actually accepted. It is not the full extension contract.

- **Fixture corpus:** `compat/kilo-v756/` golden tests in
  `kilop-protocol` lock request/response/SSE/JSON-field-presence/null-
  behavior/error-code behavior byte-for-byte. Fixture files present:
  `startup_line.json`, `hello.json`, `create_session.json`,
  `messages_page.json`, `sse_frames.json`, `global_event.json`,
  `password_auth.json`, `basic_auth.json`, `errors.json`,
  `provider_list.json`, `wire_session_create.json`,
  `wire_message_send.json`, `wire_part_union.json`.
  `compat/jetbrains-712/` is reserved for the JetBrains split-mode corpus.
- **Startup line:** `kilo serve --port 0` prints exactly
  `kilo server listening on http://127.0.0.1:<port>` and **nothing else**
  to stdout (logging goes to stderr). The frozen client parses stdout for
  this line — there is no JSON handshake on stdout (the legacy handshake
  type is test-only and never printed). The password never appears on
  stdout.
- **Auth:** the frontend generates a 64-hex `KILO_SERVER_PASSWORD` and
  passes it via the environment. The frozen v7.5.6 extension authenticates
  **every** request — `/global/health` included — with
  `Authorization: Basic base64("kilo:" + password)`. The Kilo+-native
  forms `Authorization: Bearer <password>` and `x-kilo-server-password:
  <password>` remain accepted, and the legacy per-start `AuthToken` keeps
  old clients/tests working. `Bearer <password>` is **not** the only
  accepted claim; wrong/missing credentials → 401.
- **REST surface (exactly what is wired):**
  - SDK-shaped: `/session/create`, `/session/prompt`, `/session/abort`,
    `/session/messages`, `/session/state`, `/session/list`,
    `/permission/reply`, `/permission/list`, `/provider/list`,
    `/global/health`, `/global/event`, `/question/reply`,
    `/question/list`, `/network/reply`, `/network/list`, `/config/get`,
    `/config/set`.
  - Legacy aliases (kept wired; their tests must keep passing):
    `/api/hello`, `/api/session`, `/api/sessions`,
    `/api/session/{id}`, `/api/session/{id}/state`,
    `/api/session/{id}/messages`, `/api/session/{id}/events`,
    `/api/session/{id}/prompt`, `/api/session/{id}/abort`,
    `/api/perm/{id}/resolve`, `/api/provider`.
  - Wire surface: `/session`, `/session/{sessionID}`,
    `/session/{sessionID}/message`, `/session/{sessionID}/abort`,
    `/session/{sessionID}/diff`, `/session/{sessionID}/revert`,
    `/session/{sessionID}/unrevert`.
- **Strictness:** `deny_unknown_fields` on request bodies (unknown fields
  → 422); unknown sessions → 404; malformed ids/empty prompts → 400;
  oversized → 413-class `Oversized`; protocol drift is loud, never silent.
- **GlobalEvent envelope:** every global frame is
  `{ directory, project, workspace, payload }` with the payload's `type`
  as discriminator; SSE `event:` field is optional. Resume cursor =
  event id, replaying `id > after` from the bounded ring.
- **SseEvent frame contract:** `to_frame(seq)` / `from_frame` are golden-
  tested against `sse_frames.json`; interior chunk events without message
  context are skipped by the per-session projection (the state endpoints
  carry that information).
- **Error mapping:** `kilop_protocol::error::from_core` maps every
  `kilop-core` error to a frozen `ApiError { code, message, http_status,
  retryable }` (locked by `errors.json`).
