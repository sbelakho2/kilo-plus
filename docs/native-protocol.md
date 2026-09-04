# Faktor Native Protocol v1

The daemon's own HTTP/SSE surface (architecture spec §16). UI
compatibility is the target — visual/behavioral, not wire-level — and
this protocol is optimized around the Faktor runtime. The v7.5.6 wire
contract is optional migration/test glue against the old UI
(`compat/kilo-v756`, architecture §16); nothing here pretends to be it.

All endpoints require daemon auth (same `FAKTOR_SERVER_PASSWORD` /
`Authorization` forms as the rest of the server). JSON field names on the
native surface are camelCase unless noted. Request bodies of the native
surface are first-class strict DTOs (`deny_unknown_fields`; audit 56):
an unknown field — a misspelled option such as `hardBudegt` included —
is a loud 400, never silently ignored. They are strict *native* shapes of
this runtime, not frozen v7.5.6 wire envelopes.

## Lifetimes

| Object | Lifetime | Identity | Notes |
|---|---|---|---|
| Session | Created → Open for days → Suspended/Closed | numeric `id` | Row + journal + queue survive crashes (§4–§6 of the architecture spec) |
| Turn | One logical turn per prompt admission; `active → completed/cancelled/failed` | `opId` + durable turn record | Exactly one `TurnCompleted` journal event per genuine end |
| Task | The durable structured task state of a session (`faktor-context` ledger): goal, steps, decisions, changed files | session-scoped JSON | Survives compaction; bounded by construction |
| Operation | Any async op (`OpMeta` envelope: operation_id, session_id, state, start_time, deadline, retry, cancellation, recovery) | `opId` | Tools and provider calls are sub-operations of a turn |
| Agent | A background agent session owned by the daemon (Agent Manager) | separate `sessionId` | Listed under `/session/{id}/agents` |

## Paging and cursors

- Message pages are **cursor-based**: `GET /session/{id}/messages?cursor=<seq>&limit=<n>`.
  The cursor is the exclusive lower message sequence of the page
  (`seq > cursor`), newest first; a page carries `nextCursor`
  (`null` = end of history) and `hasMore`. The client never derives
  ordering from offsets; inserts are append-only, so pages never shift.
- Event resumes are **journal-sequence cursors**: `after` is the last
  consumed event seq and replays `seq > after` (0 = from the beginning).
  The SSE `id:` field of every frame IS the event sequence, so a
  reconnect simply sends the last id received. Oversized/unknown cursors
  are clamped, never errors.
- Turn records, checkpoints, verifications and agent lists are bounded
  newest-first listings; paging over them uses the same `cursor/limit`
  convention where the store supports it, and full bounded listings
  otherwise.

## Liveness and readiness

- `GET /native/health` — liveness: 200 `{ok: true, version}` whenever the
  process responds (auth-gated like every route). "The process is up",
  nothing more; it exists so probes that must not flap on recovery do not
  have to distinguish readiness semantics.
- `GET /native/ready` — readiness: 200 `{ready: true}` only when the
  session store has **recovered** (the flag is set at the very end of
  `serve()` setup — the caller opens the store, applies migrations at open
  and runs crash recovery *before* serve, so the end of setup is exactly
  the recovered moment), the required runtime components exist (the
  `SessionManager` is a non-optional `Arc` in `ServerDeps`, so its
  presence is structural), and migrations are applied (implicit at store
  open). Before that moment the endpoint answers 503 `{ready: false}`;
  because the flag flips before `serve()` returns its handle, a client
  that holds a live handle observes only the ready state — the not-ready
  window is a startup property (`ServerDeps.simulate_not_ready` keeps it
  observable in tests). Liveness ≠ readiness: a daemon whose store failed
  recovery is alive but never ready.

## Endpoint list (v1)

Implemented (this revision of the daemon):

- `GET /session/{id}/projection` — one JSON snapshot of the session's
  state for UI badges/polling:
  `{ session: {id,title,provider,model,lifecycle}, state:
  {machine,label,active,terminal}, activeModel?, activeTool?, progress?,
  filesChanged: [...], lastCheckpoint?, verification: [...],
  contextUsage?, queued: n }`. `activeModel` = the effective
  provider/model envelope of the current or most recent logical turn
  (durable turn record; `null` before the first turn). `activeTool` =
  the newest still-running durable tool-run row. `filesChanged` = the
  durable task ledger's changed files. `lastCheckpoint` = newest
  checkpoint row when the daemon runs with a checkpoint service wired,
  else `null`. `verification` = pending tool runs whose effects are
  `unknown` (recovery `mark_unknown`), capped. `progress` and
  `contextUsage` are `null` in this revision (no numeric progress
  channel, and no durable usage read API yet); the machine state and
  provider-call journal are the source of phase information.
- `GET /models` — the flat daemon model catalog:
  `[{provider, model, context, maxOutput, tools, parallelTools,
  reasoning, thinking, vision, structuredOutput, embeddings, streaming,
  source}]`, walking every registered provider instance × its
  `known_models()` × `capabilities(model)`. `source` is the provenance
  string: `"liveProbe"` when the provider reports a live runtime context
  limit for the model (e.g. an Ollama `/api/ps` allocation),
  `"providerCatalog"` when the entry carries a non-default capability
  profile (configured or probed), else `"conservativeDefault"` (the
  fail-safe default profile).
- `GET /capabilities` — introspection map for capability-driven UI:
  `{ "<provider>": { models: [{id, capabilities}],
  runtimeContextLimitSupported: bool } }` (same registry walk; the
  boolean is true when any known model of the provider reports a live
  runtime limit).
- `GET /native/health` / `GET /native/ready` — see "Liveness and
  readiness" above.
- `GET /native/usage` — cross-session aggregate of the durable
  context-usage facts the runtime records (memory facts kind `usage`,
  keys `budget`/`spent`, integer values): `{sessions, totals:
  {budget, spent}, perSession: [{sessionId, budget, spent}]}`.
  `totals` sum every numeric fact across sessions; `perSession` lists
  only sessions carrying usage facts (non-numeric hostile values are
  skipped). No runtime path writes those facts yet, so today the totals
  are honest zeros and the list is empty — the aggregate shape is frozen
  for the UI; a future audit wires the writer.
- `GET /native/session/{id}/turns` — the session's durable turn records,
  newest first: `[{opId, status, provider, model, variant?, toolMode?,
  startedAt, updatedMs, queueSeq?, promptMessageId?}]` (`status` =
  `active|completed|cancelled|failed`). One record per admitted logical
  turn; empty before the first turn.
- `GET /native/session/{id}/tasks` — the durable task ledger as typed
  JSON: `[{goal, constraints, state, milestones: {completed, open},
  decisions, failures, changedFiles, tests: {run, failed}, preferences,
  verification}]`. One entry while a task is tracked, `[]` before any
  task data exists (a stored-but-empty ledger is not a task). `state` is
  derived: `running` while the turn machine is active, `in_progress`
  with open milestones (or a fresh goal with nothing completed yet),
  `done` once completed work exists with nothing left open, else `idle`.
  `verification` repeats the session's durable verification facts (same
  source as `/verification` below).
- `GET /native/session/{id}/checkpoints` — the session's durable
  checkpoint rows, newest first:
  `[{sequence, path, beforeHash, afterHash, beforeExists, afterExists,
  createdMs, restoredMs?}]`. Empty when the daemon runs without a
  checkpoint service wired or nothing was recorded yet.
- `GET /native/session/{id}/verification` — everything the session owes
  verification: `{owed: [{opId, tool, startedMs, status, effectStatus}],
  failedChecks: [{id, detail, status}]}`. `owed` = still-open durable
  tool runs whose recovery strategy is `mark_unknown` (unknown external
  effects are forced to verification — spec §7); `failedChecks` = the
  durable memory facts of kind `verification` (one per failed REQUIRED
  check, recorded at genuine turn ends; `detail` carries
  `failed:<command>`). Bounded; empty arrays when nothing is owed.
- `GET /native/session/{id}/agents` — background agents owned by the
  session. Always `[]` in this revision: child sessions appear when
  orchestration (Agent Manager subagent sessions) lands in the runtime.
  The route exists so the UI can poll the shape now.
- `GET /native/session/{id}/terminal` — the session's terminal view:
  `[{id, pid, alive}]`. Live PTYs have no durable session binding yet, so
  every live PTY of the daemon is listed (session-scoped ownership is the
  next wiring step); the path session id is still validated (unknown →
  404).
- `POST /native/session/{id}/abort` — the native abort
  (`sdk_abort` semantics behind the strict DTO): body
  `{"session_id": <id>, "op_id": <opId>?}` (`op_id` targets one queued
  prompt or the active turn; absent = abort everything of the session).
  The body `session_id` must equal the path id. Unknown fields/typos in
  the body are 400; unknown sessions 404; a queued-prompt kill cancels
  its durable row without touching the state machine. Response
  `{aborted: [<opId>...]}`.

Designed; not yet wired (one-line semantics):

- `GET /session/{id}/messages?cursor=` — cursor-based message page
  (`seq > cursor`, newest first, `nextCursor`/`hasMore`).
- `GET /session/{id}/events?after=` — journal event frames with `seq >
  after` resume cursors (SSE `id:` = event seq; see §11.3 of the
  architecture spec).
- `GET /providers` — provider instances with their auth/endpoint
  metadata (never secrets).

## UI-adaptation principle

The old UI is a UI, not a protocol peer. Adaptation happens at the
boundary, in one direction only:

1. **UI posts typed messages** — text, tool parts, file references —
   never v7.5.6 control envelopes. The IDE shells (`apps/vscode`,
   `apps/jetbrains`) and any bridge translate UI gestures into typed
   native requests.
2. **Bridge → Rust**: the bridge is a thin client of the native
   protocol; all state lives in the daemon, all validation happens in
   the daemon.
3. **Never pretend to be v7.5.6**: when an old-UI client speaks the old
   wire contract, it hits the optional compat glue (`compat/kilo-v756`)
   — a deliberately separate, fixture-locked surface. A native client
   never fabricates v7.5.6 frames, and native endpoints never inherit
   v7.5.6 DTO strictness quirks (unknown-field rejection, snake_case
   envelopes) that exist only for the frozen glue.
