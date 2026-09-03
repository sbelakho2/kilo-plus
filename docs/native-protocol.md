# Faktor Native Protocol v1

The daemon's own HTTP/SSE surface (architecture spec §16). UI
compatibility is the target — visual/behavioral, not wire-level — and
this protocol is optimized around the Faktor runtime. The v7.5.6 wire
contract is optional migration/test glue against the old UI
(`compat/kilo-v756`, architecture §16); nothing here pretends to be it.

All endpoints require daemon auth (same `FAKTOR_SERVER_PASSWORD` /
`Authorization` forms as the rest of the server). JSON field names on the
native surface are camelCase unless noted. Unknown fields on request
bodies are ignored by native v1 handlers (they are first-class DTOs of
this runtime, not frozen wire shapes).

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

Designed; not yet wired (one-line semantics):

- `GET /session/{id}/messages?cursor=` — cursor-based message page
  (`seq > cursor`, newest first, `nextCursor`/`hasMore`).
- `GET /session/{id}/events?after=` — journal event frames with `seq >
  after` resume cursors (SSE `id:` = event seq; see §11.3 of the
  architecture spec).
- `GET /session/{id}/turns` — durable turn records (envelope, status,
  provider/model, timestamps).
- `GET /session/{id}/tasks` — the durable task ledger (goal, steps,
  decisions, changed files, failures) as typed JSON.
- `GET /session/{id}/checkpoints` — checkpoint rows (sequence, path,
  hashes, restore audit).
- `GET /session/{id}/verification` — tool runs owed verification
  (unknown external effects; effect-status audit).
- `GET /session/{id}/agents` — background agents owned by the session.
- `GET /session/{id}/terminal` — the session's supervised terminal
  (PTY) view.
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
