# Faktor v7.5.6 wire compatibility manifest

**v7.5.6 behavioral compatibility scaffold — NOT an exact v7.5.6 wire
implementation (real SDK fixtures BLOCKED_EXTERNAL).**

Frozen client SDK operations vs. the daemon surface implemented in
`crates/server/src/api.rs`, as of the P0 "v7.5.6 protocol is still not the
v7.5.6 protocol" round. Every row is auth-required (the frozen Basic
`base64("kilo:"+password)` form, plus the legacy Bearer /
`x-kilo-server-password` forms). Wire ids are numeric strings; wire
message ids on this surface are the durable message SEQUENCE (the same
identity `revert`/`diff`/`deleteMessage` consume; on a single-session
store the sequence equals the row id).

## Corrected shapes (the three named mismatches)

### 1. `GET /session/{sessionID}/message?before=&limit=` — page

Old: `{sessionID, messages: WireMessage[], hasMore}`.
New: a bare JSON **array** of `{info: Message, parts: Part[]}`,
newest first (`before`/`limit` paging unchanged; the wire omits `seq`):

```json
[
  {
    "info": {
      "sessionID": "1",
      "messageID": "3",
      "role": "assistant",
      "createdMs": 1750000002000,
      "providerID": "ollama",
      "modelID": "qwen3.8"
    },
    "parts": [ { "type": "text", "text": "pong" } ]
  },
  {
    "info": { "sessionID": "1", "messageID": "2", "role": "user",
              "createdMs": 1750000001000,
              "providerID": "ollama", "modelID": "qwen3.8" },
    "parts": [ { "type": "text", "text": "fix it" } ]
  }
]
```

Paging signal: `x-has-more: true|false` response header (the frozen entry
type rejects unknown fields, so paging cannot ride the DTO).
Prompt (user) messages themselves appear with their parts: user rows are
stored as `{text, files}` message data, projected as their wire text part.

### 2. `POST /session/{sessionID}/message` — send (response)

Old: `{messageID, accepted, queued}`.
New: `{info: AssistantMessage, parts: Part[]}` where `info` is the durable
assistant message row the accepted turn produced and `parts` its wire
parts (turn runs to completion inside the request; SSE carries progress):

```json
{
  "info": {
    "sessionID": "1",
    "messageID": "3",
    "role": "assistant",
    "createdMs": 1750000002000,
    "providerID": "ollama",
    "modelID": "qwen3.8"
  },
  "parts": [ { "type": "text", "text": "pong" } ]
}
```

Queueing semantics (documented choice): a prompt that durably queues
behind an active logical turn answers **HTTP 202 Accepted** with the same
shape — `parts: []` and `info.messageID: ""` (nothing is materialized
until the queued turn starts; the client polls the page / SSE). The 202
status IS the queueing signal: the frozen `{info, parts}` type has
`deny_unknown_fields`, so a `queued` field inside the DTO would be
protocol drift. A turn that ends without any assistant content (provider
failure before the first chunk) is an honest `502 {ok:false, message}`,
never a fabricated message.

### 3. `GET /session/{sessionID}/diff?message=&file=&full=1` — file diffs

Old: custom `{diff, path, status}` single object (or nulls).
New: a bare JSON **array** of `SnapshotFileDiff[]`, one entry per recorded
checkpoint (file-change) row, newest first. Status is the recorded
before→after transition (`added` | `deleted` | `modified`). Without
`full=1` entries carry only `path`+`status`:

```json
[
  { "path": "f.txt", "status": "deleted" },
  { "path": "created-empty.txt", "status": "added" },
  { "path": "f.txt", "status": "modified" }
]
```

With `?full=1` each entry also carries the unified diff text (`diff`),
resolved through the CAS (pre-after-blob rows are refused honestly with
409, exactly like the snapshot `diff_latest`):

```json
[
  { "path": "f.txt", "status": "modified",
    "diff": " line1\n line2\n-old\n+new\n line6" }
]
```

Filters: `?message=<seq>` limits the projection to ONE checkpoint — the
newest checkpoint recorded at-or-before that message's `created_ms`
(unknown message → honest 409). `?file=<rel path>` keeps only the entries
whose recorded path equals the relative path (exact match; no filesystem
access). No checkpoints → `[]`.

## Operation manifest

| operation | route | method | shape | status |
|---|---|---|---|---|
| session.create | `/session` | POST | `{sessionID,title,createdMs}` | implemented, tested |
| session.list | `/session` | GET | `{sessions:[SessionSummary]}` | implemented, tested |
| session.get | `/session/{sessionID}` | GET | `SessionSummary` | implemented, tested |
| session.update | `/session/{sessionID}` | POST | `{sessionID,title,updatedMs}` — title is the one durable session-row field the daemon owns: control chars stripped, bounded 1..=200 chars, persisted through the session layer (store row + bumped `updatedMs`); 400 malformed (no title), 404 unknown session | implemented, tested |
| session.status | `/session/{sessionID}/status`, `/session/status?session_id=` | GET | `SessionState` projection | implemented (aliases of `/state`) |
| session.fork | `/session/{sessionID}/fork` | POST | `{sessionID,title,createdMs}` (`<title> (fork)`) | implemented, tested (rows+parts copied in order; fork independent) |
| session.summarize | `/session/{sessionID}/summarize` | POST | `{sessionID,title,summary}` (bounded 4 KiB digest of newest messages) | implemented, tested |
| session.delete | `/session/{sessionID}` | DELETE | `{ok:true}` | implemented, tested; refused 409 mid-turn (active turn record / active machine); durable end = `SessionEnded` journal + `lifecycle=Closed`; lingering queued prompts cancelled; in-process registries closed. Residual gap: rows are retained (the store has no row-drop API in this slice); a deleted session reads as Completed/Closed and refuses prompts |
| session.deleteMessage | `/session/{sessionID}/message/{messageID}` | DELETE | `{ok:true}` — durable one-transaction removal of the message row + its parts (message identity = the durable sequence); sequences stay stable (paging skips the hole, never renumbers); 404 unknown message; 409 tool-result dependencies (result part, or a call part a result references); 409 while it is the active turn's in-flight newest message | implemented, tested |
| session.message (page) | `/session/{sessionID}/message` | GET | corrected array (above) | implemented, tested |
| session.message (send) | `/session/{sessionID}/message` | POST | corrected `{info,parts}` (above) | implemented, tested (200 done, 202 queued, 502 no-reply) |
| session.abort | `/session/{sessionID}/abort` | POST | `{aborted:[opId]}` | implemented, tested |
| session.diff | `/session/{sessionID}/diff` | GET | corrected `SnapshotFileDiff[]` (above) | implemented, tested |
| session.revert / unrevert | `/session/{sessionID}/revert`, `/unrevert` | POST | `{ok,restored?,conflict?}` / `{ok:false,message}` | implemented, tested |
| session.state | `/session/{sessionID}/state` | GET | `SessionState` | implemented, tested |
| permission.list | `/permission/list?session_id=` | GET | `{permissions:[{id,session_id,capability,detail}]}` | implemented, tested (real pending requests) |
| permission.reply | `/permission/reply`, `/api/perm/{id}/resolve` | POST | `{ok:true}` | implemented, tested |
| question.list | `/question/list?session_id=` | GET | `{questions:[...]}` over pending non-network permissions | implemented, tested |
| question.reply | `/question/reply` | POST | `{ok:true}` (decision allow/deny) | implemented, tested (over the permission requester) |
| question.reject | `/question/reject` | POST | `{ok:true}` (deny) | implemented, tested |
| network.list | `/network/list?session_id=` | GET | `{networks:[...]}` over pending `network`-capability permissions | implemented, tested |
| network.reply | `/network/reply` | POST | `{ok:true}` (decision allow/deny) | implemented, tested |
| network.reject | `/network/reject` | POST | `{ok:true}` (deny) | implemented, tested |
| config.get | `/config/get` | GET | `{config}` (daemon config RwLock) | implemented, tested |
| config.update | `/config/update` | POST | `{ok:true}` — only `model`/`compact_at_usage`/`instructions` are daemon-editable; any other key → 400 with the allowlist | implemented, tested |
| config.overlay | `/config/overlay` | POST | `{ok:true}` (bounded full replace of the config view) | implemented, tested |
| config.overlayUpdate | `/config/overlayUpdate` | POST | `{ok:true}` (bounded shallow merge) | implemented, tested |
| config.warnings | `/config/warnings` | GET | `{warnings:[...]}` real validation over the stored config | implemented, tested |
| config.set | `/config/set` | POST | `{ok:true}` (legacy full replace, kept) | implemented, tested |
| pty.create/update/remove | `/pty/create`, `/pty/update`, `/pty/remove` | POST | `409 {ok:false, message:"ptys unsupported by the local supervisor"}` | implemented as explicit rejection, tested (the supervisor has no PTY abstraction; never a fake success, never a hang) |
| global.dispose | `/global/dispose` | POST | `{ok:true}` | implemented, tested (abort + durable end of every session; idempotent) |
| instance.dispose | `/instance/dispose` | POST | `{ok:true}` | implemented (same handler), tested |
| instance.reload | `/instance/reload` | POST | `{ok:true}` after re-running daemon recover() | implemented, tested |
| auth.set | `/auth/set` | POST | `{ok:true,password}` — rotates the server password (`password` absent → fresh random); old credentials 401 immediately | implemented, tested |
| auth.remove | `/auth/remove` | POST | `{ok:true}` — back to the startup env password | implemented, tested |

## Residual gaps (need backend subsystems outside this round's allowed files)

1. **session.delete row removal** — delete durably ends the session
   (journaled `SessionEnded` + lifecycle Closed, queued prompts cancelled,
   registries closed) but retains the row: a store-level `remove_session`
   SQL does not exist in this slice, so a deleted session reads as
   Completed/Closed and refuses prompts.
2. **pty** — needs a real PTY abstraction over `faktor-terminal`
   (interactive handle + incremental reads); the supervisor only spawns
   non-interactive children, so the ops reject explicitly.
3. **config persistence** — the config view is the daemon's in-memory
   RwLock (fresh daemon = `{}`); persisting it to disk is a separate
   subsystem.
4. **message-id identity** — wire message ids are sequences; a row-id
   ↔ seq mapping on multi-session stores requires a store-level lookup by
   row id.
