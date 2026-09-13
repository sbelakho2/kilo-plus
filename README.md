# Faktor

**Same Kilo Code UX. A substantially better native engine.**

Faktor replaces the Kilo Code engine (TypeScript/Bun) with a native Rust
runtime while targeting the frozen Kilo v7.5.6 IDE UX. The v7.5.6 VS Code
webview sources are vendored under `ui/` (pinned at commit `fa02955` with a
SHA-256 manifest) and served through `apps/vscode`; the pinned closure plus
the Faktor companion overlay are staged into the extension's `media/` and
ship inside the VSIX. The JetBrains 7.1.2 Kotlin shell is NOT vendored —
`apps/jetbrains` carries the real Faktor-owned frontend/backend panels
(task tree, blockers, tournament, board, evidence, attachments) that talk
to the native daemon.

```
same UI
   ↓
small compatibility shell
   ↓
native Rust engineering runtime
   ↓
LLM used only where reasoning is actually needed
```

## Architecture at a glance

- **Durable state machine** — every session is an explicit state machine fed by
  an append-only event journal. No `await Promise` implicitly defines
  application state. On daemon restart, unfinished operations are reconstructed
  from durable state.
- **Bounded context** — five separate memory classes (immutable instructions,
  durable task state, repository knowledge, recent conversation, historical
  artifacts). Compaction cannot enter a death spiral: a successful compaction
  must achieve a configured minimum reduction or it is rejected.
- **Native checkpoints** — content-addressed (BLAKE3 + Zstd) snapshot store
  instead of Git repositories pretending to be undo history. Git stays for
  branches/commits/worktrees/diffs only.
- **Transactional editing** — every agent edit is optimistic and versioned
  against `expected_hash`; parse-before-accept; atomic writes; no old patch
  applied to unexpected contents.
- **Hybrid retrieval** — exact + lexical + symbol + optional semantic search
  fused by rank; automatic retrieval is a TARGET (the machinery exists; the
  production daemon wiring is in progress).
- **Explicit concurrency** — resource-class budgets, dependency DAG scheduling,
  state-aware retries with jitter, circuit breakers.
- **Process supervision** — no orphans. Process groups on Unix; Windows
  creates a kill-on-close Job Object per supervised child
  (`CreateJobObjectW` + `AssignProcessToJobObject` +
  `SetInformationJobObject` in `crates/winjob`, wired through
  `faktor-terminal`) with `taskkill /T` as the escalation path.
- **Provider normalization** — ~10 transport families + dynamic model registry.
  No `if provider == "deepseek"` anywhere in the agent.

## Layout

```
apps/        (real Faktor IDE panels: the VS Code extension host/chat/cockpit and the JetBrains split-mode frontend/backend — task tree, blockers, tournament, board, evidence; no upstream JetBrains sources)
crates/      (the Rust engine workspace, incl. winjob/agent/verify/sandbox/index/cas/snapshot)
compat/      (permanent protocol fixtures: kilo-v756/, jetbrains-712/)
fixtures/    (protocol, providers, screenshots, repositories)
tests/       (integration, soak, fault, visual, performance — adversarial only)
ui/          (vendored frozen upstream UI: kilo-v756-webview/ + kilo-ui/, pinned manifest)
```

## Frozen baselines

- **VS Code:** Kilo Code v7.5.6 UI (webview, CSS, images, message layout) —
  byte-for-byte fixture; the upstream trees are vendored under `ui/` at
  commit `fa02955` with a SHA-256 manifest (`ui/upstream.json`,
  `scripts/verify-upstream.mjs`), and `apps/vscode/src/kilo-bridge.ts`
  translates native state onto the frozen message ABI. `npm run
  prepackage:vsix` stages the verified pinned closure plus the additive
  Faktor companion overlay at `media/kilo-v756-webview/`, so the packaged
  VSIX ships the vendored UI self-contained (hash-asserted at package
  time); without a staged bundle the built-in Faktor chat panel is the
  fallback. Later releases are never merged wholesale.
- **JetBrains:** JetBrains 7.1.2 (Kotlin frontend stays; process manager is
  modified only to launch the Faktor binary).
- **Protocol (TARGET):** the real v7.5.6 contract is the compatibility
  destination; `compat/kilo-v756/` currently holds a hand-written wire
  surface (subset) labelled as such. The Rust daemon must pass the real
  contract before the old backend is removed.

## Building

```bash
cargo build --workspace
cargo test --workspace
```

## Running

```bash
cargo run -p faktor-cli -- serve --port 0
cargo run -p faktor-cli -- run --data-dir /tmp/kp-demo "explain this repo"
cargo run -p faktor-cli -- doctor
```

## CI

CI runs on [Woodpecker](https://woodpecker-ci.org) from four event-scoped
workflows in `.woodpecker/` (targeted at Woodpecker 3.x; the folder takes
precedence over a root `.woodpecker.yml`, and this repository has none):

- **`pr.yaml`** (`pull_request`, reduced lane set, **no named volumes at
  all**) — `linux` (fmt, clippy `-D warnings`, `check`/`test --workspace
  --all-features`, doctor), `static`, `docs`, `vscode` (npm build + offline
  selftest + self-contained VSIX verify), `vscode-visual` (required chromium
  render gate) with its diagnostics twin, `jetbrains-build` and
  `jetbrains-smoke`, then the aggregate `certificate`. A `storage-policy`
  step fails closed if the PR workflow ever declares volumes.
- **`trusted.yaml`** (`push`/`tag`, darwin/windows on `push` to `main`) — the
  full lane set including the release `[perf]` lane and the darwin/windows
  matrix combos, using the trusted named-volume caches (`faktor-trusted-*`).
- **`nightly.yaml`** / **`soak.yaml`** (`cron` jobs `nightly`/`soak`) —
  `[fault]` campaigns at scale, longrun, efficiency, economy, coding-benchmark
  smoke (provider-key runs are recorded as explicit skips, never silent) and
  supply-chain evidence; and the `[soak]` 12h synthetic + 24h zero-drift
  wall-clock suites. Registration and the required timeout cap are in
  `scripts/woodpecker/setup.md`; `scripts/woodpecker/activate.sh` does it via
  the Woodpecker API.

The `certificate` job is the aggregate gate in every workflow: every lane
writes `target/certification/lanes/<lane>.json` into the shared workflow
workspace, and the certificate (depending on all lanes, running on `success`
or `failure`) fails when a marker is missing/failed/stale/unexpected or the
workflow status is not success; darwin/windows carry per-platform
certificates inside `trusted.yaml`. Woodpecker reports one commit status per
workflow, so branch protection requires `ci/woodpecker/pr/pr`. No secrets are
required; the named cache volumes need the repository to be marked trusted by
a server admin (see `scripts/woodpecker/setup.md` for the PR-vs-trusted
storage policy and its residual risks).

The offline per-host certificate is unchanged: `bash scripts/certify-local.sh
fast` (or `full`) emits `target/certification/manifest.json`; see
`docs/certification.md`.

## Branding

All user-visible metadata in this repository uses Faktor branding; legacy
wordmark tokens survive only inside frozen compatibility fixtures and
attribution prose (enforced by `scripts/branding-scan.sh`). The external
GitHub repository name and description are not tracked in-tree; they were set
with `gh repo rename` (name `faktor`) and
`gh repo edit --description "Faktor — native Rust coding-agent runtime with
Kilo-compatible IDE UX"`. In-tree package/manifest metadata remains the
authoritative surface and is scan-enforced.

## Status notes

- **Completion contract (gate + step execution IMPLEMENTED).** The reviewed
  PR/CI-fix item that lets a native task declare `completion_contract`
  (`include_commit`, `include_push`, `include_pr`) is implemented end to end:
  the DTO parses strictly, the accepted contract and every per-step outcome
  are durable ledger rows (immutable per task revision, pinned across
  compaction), `VerifiedComplete` is refused with a typed error while any
  requested step lacks a succeeding status row, and
  `crates/orchestrator/src/completion_steps.rs` executes the requested steps
  in gate order (idempotent, policy-checked push, supervisor-issued PR
  command). Both IDEs expose the Task-mode checkboxes and report the durable
  step provenance; a serving daemon without the additive completion read
  shows `unavailable`, never a fabricated success. Normative semantics and
  the exact seams are in `docs/certification.md` §3.3.

| Surface | Status | Evidence |
| --- | --- | --- |
| PR/CI-fix completion contract | IMPLEMENTED (gate + ordered step execution) | `crates/session/src/task.rs`, `crates/session/src/ledger.rs`, `crates/orchestrator/src/task_executor.rs`, `crates/orchestrator/src/completion_steps.rs`, §3.3 |
| Coordination board | IMPLEMENTED (durable ledger rows + native `GET/POST /native/session/{id}/board` + both IDE board panels with truthful unavailable state) | `crates/session/src/board.rs`, `crates/server/src/native/board.rs`, `apps/vscode/src/nativeClient.ts`, `apps/jetbrains/frontend/src/main/kotlin/dev/faktor/frontend/BoardPanel.kt` |
| Multi-candidate tournament | IMPLEMENTED | `crates/orchestrator/src/tournament.rs` + native start/state/list endpoints; integration stays an explicit approved merge |
| Pixel agents | IMPLEMENTED | `apps/vscode/src/pixelAgents.ts` + JetBrains `PixelAgents.kt` (identical FNV-1a hashes) |
| Canonical child blockers | IMPLEMENTED | `crates/session/src/child.rs` (`child_runtime` v23 row) + native agent projection |
| Presentation continuity | IMPLEMENTED | durable `child_presentation_changed` fold + `POST /native/session/{id}/agents/{child}/presentation` (presentation only; same ChildId/lineage) |
| Repo rename (faktor) | DONE (external) | `gh repo rename`; in-tree branding was already Faktor and is unchanged |
