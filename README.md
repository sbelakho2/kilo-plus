# Faktor

**Same Kilo Code UX. A substantially better native engine.**

Faktor replaces the Kilo Code engine (TypeScript/Bun) with a native Rust
runtime while targeting the frozen Kilo v7.5.6 IDE UX (TARGET baseline — the
actual upstream webviews are not vendored in this repo; `apps/` holds launcher
scaffolds and wire-level harnesses that will host them).

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
- **Process supervision** — no orphans. Process groups on Unix; Windows uses
  taskkill /T /F today (Job Objects are a documented TARGET),
  Windows, deliberate ownership transfer.
- **Provider normalization** — ~10 transport families + dynamic model registry.
  No `if provider == "deepseek"` anywhere in the agent.

## Layout

```
apps/        (TARGET: frozen v7.5.6 VS Code webview + JetBrains 7.1.2 Kotlin shell — currently launcher scaffolds + compatibility harnesses)
crates/      (the Rust engine workspace)
compat/      (permanent protocol fixtures: kilo-v756/, jetbrains-712/)
fixtures/    (protocol, providers, screenshots, repositories)
tests/       (integration, soak, fault, visual, performance — adversarial only)
ui/          (vendored frozen upstream UI: kilo-v756-webview/, kilo-ui/, pinned manifest)
```

## Frozen baselines

- **VS Code:** Kilo Code v7.5.6 UI (webview, CSS, images, message layout) —
  byte-for-byte fixture; the upstream trees are vendored under `ui/` at
  commit `fa02955` with a SHA-256 manifest (`ui/upstream.json`,
  `scripts/verify-upstream.mjs`), and `apps/vscode/src/kilo-bridge.ts`
  translates native state onto the frozen message ABI. A built bundle
  (`ui/kilo-v756-webview/dist/`) is loaded when present; otherwise the
  built-in Faktor chat panel is the fallback. Later releases are never
  merged wholesale.
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

## Branding

All user-visible metadata in this repository uses Faktor branding; legacy
wordmark tokens survive only inside frozen compatibility fixtures and
attribution prose (enforced by `scripts/branding-scan.sh`). The external
GitHub repository name and description cannot be changed from this
repository — rename them in the repository settings; package/manifest
metadata in-tree is the authoritative surface and is scan-enforced. The
external repository has since been renamed to `faktor` via `gh repo rename`;
in-tree metadata is unchanged.

## Status notes

- **Completion contract (P2 follow-up).** The reviewed PR/CI-fix item that
  lets a native task declare `completion_contract` (`include_commit`,
  `include_push`, `include_pr`) and gates `VerifiedComplete` on the
  corresponding steps is specified but **not implemented** in this tree; the
  normative semantics, the exact code seams, and why the current change's
  allowed files block them are recorded in `docs/certification.md` §3.3.
