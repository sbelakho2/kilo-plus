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
```

## Frozen baselines

- **VS Code:** Kilo Code v7.5.6 UI (webview, CSS, images, message layout) —
  byte-for-byte fixture (TARGET — the upstream webview is not vendored).
  Later releases are never merged wholesale.
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
