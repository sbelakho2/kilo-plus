# Kilo+ repository agent guide

This workspace is a Rust implementation of the Kilo+ architecture spec
(`docs/architecture.md`). Follow these rules:

## Commandments

1. **Dependencies point inward toward pure core types.** `kilop-core` has no
   workspace dependencies. Provider code never touches session persistence.
   Tools never mutate session state directly. Everything enters the runtime
   through commands/events.
2. **Every async operation is explicit.** No bare `tokio::spawn` that defines
   application state. Operations carry `operation_id, session_id, state,
   start_time, deadline, retry_policy, cancellation_token, recovery_strategy`.
3. **Adversarial-only testing.** Every test must attempt to break the system:
   crash mid-operation, corrupt storage, truncate streams, race writers,
   malicious payloads, oversized input, path traversal, illegal state
   transitions, duplicate replay, out-of-order events. Happy-path-only tests
   are rejected in review.
4. **No `if provider == "..."` in the agent.** Provider behavior is decided by
   `ModelCapabilities` and provider profiles. Provider quirks stay inside
   adapters.
5. **Bounded everything.** No unlimited command output in RAM or context (ring
   buffer + compressed artifact). No unbounded transcripts (paging is
   fundamental). No unbounded resource lifetimes (explicit scopes, idle
   unload).
6. **Never blindly re-run a command after a crash.** Unfinished operations are
   reconstructed from durable state; deterministic FS ops verify the expected
   resulting hash; unknown external effects are marked `effect_status =
   unknown` and forced to verification.
7. **Compaction hard invariant.** A successful compaction must achieve the
   configured minimum reduction. Reject summaries that reduce context by ~1%.
8. **Zero orphans.** Every child process has a runtime owner. If the session
   dies, ownership transfers deliberately or the process dies.
9. **Wire compatibility is a frozen contract.** `kilop-protocol::v756` golden
   tests lock request/response/SSE/JSON-field-presence/null-behavior/error-code
   behavior. Do not change it without updating the fixtures in
   `compat/kilo-v756/`.

## Verification

CI (`.github/workflows/ci.yml`, on both ubuntu-latest and macos-latest)
runs exactly these commands; they must pass locally before pushing:

- `cargo fmt --check`
- `cargo check --workspace`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo run -p kilop-cli -- doctor --data-dir /tmp/kp-ci`

- Each crate's unit tests must pass before the crate is considered done.
- Long tests are `#[ignore]`-gated and named with `[soak]`, `[perf]`,
  `[visual]`, or `[fault]` prefixes; they are excluded from the CI test
  run and executed manually.

## Workspace identity

Every file/tool call explicitly carries its `WorkspaceId`/`WorktreeId`/`TaskId`.
There is no global mutable "current directory" anywhere in the runtime.
