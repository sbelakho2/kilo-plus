# Faktor executable coding benchmark (`faktor-tests-coding-benchmark`)

Executable benchmark (audit follow-up P0-87/88/83): the economy tests
score *synthetic* success probabilities; this crate scores *real* pinned
mini repositories across languages, driven through the **actual daemon**
(the built `faktor-cli serve` binary, native HTTP API — never the
RouterService), with immutable expected criteria, repository-native
verification, and cost-to-verified-success accounting.

Normal `cargo test` mode is a **fake, in-process, no-model harness**: it
never spawns the daemon and never contacts a provider. Real-model runs are
`#[ignore]`-gated and need provider keys.

## Corpus (`corpus/`)

Checked-in mini repositories, one per language. Every repo contains a
deliberately introduced bug, the passing test suite the bug breaks
(pristine `verify.sh` MUST fail; fixed MUST pass — both directions are
proved by tests), and three metadata files:

| task id | lang | repo | planted bug | toolchain |
|---|---|---|---|---|
| `rust-reverse-words` | Rust | zero-dependency crate (`cargo test --offline`) | reverses letters instead of word order | cargo |
| `c-ringbuf` | C | plain `cc` + Makefile + assertion main | full-buffer push overwrites newest, never evicts oldest | cc, make |
| `python-dedup` | Python | stdlib-only module + unittest suite (pytest when present, else `python3 -m unittest`) | removes only consecutive duplicates | python3 |
| `go-sumranges` | Go | zero-dependency module (`go test ./...`) | range end summed exclusively | go |
| `ts-clamp-sum` | TypeScript | real `.ts` sources run by `node --test` with node's type stripping (no npm) | lower clamp folds to `hi` | node ≥ 22.6 |
| `java-leap-years` | Java | two files, `javac` + main-based runner (no JUnit) | century rule missing | javac |

`task.md` = the immutable issue text (the prompt). `criteria.md` =
immutable expected criteria, deterministic format `- crit-NN: <text>`.
`verify.sh` = the repository-native verification command the benchmark
runs in a COPY of the workspace.

A task whose toolchain is missing on the runner is a **documented skip,
never a failure** (each `#[test]` prints the reason).

## Runner modes

**Fake / no-model (normal tests, `tests/smoke.rs`).** Exercises the whole
harness deterministically in seconds: corpus loading + adversarial
rejection, workspace copies, `verify.sh` on the pristine repos (must
FAIL — proves harness + scorer), toolchain skip gates, deterministic
criteria scoring, cost aggregation, corpus immutability, verify
timeout/cap/process-group kill.

**Real daemon (`tests/real.rs`, `#[ignore]`).** One task run =
fresh `faktor-cli serve` daemon (own temp data dir, config generated with
the pinned provider/model, `FAKTOR_SERVER_PASSWORD` auth) + fresh session
whose durable workspace root is a temp copy of the task repo + the
`task.md` prompt (plus the immutable criteria summary contract) through
the daemon's own prompt path. The harness plays the UI permission
channel (auto-allow pending permissions/questions; the daemon sandbox
policy still gates every command). Completion is polled from the durable
native turn records, bounded by `FAKTOR_BENCH_TASK_TIMEOUT_S` (default
600 s); on timeout the run is aborted via `POST /native/session/{id}/abort`.
Then the harness runs the task's `verify.sh` in the workspace copy.

Run one:

```sh
cargo build -p faktor-cli            # daemon binary (or FAKTOR_BENCH_BIN=...)
FAKTOR_BENCH_PROVIDER=anthropic \
FAKTOR_BENCH_MODEL=<model-id> \
FAKTOR_BENCH_API_KEY=<key> \
cargo test -p faktor-tests-coding-benchmark --test real -- --ignored --nocapture
```

Env: `FAKTOR_BENCH_PROVIDER` (`anthropic|openai|google|deepseek|gateway|ollama`),
`FAKTOR_BENCH_MODEL`, `FAKTOR_BENCH_API_KEY` (or `FAKTOR_BENCH_API_KEY_ENV`
to name a different env var; Ollama is keyless), `FAKTOR_BENCH_BIN`,
`FAKTOR_BENCH_BASE_URL` (openai/deepseek/gateway/ollama only; also added
to the daemon's sandbox network rows), `FAKTOR_BENCH_NETWORK` (extra
allowlist rows), `FAKTOR_BENCH_TASK_TIMEOUT_S`, `FAKTOR_BENCH_VERIFY_TIMEOUT_S`,
`FAKTOR_BENCH_VERIFY_OUTPUT_CAP`, `FAKTOR_BENCH_SUMMARY_PROMPT=0` (skip the
criteria suffix), `FAKTOR_BENCH_ONLY=<task-id>`, `FAKTOR_BENCH_TAG`,
`FAKTOR_BENCH_DEBUG=1` (append the daemon stderr tail to each row's note),
`FAKTOR_BENCH_REPORT_PATH`. Missing env/keys/binary = documented skip.

The model id must be one the provider's daemon-side catalog serves (the
daemon refuses an unknown pin at boot with the exact message).

## Scoring and cost definitions

- **verified** — deterministic, objective: the task's own `verify.sh`
  exits 0 in the workspace copy within the verify timeout and output cap.
- **criteria** — deterministic checker (no LLM judging): each `crit-NN`
  key of `criteria.md` counts as met when it appears as its own token in
  the assistant's final summary (the prompt demands `crit-NN: PASS/FAIL —
  evidence` lines), corroborated by the durable verification-record rows
  of the session when they exist (criterion key or full canonical line
  with `passed=true`).
- **attempts** — admitted logical turns from `/native/session/{id}/turns`
  (durable turn records).
- **cost** — durable spent cost in micro-US dollars (the daemon's cost
  ledger via `/native/session/{id}/tasks` → `budget.spentCostMicro`,
  falling back to `/native/usage` totals). Tokens = the ledger's
  `budget.spentTokens`. Both are honest zeros + report notes when the
  runtime recorded nothing.
- **cost-to-verified-success** = total spend / verified tasks (`None`
  when nothing verified — undefined is never silently 0).

## Gates

- Corpus immutability is tested (byte snapshot before/after runs); the
  harness only ever writes temp copies.
- Hostile corpus entries (traversal ids, symlinks, oversized metadata,
  malformed criteria) are loader errors.
- `verify.sh` output is capped (1 MiB default; only a diagnostic tail is
  retained) and wall-bounded (180 s default); timeout kills the whole
  process group (`verify.sh` + cargo/rustc/make/go/node/javac children).
- Every daemon/workspace lives in its own `tempdir`; concurrent runs are
  isolated; no global mutable state.
- Hard caps: ≤ 12 tasks, ≤ 200 corpus files, ≤ 64 files per task,
  metadata ≤ 64 KiB, ≤ 32 criteria per task.

## Residual risks

- Real-mode prompt-drive sessions do not go through the TaskExecutor
  wrapper (no HTTP route starts one); a prompt drive is the same agent
  drive path the UI uses, and verification-record rows may be absent for
  such sessions — `criteria` then rests on the final summary, and
  `record_status` is reported as `None`.
- The session's numeric task id is not exposed on the read surface; the
  verification-record endpoint is probed for the first ids (bounded,
  session-scoped server-side).
- Token detail (input vs output) is not exposed; the ledger's total is
  reported.
- Daemon version drift: the runner reads the wire defensively (normalized
  field lookup) and fails loud on the frozen startup line only.
- GitHub-hosted runner toolchains vary (e.g. node < 22.6, missing make or
  javac): those tasks skip with a note; coverage is best on a dev machine.
