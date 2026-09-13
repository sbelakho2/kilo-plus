# Faktor certification

This document defines what **certified** means in this repository, the exact
commands that establish it, the certificate manifest schema, and the release
certification rule. The normative architecture is `docs/architecture.md`;
this file only defines the evidence bar.

Certification is **evidence over the exact commit** it ran on. There is no
"certified branch", no "it passed last week", and no claim inherited from a
different SHA. The local harness never uses the network, provider keys, or
an LLM: everything below is deterministic and offline.

---

## 1. Commands and profiles

| Command | Profile | What it establishes |
| --- | --- | --- |
| `bash scripts/certify-local.sh fast` | fast (default) | The change-level gate for the host lane: formatting, check, derived capability manifest + docs drift, clippy, workspace tests, static-authority scans, fault smoke, doctor `--deep`, branding scan, release CLI doctor. Minutes. |
| `bash scripts/certify-local.sh full` | full | Everything in fast plus the long lanes: release `[perf]` distribution gates, `[fault]` campaigns at scale, coding-benchmark harness smoke, efficiency harness, ACP interop, artifact packaging and the installation matrix. Longer. The packaging section is the only one that may fetch npm packages (VSIX tooling); an unreachable registry is a recorded skip. |
| `CERTIFY_SELFTEST=force_fail CERTIFY_OUT_DIR=/tmp/cert-selftest bash scripts/certify-local.sh fast` | selftest | Injects a synthetic failing section and proves the harness exits non-zero, records the failure, fail-fast marks the remainder skipped, and the manifest carries the certification schema with every flag false. Does not touch the real certificate. |
| `CERTIFY_SELFTEST=release_gates bash scripts/certify-local.sh fast` | selftest | Proves the pure release rule: `release_certified` requires `local_offline_certified` AND all three external evidence gates (cross-platform lanes, real provider, real soak). Exits 0 only when every assertion holds. |
| `cargo test -p faktor-tests-fault --release -- --ignored` | long lane | The full `[fault]` campaigns (also part of `full`). |
| `cargo test -p faktor-tests-performance --release -- --ignored` | long lane | The `[perf]` distribution gates (also part of `full`). |
| `cargo test -p faktor-tests-fuzz-seeds` | long lane | Seeded pseudo-fuzz harnesses + bounded deterministic campaign; owned by the Woodpecker `static` job and manual runs. |
| `bash scripts/package-artifacts.sh` | packaging (part of `full`) | Builds the release daemon bundle (`tar.gz`), the VS Code VSIX (via `npx @vscode/vsce`) and copies the JetBrains plugin zip when present; writes `target/certification/artifacts.json` with `{name, path, sha256, size, commit, status, detail}` per artifact, recording exact errors and retry commands for anything not produced. |
| `node scripts/install-matrix.mjs` | matrix (part of `full`) | Installs/verifies every built artifact on this host into clean temp prefixes: daemon extraction + `faktor-cli doctor --data-dir <tmp>`, VSIX zip/manifest structure, JetBrains `plugin.xml` id/version; writes `target/certification/install-matrix.json`. Non-zero on any verification failure. |
| `TAMPER=1 node scripts/install-matrix.mjs` | matrix self-test | Copies a built artifact, flips one byte, and requires the verifier to reject the copy (sha256 mismatch); exits 0 only on rejection. Evidence: `target/certification/install-matrix-tamper.json`. |
| `bash scripts/certify.sh` | legacy wrapper | The older 8-gate release wrapper; `certify-local.sh full` supersedes it with the manifest. Kept for compatibility. |

### Certification levels

The manifest records exactly one `certification_level`:

| Level | Rule | Meaning |
| --- | --- | --- |
| `none` | anything else | No certificate; a failed run, a fast-profile run, or a dirty tree. |
| `local_offline` | clean tree + `full` profile + `status=pass` + tests not skipped | This host certified the change offline: no network, no provider keys, no LLM. It is **not** a release certificate. |
| `release` | `local_offline` **plus all three** external evidence gates at the same SHA | A shippable release certificate. |

The three external evidence gates are inputs the offline harness can never
produce by itself and must never fabricate:

| Env var | Gate |
| --- | --- |
| `CERTIFY_CROSS_PLATFORM_LANES=1` | Every CI platform lane (§2.1) green at this exact SHA. |
| `CERTIFY_REAL_PROVIDER=1` | A recorded real-provider (keyed) run at this SHA. |
| `CERTIFY_REAL_SOAK=1` | A recorded wall-clock soak at this SHA. |

`release_certified` is `false` whenever any gate is absent, however green
the local run is. `FAST_TESTS_SKIP=1` is a **dry-run aid only**: it records
`workspace-tests` as a skipped section with the reason instead of running
it. Such a manifest can never be `local_offline` or release certified.

### Fast section order (fail-fast)

1. `cargo fmt --check`
2. `cargo check --workspace`
3. capability manifest + docs drift (`node scripts/capabilities-manifest.mjs`, skipped when node is absent)
4. `cargo clippy --workspace --all-targets -- -D warnings`
5. `cargo test --workspace` (wrapped in `caffeinate -i` on macOS)
6. static-authority scans (`faktor-tests-static-authority`)
7. fault campaign smoke (`faktor-tests-fault`, non-ignored)
8. `doctor --deep` on a fresh temp data dir
9. branding scan (`scripts/branding-scan.sh`, plus packaged artifacts when present)
10. release CLI build + `doctor --deep` on an empty data dir

### Full adds (after fast, same order)

11. `[perf]` release distribution gates (`faktor-tests-performance --release -- --ignored`)
12. `[fault]` campaigns at scale (`faktor-tests-fault --release -- --ignored`)
13. coding-benchmark smoke (`faktor-tests-coding-benchmark --test smoke`)
14. efficiency harness (`faktor-tests-efficiency`)
15. ACP interop (`faktor-acp --test interop`)
16. artifact packaging (`scripts/package-artifacts.sh` → `artifacts.json`)
17. installation matrix (`node scripts/install-matrix.mjs` → `install-matrix.json`)

The first failure stops the run; every unrun section is recorded in
`skipped[]` with `fail-fast: not run after section '<id>' failed`. The
manifest is written **even on failure** so the evidence trail is complete.

---

## 2. Definition of 100%

A release candidate is at **100%** only when every item below holds for the
exact commit being shipped. "Lane green" means the lane's commands pass on
that commit; a lane the local host cannot run is evidenced by CI at the same
SHA, never assumed.

### 2.1 Platform lanes

CI (`.woodpecker.yml`, Woodpecker 3.x) runs exactly these jobs. GitHub
Actions was the previous runner; its workflows were removed when CI migrated
(historical note only, no workflow files remain under `.github/`).

| Job | Agent | Content |
| --- | --- | --- |
| `linux` | linux/amd64 | fmt; clippy `--workspace --all-targets --all-features -D warnings`; `check` + tests `--workspace --all-features` (protocol/compat/stream codecs ride this run); `doctor` smoke |
| `static` | linux/amd64 | static-authority scans; security suite; seeded fuzz; supply-chain SBOM/checksums/advisories with recorded skips |
| `docs` | linux/amd64 | `cargo doc --workspace --all-features --no-deps`; branding scan; docs-sync guard |
| `vscode` | linux/amd64 | `npm ci` + build; offline extension/bridge selftest; vendored webview staging; `vsce` VSIX; unzip + pinned-hash verify + packaged selftest; IDE-load record (recorded skip when no `code` CLI exists) |
| `vscode-visual` | linux/amd64 | REQUIRED render gate: playwright + chromium installed explicitly, then fails unless `dist/visual-report.json` records `render.status == "passed"` (a skip is not a pass) |
| `vscode-visual-with-skip` | linux/amd64 | diagnostics twin: offline structural/manifest gate + render attempt; records an explicit skip when chromium cannot be installed (never required) |
| `jetbrains-build` | linux/amd64 | `./gradlew :frontend:buildPlugin --no-daemon --stacktrace` |
| `jetbrains-smoke` | linux/amd64 | kotlinc split-mode compile + wire/native smokes against a real daemon |
| `perf` | linux/amd64 | release `[perf]` gates; serialized after the other Rust lanes so budgets do not race a loaded agent |
| `darwin-check` / `darwin-test` / `darwin-doctor` | self-hosted macOS (`local` backend) | `check`/`test --workspace` + `doctor`; `push` to `main` only |
| `windows-check` / `windows-test` | self-hosted Windows (`local` backend) | `cargo check --workspace`; workspace tests INCLUDING faktor-agent, faktor-verify, faktor-sandbox, faktor-index, faktor-cas and faktor-snapshot (unix-only tests are `cfg(unix)`-gated); only faktor-cli, faktor-hooks, faktor-tests-coding-benchmark, faktor-tests-fuzz-seeds and faktor-tests-performance stay excluded for documented unix-only symbols/scripts; `push` to `main` only |
| `certificate` | linux/amd64 | aggregate gate over every linux lane: verifies each lane's marker and the workflow status, writes `target/certification/ci-certification.json` |
| `certificate-darwin` / `certificate-windows` | self-hosted platform agent | per-platform aggregate marker/status gate |

Aggregation is marker-based because Woodpecker workflows are
filesystem-isolated: every lane writes
`target/certification/lanes/<lane>.json` (`faktor-woodpecker-lane/v1`) into
the shared workflow workspace, and the `certificate` step `depends_on` every
lane and runs with `when.status: [success, failure]` (dependents are skipped
by default when a dependency fails). A missing, failed, stale or unexpected
marker — or a non-success `CI_PIPELINE_STATUS` — fails the certificate, so an
unexpected skip cannot masquerade as a pass. `scripts/woodpecker/setup.md`
documents the agent labels, self-hosted macOS/Windows agents, trusted-volume
caching and the free cloud tier (linux runners only).

100% requires the linux certificate green at the exact commit, plus the
platform certificates when self-hosted darwin/windows agents exist. The
release real-time soak (12–24h wall clock) is a self-hosted hook and is
deliberately **not** part of the default pipelines; `bash
scripts/certify-local.sh full` keeps the long release lanes ([perf], [fault]
at scale, coding benchmark, efficiency, ACP interop, packaging, installation
matrix) runnable offline, and the real soak must be run and recorded
separately when required for a release.

### 2.2 UI builds and parity

- **VS Code** (`apps/vscode`): `npm ci && npm run build` green in CI, and
  the wire harness (`bash scripts/run-vscode-harness.sh`, which drives
  `apps/vscode/harness/client.mjs` against a real `faktor-cli` binary)
  green. The derived client shell is **IMPLEMENTED**; the pinned v7.5.6
  webview bundle is **vendored** (`ui/kilo-v756-webview`, hashed by
  `ui/upstream.json`) and built (`dist/webview.js` + `dist/webview.css`),
  with the visual gate baseline recorded
  (`dist/visual-baseline.json`). The `vscode-visual` job installs
  playwright + chromium explicitly and is required to record
  `render.status == "passed"` for the built bundle; the
  `vscode-visual-with-skip` twin is diagnostics only. A real-IDE screenshot
  comparison against a launched VS Code remains a host/CI capability not
  claimed by the offline certificate.
- **JetBrains** (`apps/jetbrains`): `bash apps/jetbrains/compile-and-smoke.sh`
  green (`:shared` + `:backend` + `:frontend` Swing panel, real kotlinc,
  real daemon: v7.5.6 wire smoke plus native-protocol fake-server unit
  suite and end-to-end native smoke), and the real Gradle lane green:
  `./gradlew buildPlugin`, `./gradlew verifyPluginProjectConfiguration`
  (no configuration issues) and `./gradlew verifyPlugin` against the
  pinned IntelliJ IDEA Community 2024.1.7 distribution
  (`IC-241.19416.15`) with verdict `Compatible` and zero reported API
  problems (report under `frontend/build/reports/pluginVerifier/`).
  Status: **native bridge
  IMPLEMENTED** — daemon lifecycle, protected-channel bearer auth, HTTP +
  SSE cursor-resume clients, and routing for task-runs, agents, usage,
  verification and evidence; the Faktor-owned Swing frontend renders the
  task tree, blockers, tournament, evidence navigator and the Task-mode
  completion-contract controls. The upstream 7.1.2 UI sources are still not
  vendored, so `jetbrains_frontend` is **PARTIAL** (Faktor frontend + daemon
  smoke) and 7.1.2 UI parity remains **BLOCKED_EXTERNAL**. Only the
  2024.1.7 distribution was verified; `until-build` stays unbounded, so
  newer-platform compatibility is not claimed.

100% requires the builds and smokes green **and the derived capability
manifest labels honest**. It does not require byte-for-byte parity where
the upstream assets are absent, but no certificate may claim parity that
the tree cannot substantiate. `target/certification/capabilities.json`
(§2.10) carries the derived labels; `ui_parity` is PARTIAL while the
JetBrains 7.1.2 sources are not vendored.

### 2.3 Compat fixtures

- `compat/kilo-v756/` golden fixtures (startup line, Basic auth, session
  create, message send, paging, SSE frames, provider list, errors) are
  frozen: `tests/compat` asserts the daemon against them and regenerating a
  golden requires an explicit, reviewed contract change.
- `compat/jetbrains-712/` is a reserved fixture corpus; existence is
  recorded per-run in the manifest (`compat_fixtures.jetbrains712`). While
  absent it must be `false`, never silently assumed.

100% requires the compat suite green and the v756 golden files byte-stable.

### 2.4 ACP interop

`cargo test -p faktor-tests-acp-official` must be green: the OFFICIAL
`agent-client-protocol` v2.1 client (unmodified `ByteStreams` NDJSON
transport, string/UUID request ids) drives a real `faktor-acp` server end to
end. It certifies the official wire surface — NDJSON framing with no
Content-Length translation, notifications that omit `id`, verbatim string
request-id echo, `$/cancel_request` cancellation with exactly one terminal,
and the retained legacy Content-Length compatibility mode.

`cargo test -p faktor-acp --test interop` must also be green, covering the
official handshake, session lifecycle, cancellation exactly once,
reconnect after server drop, per-connection/session isolation, official
error shapes on bad params, oversized requests bounded (close, not hang),
slow-client backpressure lossless, unknown-kind classification (never
guessed), rogue truncation/garbage never panicking or hanging, fragmented
requests over real TCP, idle close as clean EOF, and `session/load` replay
order. These are adversarial interop families, not happy paths.

### 2.5 Fault and fuzz clean

- Fault smoke (`cargo test -p faktor-tests-fault`) and, for a release, the
  full seeded campaigns in release mode: store/journal crash at every
  durability boundary (500 seeds), CAS put/read mid-write (500), scheduler
  DAG crash vs reference terminal set (300), edit-transaction begin/commit/
  recover (300), accounting crash seams (200).
- Fuzz hygiene: `cargo test -p faktor-tests-fuzz-seeds` (2000 seeded cases per
  harness: destination policy parser, event payload decode, line framing,
  path normalization, SSE frame decode).
- Containment: after a campaign, `doctor --deep` on a clean dir must pass —
  proven corruption must not leak into a fresh image.

100% means zero escapes: no panic, no hang, no orphan, no corrupted store
or CAS state, no lost/duplicated durability boundary.

### 2.6 Doctor invariants zero

`doctor --deep` must print `doctor: all checks passed` and exit 0, with
zero `issue(s)`, on both a fresh data dir and an empty release data dir.
The invariants it audits:

1. store open + quick diagnostics;
2. unfinished tool runs across sessions;
3. deep store integrity scan;
4. CAS blob verification (every referenced blob hashes correctly);
5. CAS references (artifact rows + checkpoint after-blobs) — zero dangling;
6. active logical turns across sessions;
7. journal projection consistency (gapless 1..=N per session);
8. cost reservations — zero dangling (a reservation whose task row is gone
   can never settle or refund);
9. verification-record consistency (Passed records reference existing task
   rows, current revision, and `VerifiedComplete` tasks carry theirs);
10. active turns with recoverable owners (a crashed daemon can recover or
    deliberately fail every active turn);
11. orphan children (child identity/registry rows and non-terminal
    worktrees must resolve);
12. orphan-process ownership (informational in a separate doctor process;
    the zero-orphan guarantee is the in-process lifetime contract).

### 2.7 Perf distributions within budgets

Budgets are enforced as assertions on distributions (p50/p95), not single
samples, per `tests/performance`:

| Gate | Budget |
| --- | --- |
| Warm page load / state read | < 5 ms warm (debug assert allows 20 ms headroom) |
| 50k-message history | initial page no worse than a small session + 5 ms |
| Cold start (daemon stack) | < 150 ms typical release (debug assert < 500 ms) |
| Idle daemon memory | RSS < 80 MB after churn |
| Cached symbol lookup | < 10 ms |
| Paging over 50k messages | p95 < 5 ms per page |
| 4 KiB JSON wire round trip | p50 < 100 µs |
| Growing transcript (2k → 20k) | large-end p50 ≤ 3× small-end p50 |
| 20k-message context plan (release) | p95 < 250 ms |

100% means the release `[perf]` lane is green for the commit, with the
runner/build metadata recorded by the harness.

### 2.8 Verified-success economics targets

Economy and router certification (`tests/economy`, `tests/efficiency`) must
be green:

| Gate | Target |
| --- | --- |
| Aggregate routing cost | economy ≤ 65% of the frontier lane on the fake corpus |
| Cache economics | cached route ≥ 25% cheaper than uncached |
| Certification mix (5 fixed seeds) | every task verified; realized cost-to-success ≤ 1.05× the frontier lane per seed and in aggregate |
| Cost-to-success dispersion | p95 ≤ 3× mean across seeds |
| Escalation discipline | naive cheapest-always loses the hard mix; the router escalates and still holds the frontier gate |
| Hard budget | routing never overshoots a task's remaining budget |
| Efficiency KPI | exact deterministic derivation from durable rows (no double counting of prefix rows, corrupt shapes rejected, reopen stable) |

Real-model coding-benchmark runs (`--test real -- --ignored`) need provider
keys and are **always recorded as skipped** by the local profile. A release
that claims verified real-model economics must attach one such run for the
same SHA and record the `real_provider` gate (§1); the offline certificate
never implies it.

### 2.9 Installable artifacts and the installation matrix

The `full` profile ends with packaging and installation evidence; both are
bound to the same commit as the rest of the manifest.

- `scripts/package-artifacts.sh` builds and records:
  - `faktor-cli-<version>-<os>-<arch>.tar.gz` — the release daemon bundle
    (`bin/faktor-cli` plus `checksums.txt` and `RELEASE` metadata). The
    daemon is built with `cargo build --release -p faktor-cli`.
  - `faktor-<version>.vsix` — the VS Code extension, built with
    `npm ci` (only when `node_modules` is absent) + `npm run build` and
    packaged with `npx --yes @vscode/vsce package`. If node/npm/vsce or the
    registry is unavailable, the exact error and the runnable retry command
    are recorded as a skip, never silently claimed.
  - `faktor-jetbrains-plugin-<version>.zip` — a copy of the Gradle plugin
    zip from `apps/jetbrains/frontend/build/distributions/` when present.
  - `target/certification/artifacts.json` — `{name, path, sha256, size,
    commit, status, detail}` per artifact, plus recorded skips.
- `scripts/install-matrix.mjs` reads that manifest, verifies each built
  artifact's sha256/size, and installs it into a clean temp prefix on this
  host:
  - daemon bundle: tar layout (`bin/faktor-cli`, `checksums.txt`), extraction,
    bundle-internal checksum recomputation, `faktor-cli --version`, and
    `faktor-cli doctor --data-dir <fresh tmp dir>` requiring exit 0 and
    `doctor: all checks passed`;
  - VSIX: zip structure (`extension/package.json`,
    `extension.vsixmanifest`), parseable `package.json`, the declared `main`
    entry present in the archive, non-empty `contributes` with commands and
    views;
  - JetBrains zip: bundled plugin jars and `META-INF/plugin.xml` inside the
    frontend jar, with a non-empty id and version.
  The report is `target/certification/install-matrix.json`; any failed check
  exits non-zero. `MATRIX_REQUIRE` selects which kinds must verify
  (default `daemon-bundle`).
- `TAMPER=1 node scripts/install-matrix.mjs` is the matrix self-test: it
  copies a built artifact, flips one byte, points a cloned manifest at the
  tampered copy (same recorded sha256) and requires the verifier to reject
  it. It exits 0 only when the tampered copy was rejected, and records
  `target/certification/install-matrix-tamper.json`.
- **Residual (CI-only, never claimed locally):** installing the VSIX into a
  real VS Code instance (`code --install-extension`) and the JetBrains plugin
  into a real IDE sandbox need an IDE host. The Woodpecker `vscode` job owns those
  steps; the host matrix only proves the archives are structurally
  installable and that the daemon actually runs from an extracted bundle.
  The report's `residual[]` records both items.

The offline contract still holds: packaging may *attempt* the npm registry
for VSIX tooling, but the certificate never depends on that attempt
succeeding — a failure is recorded as a skip with the exact error.

### 2.10 Capability manifest (derived, machine-readable)

`node scripts/capabilities-manifest.mjs` derives every surface status from
repository files and scripts — never from prose — and writes
`target/certification/capabilities.json`. Its default mode (also invoked by
`bash scripts/certify-local.sh fast`) is the drift test: it exits non-zero
when the table below disagrees with the derived manifest, when a surface row
is missing, or when the table lists an unknown surface. The manifest is
bound to the commit it was generated on; a stale file from a different SHA
is not evidence. Any change to the probed files that moves a status must
update this table in the same commit.

| Capability | Status | Derived from |
| --- | --- | --- |
| `vscode_native_client` | IMPLEMENTED | `apps/vscode/src/nativeClient.ts` + `apps/vscode/scripts/selftest.mjs` |
| `vscode_webview` | IMPLEMENTED | `apps/vscode/src/webview.ts` + pinned `ui/kilo-v756-webview/dist` bundle (`webview.js`, `webview.css`) + `dist/visual-baseline.json` |
| `jetbrains_native_bridge` | IMPLEMENTED | `NativeClient.kt`, `NativeEventStream.kt`, `apps/jetbrains/compile-and-smoke.sh` |
| `jetbrains_frontend` | PARTIAL | Faktor-owned Swing frontend (`FaktorChatPanel.kt`, `TaskTreePanel.kt`, `plugin.xml`, `build.gradle.kts`); upstream 7.1.2 UI not vendored |
| `compat_v756` | IMPLEMENTED | `compat/kilo-v756` golden fixtures + `tests/compat` |
| `ui_parity` | PARTIAL | vendored v7.5.6 webview + visual baseline; JetBrains 7.1.2 sources absent |
| `acp_subset` | IMPLEMENTED | `crates/acp` + `tests/acp-official` official-client interop |

---

## 3. Honest current status

Status labels used: **CERTIFIED** (evidence exists at the referenced
commit), **CI-LANE** (owned and run by CI, not by the local harness),
**PARTIAL** (implemented subset), **BLOCKED_EXTERNAL** (needs assets not in
this repository), **NOT RUN HERE** (deliberately out of the offline local
profile).

| Surface | Gate | Evidence | Status |
| --- | --- | --- | --- |
| Local host lane (darwin) | fast profile: fmt, check, clippy, tests, static authority, fault smoke, doctor deep, branding, release CLI | `scripts/certify-local.sh fast` → `target/certification/manifest.json` | CERTIFIED per run (see §3.1) |
| Perf distributions | release `[perf]` gates | `full` profile / Woodpecker `perf` job | CI-LANE |
| Fault at scale | `[fault] --ignored` campaigns | `full` profile / nightly `fault` job | CI-LANE |
| Coding benchmark (harness) | `smoke` suite, offline | `full` profile | CERTIFIED in `full` only |
| Coding benchmark (real model) | `real --ignored`, provider keys | manual, keyed | NOT RUN HERE (recorded skip) |
| Efficiency | KPI harness | `full` profile | CERTIFIED in `full` only |
| ACP interop | `faktor-acp --test interop` | `full` profile / Woodpecker `windows-*` + `linux` jobs | CERTIFIED in `full` only |
| Installable artifacts (host) | daemon tar.gz + VSIX + JetBrains zip + `artifacts.json` | `full` profile (`scripts/package-artifacts.sh`, §2.9) | CERTIFIED per full run (recorded skips with exact errors when a tool/registry is absent) |
| Installation matrix (host) | clean-prefix extract + `doctor`; VSIX/zip structure + entry points | `full` profile (`node scripts/install-matrix.mjs` → `install-matrix.json`, §2.9) | CERTIFIED per full run |
| IDE-launched install | `code --install-extension` / JetBrains sandbox install | requires an IDE host; the Woodpecker `vscode` job owns it | NOT RUN HERE (residual, recorded in `install-matrix.json`) |
| Windows lane | check + workspace tests incl. agent/verify/sandbox/index/cas/snapshot | Woodpecker `windows-*` jobs | CI-LANE |
| Linux lane | fmt/check/test/clippy/doctor | Woodpecker `linux` job | CI-LANE |
| VS Code shell build | `npm ci && npm run build` + wire harness | Woodpecker `vscode` job | CI-LANE (shell IMPLEMENTED) |
| VS Code vendored webview | pinned v7.5.6 tree `ui/kilo-v756-webview` + `ui/upstream.json` hashes + built dist + visual baseline | `node scripts/webview-visual-check.mjs` + required Woodpecker `vscode-visual` job (chromium render must pass); §2.10 | PARTIAL (vendored, hashed and render-gated; real-IDE screenshot parity stays a host/CI capability not claimed offline) |
| JetBrains bridge | kotlinc `compile-and-smoke.sh` (wire + native smokes); Gradle plugin build + verifier vs IC-2024.1.7 | Woodpecker `jetbrains-*` jobs / local script; §3.2 | CI-LANE (native bridge IMPLEMENTED; plugin verifier PASS locally 2026-09-10) |
| JetBrains 7.1.2 UI parity | frozen 7.1.2 sources | not vendored | BLOCKED_EXTERNAL |
| Compat fixtures v756 | golden suite + fixtures | `tests/compat` | CI-LANE / fast tests |
| Compat fixtures jetbrains-712 | reserved corpus | absent (`false` in manifest) | BLOCKED_EXTERNAL |
| Fuzz harnesses | seeded pseudo-fuzz | Woodpecker `static` job / manual | CI-LANE |
| Real-time soak (12–24h) | wall-clock soak | self-hosted hook (disabled by default) | NOT RUN HERE |
| PR/CI-fix completion contract | native DTO `completion_contract` + `CompletionContractSet`/`CompletionStepStatus` ledger rows + `VerifiedComplete` gate + ordered step executor | gate + durable rows + `crates/orchestrator/src/completion_steps.rs` runner (`crates/session/src/task.rs`, `crates/session/src/ledger.rs`, `crates/orchestrator/src/task_executor.rs`, `crates/agent/src/runtime.rs`); adversarial gate/step tests in-tree; Task-mode controls in both IDEs (§3.3) | IMPLEMENTED (gate + ordered/idempotent commit/push/PR execution) |
| Coordination board | durable ledger rows (`board_post`/`board_read`/`board_receipt`/`board_reset`), CAS reset, scoped reads, board tools, native `GET/POST /native/session/{id}/board`, both IDE board panels | `crates/session/src/board.rs` + `crates/session/src/ledger.rs`; `crates/server/src/native/board.rs`; `apps/vscode/src/nativeClient.ts`; JetBrains `BoardPanel.kt` | IMPLEMENTED (fast tests green; native GET/POST round-trip in both IDE smokes; unavailable state recorded truthfully) |
| Multi-candidate tournament | N = 2..=4 identical-criteria candidates, deterministic winner ordering, durable decide, loser cleanup, cross-IDE controls | `crates/orchestrator/src/tournament.rs` + `crates/orchestrator/src/task_executor.rs`; native start/state/list endpoints; VS Code cockpit + JetBrains `TournamentPanel.kt` (decide gated on every candidate settled) | IMPLEMENTED (fast tests green + both IDE smokes; integration stays the explicit approved-merge path) |
| Pixel agents | deterministic per-ChildId avatars (VS Code + JetBrains, identical FNV-1a hashes) | `apps/vscode/src/pixelAgents.ts`, `apps/jetbrains/frontend/src/main/kotlin/dev/faktor/frontend/PixelAgents.kt` | IMPLEMENTED (UI layers; daemon exposes the durable child ids/state they render) |
| Canonical child lifecycle + typed blockers | typed core `ChildBlocker` (kind/dependency/resolution/last-progress) with `blocked <=> blocker` decode invariants; corrupt rows fail loudly; canonical child-state projection | `crates/core/src/blocker.rs`, `crates/session/src/child.rs` (`child_runtime` v23 row) + native agent projection (`state`/`blocker` fields) | IMPLEMENTED (fast tests green) |
| Presentation continuity | durable foreground/background fold + native `POST /native/session/{id}/agents/{child}/presentation` | `crates/session/src/child.rs` (`set_child_presentation`/`child_presentation`), `crates/session/src/ledger.rs` (`child_presentation_changed`), strict DTO + `presentation` field in the native agent projection | IMPLEMENTED (fast tests green; presentation-only, same ChildId/lineage) |
| Durable execution phase | `ExecutionPhase` persisted at child drive boundaries and projected additively (`execution_phase`) on native child payloads; unknown/hostile values fail loud | `crates/core/src/blocker.rs` (`ExecutionPhase`), `crates/session/src/child.rs` (`set_execution_phase`/`execution_phase`), `crates/server/src/native/agents.rs` | IMPLEMENTED (fast tests green; purely a projection, never scheduling) |
| Control ack discipline | every acknowledged control transition is durable before the ACK; a persistence failure is a retriable typed 503 (`persistence_failed`), never a swallowed ack | `crates/agent/src/runtime.rs`, `crates/session/src/handle.rs`, `crates/server/src/native/agents.rs` | IMPLEMENTED (fast tests green) |
| Provider identity on children | the child session's durable `provider` rides the native entry; `(provider, model)` is the only catalog join key in both IDEs (a provider-less entry never guesses by model) | `crates/server/src/native/agents.rs`, `apps/vscode/src/state.ts` (`summarizeAgents`), JetBrains `TaskTreeModel.kt` | IMPLEMENTED (dual-provider smoke pins the join) |
| Accessibility parity | reduced-motion handling for every pixel/state animation (VS Code CSS + JetBrains `PixelMotion`), static terminal cues, identical deterministic identities | `apps/vscode/media/chat.css`, `apps/vscode/media/chat.js`, JetBrains `PixelAgents.kt` + `FrontendSmoke.kt` reduced-motion matrix | IMPLEMENTED (both IDE smokes adversarial) |
| Board provenance + delivery view | durable `BoardDeliveryView` folded from posts/reads/receipts/lifecycle; board tools emit `AgentCoordination` provenance so a sibling post never gains instruction authority | `crates/session/src/board.rs`, `crates/evidence/src/provenance.rs`, `crates/agent/src/tool.rs` (`BOARD_TOOL_PROVENANCE`) | IMPLEMENTED (malicious-post E2E green) |
| Vendored bridge completeness | every frozen inbound message bounded/validated with typed drops; additive `faktor*` frames + companion overlay; durable pages and SSE frames both mapped | `apps/vscode/src/kilo-bridge.ts` + `apps/vscode/scripts/bridge-selftest.mjs`, `apps/vscode/scripts/selftest.mjs` | IMPLEMENTED (adversarial bridge selftests green) |
| Attachment durability | workspace-relative file paths + content-addressed binary refs; per-message durable `data.files` rows and the run's immutable `files` set persisted before the drive (byte-identical after reopen); malformed entries refused individually (never a whole-message drop) | `crates/session/src/handle.rs` (`submit_prompt` `data.files`), `crates/session/src/task.rs` + `crates/orchestrator/src/task_executor.rs` (run `files`), `apps/vscode/src/kilo-bridge.ts` (`mapKiloFiles`), JetBrains `AttachmentsPanel.kt` | IMPLEMENTED (both IDEs; bridge + native smokes green) |
| Unified run settlement | one `settle_run` for both execution shapes (`InSession`/`Orchestrated`), idempotent replay after every step-status write seam | `crates/orchestrator/src/task_executor.rs` (`settle_run`), `crates/orchestrator/src/task_executor_tests.rs` | IMPLEMENTED (fast tests green) |
| VSIX packaging (P0) | pinned vendored closure + companion overlay staged under `media/kilo-v756-webview`; extensionUri-only resolution; package → unzip → hash verify → IDE-load record | `apps/vscode/scripts/prepare-vendored-webview.mjs`, `apps/vscode/scripts/verify-vsix.mjs`, Woodpecker `vscode` job | CLOSED (self-contained VSIX; the IDE-load step records its exact skip when no `code` CLI exists) |
| Repo rename (faktor) | external GitHub repository name/description | `gh repo rename` performed; in-tree metadata was already Faktor-branded and is unchanged | DONE (external; no in-tree evidence beyond branding scan) |

### 3.1 Last recorded local fast run

`bash scripts/certify-local.sh fast` on 2026-09-10 (darwin/arm64, rustc
1.98.0) for commit `f6b1c2f7fabd92b45647fb240e362dfaff268d5b`:
**PASS**, 9/9 sections, 463,589 ms, 8 recorded skips (fast-profile long
lanes, the offline provider-key run, the Windows lane, the real soak). The
worktree carried concurrent changes during that run (`dirty_count: 20`),
so it is a work certificate, not a release certificate.
`target/certification/manifest.json` is authoritative and is regenerated on
every run; a stale manifest for a different commit is not evidence.

### 3.2 Last recorded JetBrains plugin verification

Real IntelliJ Platform verification on 2026-09-10 (darwin/arm64, Gradle
9.7.1 wrapper, Kotlin 2.4.20, JDK 17, IntelliJ IDEA Community 2024.1.7 =
`IC-241.19416.15`; worktree dirty, so this is a work certificate, not a
release certificate):

- `./gradlew verifyPluginProjectConfiguration` → PASS, no configuration
  issues. The previous Kotlin stdlib conflict is resolved with
  `kotlin.stdlib.default.dependency=false` and an explicit
  `compileOnly("org.jetbrains.kotlin:kotlin-stdlib:1.9.22")` (the 2024.1
  bundled version); the plugin zip bundles no stdlib.
- `./gradlew verifyPlugin` (verifier 1.410; IDE pinned with
  `pluginVerification { ides { current() } }`) → `Compatible`, zero
  deprecated / experimental / internal API usages. Before the fix, the
  Kotlin compiler emitted synthetic `ToolWindowFactory` default-method
  bridges (4 deprecated + 2 experimental + 6 internal usages); compiling
  with `JvmDefaultMode.NO_COMPATIBILITY` removes them.
- `./gradlew build` → PASS (includes `:backend:test`).
- `./gradlew runIde` → the IDE starts with the plugin installed
  (`Loaded custom plugins: Faktor (0.1.0)` in the sandbox `idea.log`) and
  no display/headless failure on this GUI host; the run was terminated by
  the harness after 300 s (`timeout` exit 124) and the IDE logged a clean
  `IDE SHUTDOWN`. No project was opened, so tool-window behavior was not
  exercised by this run.
- `bash apps/jetbrains/compile-and-smoke.sh` → exit 0, `SMOKE PASS` plus
  `NATIVE SMOKE PASS`.

### 3.3 PR/CI-fix completion contract (gate + ordered step execution IMPLEMENTED)

The reviewed P2 item asks for a first-class PR/CI-fix completion contract so a
native task can declare:

```json
"completion_contract": { "include_commit": true, "include_push": true, "include_pr": true }
```

Normative semantics (recorded so behavior cannot drift):

- The native task-run start DTO parses `completion_contract` strictly (typed
  400 on unknown fields or non-boolean members; never a silent default). A
  non-default contract requires an explicit work item: the plain-prompt path
  refuses it with a typed 400 instead of dropping the contract.
- The accepted contract is recorded durably as the typed ledger entry
  `CompletionContractSet` before the run is driven; the completion gate reads
  the durable record, never the request.
- `include_commit`, `include_push` and `include_pr` are conditional steps:
  the completion path refuses `VerifiedComplete` with a typed error while any
  requested step has no succeeding durable step-status row. A missing row is
  "not done", never "probably fine".
- `crates/orchestrator/src/completion_steps.rs` executes the requested steps
  in gate order once the drive settles: commit (clean tree => honest
  `Skipped` "nothing to commit"; unborn HEAD => `Skipped`), push (no remote
  => `Skipped`; egress policy denial => typed refusal), then PR (executed
  ONLY when the daemon has a configured `pr_command`; strict argv template
  with no shell, `{branch}` required; unconfigured => `Skipped`). Success is
  idempotent per `(task, revision)`, a failure stops everything after it and
  records the remaining requested steps `Skipped` ("not attempted"), and
  every outcome is written through the durable
  `SessionHandle::set_completion_step_status` seam before the run settles.

Implementation map:

- DTO + strict validation: `crates/server/src/native/task.rs`
  (`StartTaskRunRequest::completion_contract`, `deny_unknown_fields`; a
  non-default contract on the plain-prompt path is a typed 400, never a
  silent drop).
- Durable contract + step status: `crates/session/src/ledger.rs`
  (`LedgerPayload::CompletionContractSet` / `CompletionStepStatus`,
  `entry_tag_of`, strict decode, head fold, pinned across compaction) with
  the public accessors in `crates/session/src/task.rs`
  (`set_completion_contract`, `set_completion_step_status`). An all-false
  contract is refused (the default path carries no row); a contract is
  immutable per `(task, revision)`; a step status without a recorded
  contract is a typed Conflict.
- Gate at completion: `crates/session/src/task.rs`
  (`SessionHandle::complete_verified_task` → `completion_contract_gate`).
  A missing step row is `CompletionStepMissing`, a failed row is
  `CompletionStepFailed`, a latest non-succeeded row is
  `CompletionStepNotSucceeded`; the task STAYS `Verifying` and no
  `VerifiedComplete` is written. The only production `VerifiedComplete`
  producer (`crates/agent/src/runtime.rs`) surfaces that refusal.
- Executor seam: `crates/orchestrator/src/task_executor.rs`
  (`TaskRunRequest.completion_contract`, `record_completion_contract`
  recorded BEFORE the first model call; `run_completion_steps` drives the
  runner after the shadow/in-session drive settles and before
  `complete_verified_task`).

Task-mode IDE controls (both IDEs):

- VS Code: the built-in Task composer shows the Commit/Push/Create-PR
  checkboxes and posts a strict `completionContract`; the host builds the
  explicit `main` Implementation work item plus `completion_contract` in
  `apps/vscode/src/taskStart.ts` and forwards composer `files`; the
  `faktor.newTask` command offers the same multi-select for the vendored UI
  path; the bridge accepts the contract only as three exact booleans (a
  malformed contract is a loud drop, never a silently contract-free start)
  and the cockpit/task card renders the step rows with their provenance.
- JetBrains: the Task tab owns the same three checkboxes; the request
  builder emits the explicit work item plus contract; the task tree renders
  the contract and its step statuses.
- Neither client fabricates statuses. When a daemon serves the additive
  `completion` block on the task view the rows are `source=daemon`; without
  it the block is derived from the DURABLE task-run state (pending, or
  all-succeeded only because the durable gate certified the task) and a
  terminal non-certified run is reported `unavailable` with its reason. The
  exact per-step read remains an additive native surface; a missing read is
  never rendered as success.

Adversarial coverage in-tree: non-succeeded step refuses `VerifiedComplete`
and succeeded steps do not gate (`crates/session/src/task.rs`), contract
default parity is byte-identical and the immutable-per-revision Conflict is
enforced (`crates/orchestrator/src/task_executor.rs`
`completion_contract_executor_tests`), the agent path refuses the verified
completion until the step succeeds (`crates/agent/src/runtime.rs`
`completion_contract_gate_refuses_verified_complete_until_step_succeeds`),
the runner's ordering/idempotency/clean-tree/skipped-after-failure paths are
covered in `crates/orchestrator/src/completion_steps.rs`, and the IDE
surfaces are adversarially tested in `apps/vscode/scripts/selftest.mjs` +
`apps/vscode/scripts/bridge-selftest.mjs` and `FrontendSmoke.kt`.

---

## 4. Manifest schema (`target/certification/manifest.json`)

```json
{
  "schema": "faktor-certification-manifest/v1",
  "profile": "fast",
  "status": "pass",
  "certification_level": "none",
  "local_offline_certified": false,
  "release_certified": false,
  "release_gates": {
    "cross_platform_lanes": false,
    "real_provider": false,
    "real_soak": false
  },
  "commit": "<40-hex sha>",
  "dirty_count": 0,
  "rustc": "rustc 1.xx.y (...)",
  "cargo": "cargo 1.xx.y (...)",
  "os": "darwin",
  "arch": "aarch64",
  "timestamp": "2026-01-01T00:00:00Z",
  "duration_ms": 123456,
  "fast_tests_skipped": false,
  "sections": [
    {"name": "fmt", "label": "cargo fmt --check", "status": "pass",
     "duration_ms": 123, "detail": "ok"}
  ],
  "skipped": [
    {"name": "release-perf", "reason": "fast profile: run the full profile for [perf] release gates"}
  ],
  "capabilities": {
    "schema": "faktor-capability-manifest/v1",
    "platform": {"os": "darwin", "arch": "aarch64"},
    "platform_lanes": {"...": "..."},
    "ui_parity": {"vscode": "IMPLEMENTED", "jetbrains": "PARTIAL", "overall": "PARTIAL", "manifest": "capabilities.json"},
    "compat_fixtures": {"v756": true, "jetbrains712": false},
    "surfaces": {"workspace_tests": true, "...": false},
    "offline": {"network_required": false, "provider_keys_required": false},
    "release_rule": "a release is certified only for its exact commit with dirty=false AND local_offline_certified AND cross-platform lanes + real-provider + real-soak evidence"
  }
}
```

Field semantics:

| Field | Meaning |
| --- | --- |
| `commit` | `git rev-parse HEAD` at run start; evidence is bound to this SHA only |
| `dirty_count` | `git status --porcelain` line count; any non-zero invalidates every certificate |
| `status` | `pass` iff every attempted section passed; `fail` otherwise |
| `certification_level` | `none`, `local_offline` or `release` (see §1) |
| `local_offline_certified` | `true` only for `full` + `status=pass` + `dirty_count=0` + `fast_tests_skipped=false` |
| `release_gates` | The three external evidence inputs (`CERTIFY_CROSS_PLATFORM_LANES`, `CERTIFY_REAL_PROVIDER`, `CERTIFY_REAL_SOAK`); each defaults `false` and is never inferred |
| `release_certified` | `true` only when `local_offline_certified` AND all three `release_gates` are true |
| `sections[].status` | `pass` or `fail`; failed sections carry the first error line in `detail` |
| `sections[].duration_ms` | wall time of that section |
| `skipped[]` | sections not attempted, each with the exact reason (profile, fail-fast, offline contract, platform) |
| `capabilities` | capability manifest for this host/profile: platform lanes, derived UI parity labels (from `capabilities.json`, §2.10; `unknown` when absent/stale), compat fixture presence, surface pass flags, offline contract, release rule |

Per-section logs live in `target/certification/logs/<name>.log`.

Sibling evidence written by the `full` profile (never a substitute for the
certificate manifest): `target/certification/capabilities.json` (derived
capability manifest, §2.10; written by every profile that has node),
`target/certification/artifacts.json` (packaging
manifest, §2.9), `target/certification/install-matrix.json` (host
installation matrix, §2.9) and `target/certification/install-matrix-tamper.json`
(the `TAMPER=1` self-test evidence).

---

## 5. Release certification rule

> **A release is certified only for its exact commit with `dirty_count = 0`,
> `local_offline_certified = true`, and all three external evidence gates
> (`cross_platform_lanes`, `real_provider`, `real_soak`) recorded true.**

Concretely, to ship:

1. `git status --porcelain` is empty (a dirty tree can never be certified).
2. `bash scripts/certify-local.sh full` on the release host ends with
   `CERTIFICATION: PASS` and `"local_offline_certified": true` in the
   manifest. A `full` pass alone is **local offline certification**, never
   release certification.
3. Every CI lane in §2.1 is green for the same commit SHA; the release run
   is invoked with `CERTIFY_CROSS_PLATFORM_LANES=1` to record that gate.
4. A real-provider (keyed) benchmark/economics run is attached for the same
   SHA and recorded with `CERTIFY_REAL_PROVIDER=1`; the offline certificate
   never implies it.
5. A wall-clock soak is attached for the same SHA and recorded with
   `CERTIFY_REAL_SOAK=1`.
6. The manifest now reports `"certification_level": "release"` and
   `"release_certified": true`; any missing gate keeps both
   `release_certified=false` and the level at `local_offline` (or `none`).
7. The manifest's `capabilities` labels are honest: `BLOCKED_EXTERNAL` and
   `PARTIAL` surfaces are carried into the release notes; no parity claim is
   made for unvendored assets.

Any new commit — including a docs-only change — invalidates the previous
certificate and requires a fresh run.
