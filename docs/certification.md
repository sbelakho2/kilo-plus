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
| `bash scripts/certify-local.sh fast` | fast (default) | The change-level gate for the host lane: formatting, check, clippy, workspace tests, static-authority scans, fault smoke, doctor `--deep`, branding scan, release CLI doctor. Minutes. |
| `bash scripts/certify-local.sh full` | full | Everything in fast plus the long lanes: release `[perf]` distribution gates, `[fault]` campaigns at scale, coding-benchmark harness smoke, efficiency harness, ACP interop, artifact packaging and the installation matrix. Longer. The packaging section is the only one that may fetch npm packages (VSIX tooling); an unreachable registry is a recorded skip. |
| `CERTIFY_SELFTEST=force_fail CERTIFY_OUT_DIR=/tmp/cert-selftest bash scripts/certify-local.sh fast` | selftest | Injects a synthetic failing section and proves the harness exits non-zero, records the failure, and fail-fast marks the remainder skipped. Does not touch the real certificate. |
| `cargo test -p faktor-tests-fault --release -- --ignored` | long lane | The full `[fault]` campaigns (also part of `full`). |
| `cargo test -p faktor-tests-performance --release -- --ignored` | long lane | The `[perf]` distribution gates (also part of `full`). |
| `cargo test -p faktor-tests-fuzz-seeds` | long lane | Seeded pseudo-fuzz harnesses + bounded deterministic campaign; owned by CI's fuzz lane and manual runs. |
| `bash scripts/package-artifacts.sh` | packaging (part of `full`) | Builds the release daemon bundle (`tar.gz`), the VS Code VSIX (via `npx @vscode/vsce`) and copies the JetBrains plugin zip when present; writes `target/certification/artifacts.json` with `{name, path, sha256, size, commit, status, detail}` per artifact, recording exact errors and retry commands for anything not produced. |
| `node scripts/install-matrix.mjs` | matrix (part of `full`) | Installs/verifies every built artifact on this host into clean temp prefixes: daemon extraction + `faktor-cli doctor --data-dir <tmp>`, VSIX zip/manifest structure, JetBrains `plugin.xml` id/version; writes `target/certification/install-matrix.json`. Non-zero on any verification failure. |
| `TAMPER=1 node scripts/install-matrix.mjs` | matrix self-test | Copies a built artifact, flips one byte, and requires the verifier to reject the copy (sha256 mismatch); exits 0 only on rejection. Evidence: `target/certification/install-matrix-tamper.json`. |
| `bash scripts/certify.sh` | legacy wrapper | The older 8-gate release wrapper; `certify-local.sh full` supersedes it with the manifest. Kept for compatibility. |

`FAST_TESTS_SKIP=1` is a **dry-run aid only**: it records `workspace-tests`
as a skipped section with the reason instead of running it. A manifest with
the tests skipped is never release-certified (see §5).

### Fast section order (fail-fast)

1. `cargo fmt --check`
2. `cargo check --workspace`
3. `cargo clippy --workspace --all-targets -- -D warnings`
4. `cargo test --workspace` (wrapped in `caffeinate -i` on macOS)
5. static-authority scans (`faktor-tests-static-authority`)
6. fault campaign smoke (`faktor-tests-fault`, non-ignored)
7. `doctor --deep` on a fresh temp data dir
8. branding scan (`scripts/branding-scan.sh`, plus packaged artifacts when present)
9. release CLI build + `doctor --deep` on an empty data dir

### Full adds (after fast, same order)

10. `[perf]` release distribution gates (`faktor-tests-performance --release -- --ignored`)
11. `[fault]` campaigns at scale (`faktor-tests-fault --release -- --ignored`)
12. coding-benchmark smoke (`faktor-tests-coding-benchmark --test smoke`)
13. efficiency harness (`faktor-tests-efficiency`)
14. ACP interop (`faktor-acp --test interop`)
15. artifact packaging (`scripts/package-artifacts.sh` → `artifacts.json`)
16. installation matrix (`node scripts/install-matrix.mjs` → `install-matrix.json`)

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

CI (`.github/workflows/ci.yml` and `nightly.yml`) runs exactly these lanes:

| Lane | Runner | Content |
| --- | --- | --- |
| `pr-lane` | ubuntu | fmt; clippy `--all-features` `-D warnings`; tests `--all-features`; `cargo doc`; branding scan; docs-sync guard; VS Code extension build; JetBrains `compile-and-smoke.sh` |
| `linux` / `macos` | ubuntu / macos | fmt; check; tests; clippy `-D warnings`; `doctor` smoke |
| `windows` | windows | `cargo check --workspace`; tests for the process-tree and platform crates (unix-only tests are `cfg(unix)`-gated) |
| `fuzz-suite` | ubuntu | protocol/provider stream codecs |
| `perf` | ubuntu | release `[perf]` gates |
| `nightly: fault` | ubuntu | `[fault]` campaign + `doctor --deep` on a clean data dir |
| `nightly: soak` | ubuntu | accelerated 12h-scale simulation (`SOAK_SCALE=1`) |
| `nightly: benchmark` | ubuntu | release economy + benchmark suites |
| `nightly: certify-local-full` | ubuntu | `bash scripts/certify-local.sh full` + manifest artifact |

100% requires the PR and platform lanes green at the exact commit. The
release real-time soak (12–24h wall clock) is a self-hosted hook and is
deliberately **not** part of the default nightly; when required for a
release it must be run and recorded separately.

### 2.2 UI builds and parity

- **VS Code** (`apps/vscode`): `npm ci && npm run build` green in CI, and
  the wire harness (`bash scripts/run-vscode-harness.sh`, which drives
  `apps/vscode/harness/client.mjs` against a real `faktor-cli` binary)
  green. The derived client shell is **IMPLEMENTED**; byte-for-byte parity
  with the v7.5.6 webview is **BLOCKED_EXTERNAL** because the upstream
  webview/CSS/images are not vendored in this repository.
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
  verification and evidence. The upstream 7.1.2 UI sources are still not
  vendored, so 7.1.2 UI parity remains **BLOCKED_EXTERNAL**. Only the
  2024.1.7 distribution was verified; `until-build` stays unbounded, so
  newer-platform compatibility is not claimed.

100% requires the builds and smokes green **and the capability manifest
labels honest**. It does not require byte-for-byte UI parity while the
upstream assets are absent, but no certificate may claim parity that is
`BLOCKED_EXTERNAL`. The manifest's `capabilities.ui_parity` field carries
the current labels.

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
keys and are **always recorded as skipped** by the local profile. For a
release that claims verified real-model economics, one such run must be
performed and attached separately; the offline certificate never implies it.

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
  into a real IDE sandbox need an IDE host. The CI `pr-lane` owns those
  steps; the host matrix only proves the archives are structurally
  installable and that the daemon actually runs from an extracted bundle.
  The report's `residual[]` records both items.

The offline contract still holds: packaging may *attempt* the npm registry
for VSIX tooling, but the certificate never depends on that attempt
succeeding — a failure is recorded as a skip with the exact error.

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
| Perf distributions | release `[perf]` gates | `full` profile / CI `perf` lane | CI-LANE |
| Fault at scale | `[fault] --ignored` campaigns | `full` profile / nightly `fault` job | CI-LANE |
| Coding benchmark (harness) | `smoke` suite, offline | `full` profile | CERTIFIED in `full` only |
| Coding benchmark (real model) | `real --ignored`, provider keys | manual, keyed | NOT RUN HERE (recorded skip) |
| Efficiency | KPI harness | `full` profile | CERTIFIED in `full` only |
| ACP interop | `faktor-acp --test interop` | `full` profile / CI windows+pr lanes | CERTIFIED in `full` only |
| Installable artifacts (host) | daemon tar.gz + VSIX + JetBrains zip + `artifacts.json` | `full` profile (`scripts/package-artifacts.sh`, §2.9) | CERTIFIED per full run (recorded skips with exact errors when a tool/registry is absent) |
| Installation matrix (host) | clean-prefix extract + `doctor`; VSIX/zip structure + entry points | `full` profile (`node scripts/install-matrix.mjs` → `install-matrix.json`, §2.9) | CERTIFIED per full run |
| IDE-launched install | `code --install-extension` / JetBrains sandbox install | requires an IDE host; CI `pr-lane` owns it | NOT RUN HERE (residual, recorded in `install-matrix.json`) |
| Windows lane | check + process-tree crate tests | CI `windows` job | CI-LANE |
| Linux lane | fmt/check/test/clippy/doctor | CI `linux` job | CI-LANE |
| VS Code shell build | `npm ci && npm run build` + wire harness | CI `pr-lane` | CI-LANE (shell IMPLEMENTED) |
| VS Code byte parity | v7.5.6 webview/CSS/images | upstream assets not vendored | BLOCKED_EXTERNAL |
| JetBrains bridge | kotlinc `compile-and-smoke.sh` (wire + native smokes); Gradle plugin build + verifier vs IC-2024.1.7 | CI `pr-lane` / local script; §3.2 | CI-LANE (native bridge IMPLEMENTED; plugin verifier PASS locally 2026-09-10) |
| JetBrains 7.1.2 UI parity | frozen 7.1.2 sources | not vendored | BLOCKED_EXTERNAL |
| Compat fixtures v756 | golden suite + fixtures | `tests/compat` | CI-LANE / fast tests |
| Compat fixtures jetbrains-712 | reserved corpus | absent (`false` in manifest) | BLOCKED_EXTERNAL |
| Fuzz harnesses | seeded pseudo-fuzz | CI `fuzz-suite` / manual | CI-LANE |
| Real-time soak (12–24h) | wall-clock soak | self-hosted hook (disabled by default) | NOT RUN HERE |

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

---

## 4. Manifest schema (`target/certification/manifest.json`)

```json
{
  "schema": "faktor-certification-manifest/v1",
  "profile": "fast",
  "status": "pass",
  "release_certified": false,
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
    "ui_parity": {"vscode": "...", "jetbrains": "..."},
    "compat_fixtures": {"v756": true, "jetbrains712": false},
    "surfaces": {"workspace_tests": true, "...": false},
    "offline": {"network_required": false, "provider_keys_required": false},
    "release_rule": "a release is certified only for its exact commit with dirty=false"
  }
}
```

Field semantics:

| Field | Meaning |
| --- | --- |
| `commit` | `git rev-parse HEAD` at run start; evidence is bound to this SHA only |
| `dirty_count` | `git status --porcelain` line count; any non-zero invalidates release certification |
| `status` | `pass` iff every attempted section passed; `fail` otherwise |
| `release_certified` | `true` only for `full` + `status=pass` + `dirty_count=0` + `fast_tests_skipped=false` |
| `sections[].status` | `pass` or `fail`; failed sections carry the first error line in `detail` |
| `sections[].duration_ms` | wall time of that section |
| `skipped[]` | sections not attempted, each with the exact reason (profile, fail-fast, offline contract, platform) |
| `capabilities` | capability manifest for this host/profile: platform lanes, UI parity labels, compat fixture presence, surface pass flags, offline contract, release rule |

Per-section logs live in `target/certification/logs/<name>.log`.

Sibling evidence written by the `full` profile (never a substitute for the
certificate manifest): `target/certification/artifacts.json` (packaging
manifest, §2.9), `target/certification/install-matrix.json` (host
installation matrix, §2.9) and `target/certification/install-matrix-tamper.json`
(the `TAMPER=1` self-test evidence).

---

## 5. Release certification rule

> **A release is certified only for its exact commit with `dirty_count = 0`.**

Concretely, to ship:

1. `git status --porcelain` is empty (a dirty tree can never be certified).
2. `bash scripts/certify-local.sh full` on the release host ends with
   `CERTIFICATION: PASS` and `"release_certified": true` in the manifest.
3. Every CI lane in §2.1 is green for the same commit SHA (platform lanes
   the local host cannot run).
4. The manifest's `capabilities` labels are honest: `BLOCKED_EXTERNAL` and
   `PARTIAL` surfaces are carried into the release notes; no parity claim is
   made for unvendored assets.
5. Any real-model benchmark or wall-clock soak claim is backed by its own
   attached evidence; the offline certificate never implies one.

Any new commit — including a docs-only change — invalidates the previous
certificate and requires a fresh run.
