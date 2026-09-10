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
| `bash scripts/certify-local.sh full` | full | Everything in fast plus the long lanes: release `[perf]` distribution gates, `[fault]` campaigns at scale, coding-benchmark harness smoke, efficiency harness, ACP interop. Longer; offline. |
| `CERTIFY_SELFTEST=force_fail CERTIFY_OUT_DIR=/tmp/cert-selftest bash scripts/certify-local.sh fast` | selftest | Injects a synthetic failing section and proves the harness exits non-zero, records the failure, and fail-fast marks the remainder skipped. Does not touch the real certificate. |
| `cargo test -p faktor-tests-fault --release -- --ignored` | long lane | The full `[fault]` campaigns (also part of `full`). |
| `cargo test -p faktor-tests-performance --release -- --ignored` | long lane | The `[perf]` distribution gates (also part of `full`). |
| `cargo test -p fuzz-targets` | long lane | Seeded pseudo-fuzz harnesses; owned by CI's fuzz lane and manual runs. |
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
  green (`:shared` + `:backend`, real kotlinc, real daemon). Status:
  **PARTIAL** — the frozen 7.1.2 frontend sources are not vendored;
  `PlaceholderFrontend` is the documented drop-in point.

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

`cargo test -p faktor-acp --test interop` must be green, covering the
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
- Fuzz hygiene: `cargo test -p fuzz-targets` (2000 seeded cases per
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
| Windows lane | check + process-tree crate tests | CI `windows` job | CI-LANE |
| Linux lane | fmt/check/test/clippy/doctor | CI `linux` job | CI-LANE |
| VS Code shell build | `npm ci && npm run build` + wire harness | CI `pr-lane` | CI-LANE (shell IMPLEMENTED) |
| VS Code byte parity | v7.5.6 webview/CSS/images | upstream assets not vendored | BLOCKED_EXTERNAL |
| JetBrains shell | kotlinc `compile-and-smoke.sh` | CI `pr-lane` / local script | CI-LANE (PARTIAL) |
| JetBrains frontend | frozen 7.1.2 sources | not vendored | BLOCKED_EXTERNAL |
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
