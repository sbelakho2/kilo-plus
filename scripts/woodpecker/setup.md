# Woodpecker CI setup

The CI for this repository is defined by the root [`.woodpecker.yml`](../../.woodpecker.yml)
and runs on [Woodpecker CI](https://woodpecker-ci.org). This directory holds a
local server/agent stack (`docker-compose.yml`) and the operator notes below.

Targeted version: **Woodpecker 3.x** (syntax verified against the 3.18
documentation and JSON schema). The same file also validates against the 2.8
schema subset it uses (matrix, labels, steps, depends_on, status filters).

## 1. Pipeline model (what `.woodpecker.yml` does)

One root file defines one workflow **per matrix platform**:

| Matrix combo | Jobs (steps) | Where it runs |
| --- | --- | --- |
| `linux/amd64` | `linux`, `static`, `docs`, `vscode`, `vscode-visual`, `vscode-visual-with-skip`, `jetbrains-build`, `jetbrains-smoke`, `perf`, `certificate` | hosted or self-hosted linux agents |
| `darwin/arm64` | `darwin-check`, `darwin-test`, `darwin-doctor`, `certificate-darwin` | self-hosted macOS agent (`local` backend) |
| `windows/amd64` | `windows-check`, `windows-test`, `certificate-windows` | self-hosted Windows agent (`local` backend) |

`labels: platform: ${platform}` routes each combo to an agent with the matching
`platform` label. The old `runs_on` key is **not** an agent selector in
Woodpecker 2.x/3.x: it feeds the run-on-success/run-on-failure status of
dependent tasks, and `runs_on: [macos]` would make the workflow match no
status and never run. Agent selection is `labels`.

The linux combo runs on `push` and `pull_request`; the darwin and windows
combos are gated to `push` on `main` (change `branch: main` in the root
`when:` block if your default branch differs).

## 2. Start the server

```sh
cd scripts/woodpecker
export WOODPECKER_AGENT_SECRET="$(openssl rand -hex 32)"   # shared by server + agents
export WOODPECKER_HOST="http://localhost:8000"             # public URL of this server
export WOODPECKER_GITHUB_CLIENT="<oauth app client id>"
export WOODPECKER_GITHUB_SECRET="<oauth app client secret>"
docker compose up -d
```

The compose file starts `woodpecker-server` (port 8000 for UI/API, port 9000
for the agent gRPC channel, data in the `woodpecker-data` volume) and one
linux docker agent. Port 9000 must be reachable from the macOS/Windows agent
hosts and should be firewalled otherwise (only the shared agent secret
crosses it). Server state survives `docker compose down`; use `down -v` only
to reset it.

Create a GitHub OAuth application for the forge connection (Settings →
Developer settings → OAuth Apps):

- Homepage URL: `$WOODPECKER_HOST`
- Authorization callback URL: `$WOODPECKER_HOST/authorize`

## 3. Activate the repository

1. Log into Woodpecker with the admin account named in
   `WOODPECKER_ADMIN` (optional) and enable the repository.
2. Leave the **pipeline path** empty: the default resolution finds the root
   `.woodpecker.yml` (`.woodpecker/*.yaml` would take precedence over a root
   file if a folder existed, so keep a single root config).
3. Woodpecker installs the webhook automatically; verify a push or PR
   triggers a pipeline.

## 4. Linux agent

The compose `woodpecker-agent` registers with
`WOODPECKER_AGENT_LABELS=platform=linux/amd64`. Its default labels are
`hostname`, `platform`, `backend`, `repo=*`; a custom `platform` overrides the
backend-derived one, which is exactly what the matrix needs.

- Docker backend requirements: Docker Engine on the agent host, the workspace
  needs ~40 GB free disk for the parallel lanes.
- Recommended sizing: **≥ 8 vCPU, 16 GB RAM, 60 GB disk**. The mandatory
  lanes run in parallel inside one workflow; `perf` is serialized after the
  other Rust lanes so release budget assertions do not race a loaded agent.
- To run several pipelines at once raise `WOODPECKER_MAX_WORKFLOWS`; the
  lane set is resource-heavy, so keep it low on shared hosts.

## 5. macOS and Windows agents (self-hosted, `local` backend)

Hosted `woodpecker-ci.org` and most Docker-agent deployments are linux-only.
The `darwin/*` and `windows/*` combos need an agent on that OS:

1. Install the Woodpecker agent binary (GitHub releases, `woodpecker-agent`
   for the host platform) and the required toolchain:
   - macOS: Rust stable + Xcode command line tools (`cargo` on `PATH`).
   - Windows: Rust stable (MSVC toolchain + Visual Studio Build Tools) with
     `cargo` on `PATH`.
2. The `local` backend runs steps directly on the host and needs the clone
   plugin binary (`woodpeckerci/plugin-git`) on `PATH` so the default clone
   step works; install it from its release page and confirm `plugin-git
   --help` succeeds.
3. Start the agent with labels matching the matrix values:

   ```sh
   # macOS (Apple Silicon)
   WOODPECKER_SERVER=http://<server-host>:9000 \
   WOODPECKER_AGENT_SECRET=<same secret as the server> \
   WOODPECKER_BACKEND=local \
   WOODPECKER_MAX_WORKFLOWS=1 \
   WOODPECKER_AGENT_LABELS=platform=darwin/arm64,hostname=faktor-darwin-agent \
   woodpecker-agent
   ```

   ```powershell
   # Windows (PowerShell)
   $env:WOODPECKER_SERVER = "http://<server-host>:9000"
   $env:WOODPECKER_AGENT_SECRET = "<same secret as the server>"
   $env:WOODPECKER_BACKEND = "local"
   $env:WOODPECKER_MAX_WORKFLOWS = "1"
   $env:WOODPECKER_AGENT_LABELS = "platform=windows/amd64,hostname=faktor-windows-agent"
   woodpecker-agent.exe
   ```

4. Intel Macs use `darwin/amd64`; keep the value in sync with the
   `.woodpecker.yml` matrix entry. Until an agent exists, the darwin/windows
   combos stay queued (only on `push` to `main`) and no linux job is affected;
   if you never plan to add one, remove those matrix entries and the matching
   jobs from the config.

## 6. Caching volumes and repository trust

The pipeline mounts named Docker volumes for cargo registry/git, per-lane
target directories, the Gradle home, the npm cache and Playwright browsers.
Woodpecker requires the repository to be marked **Trusted** (volumes) by a
server admin before any `volumes:` entry is accepted. On
`woodpecker-ci.org`, request trusted status for OSS projects; if it cannot be
granted, delete the `volumes:` blocks (and the `CARGO_TARGET_DIR` environment
entries) from `.woodpecker.yml` — everything still runs, just without warm
caches.

## 7. Secrets

**None are required for the default pipelines.** All gates are deterministic
and offline-capable apart from image/package downloads (Rust crates, npm
packages, Gradle distribution/toolchain downloads, playwright browsers).
Provider-key (real-model) runs are deliberately not part of CI; they live in
the offline certificate's recorded skips (see §8).

## 8. Certificate and aggregation

Woodpecker workflows are filesystem-isolated, so cross-workflow file markers
are not possible; the aggregate gate therefore lives **inside** the linux
workflow:

- every lane ends by writing
  `target/certification/lanes/<lane>.json`
  (`{"schema":"faktor-woodpecker-lane/v1","lane":...,"status":"passed","commit":...}`);
- the `certificate` step `depends_on` every linux lane, runs with
  `when.status: [success, failure]` (a dependent is otherwise skipped when a
  dependency fails), and fails when any marker is missing, marked failed,
  written for a foreign commit, or unexpected;
- it additionally checks the runtime's own `CI_PIPELINE_STATUS`, which is
  `failure` when any earlier stage failed, and writes
  `target/certification/ci-certification.json`
  (`faktor-ci-certification/v1`).

The darwin and windows combos carry their own `certificate-darwin` /
`certificate-windows` steps with the same marker/status rule. Woodpecker has
no cross-matrix summary job (upstream issue #2886), so branch protection
should require the `certificate` status (linux) and, when self-hosted
platform agents exist, the two platform certificates.

## 9. Cloud tier

[woodpecker-ci.org](https://woodpecker-ci.org) offers a free cloud tier for
open-source repositories; it provides **linux runners only**. The linux
combo (including the required render gate and the aggregate `certificate`)
runs there without any agent setup. The darwin/windows combos require your
own agents as in §5, or removal of those matrix entries.

## 10. Local/offline certificate

CI covers the platform lanes; `bash scripts/certify-local.sh fast`
(and `full`) remains the **local/offline certificate** for the host and emits
`target/certification/manifest.json`. Neither replaces the other: see
`docs/certification.md` for the levels and the release rule.

## 11. Linting the config locally

```sh
woodpecker-cli lint .woodpecker.yml
```

The CI gates also parse the file with `python3 -c "import yaml,sys;
yaml.safe_load(open('.woodpecker.yml'))"`.
