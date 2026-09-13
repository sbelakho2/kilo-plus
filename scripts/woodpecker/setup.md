# Woodpecker CI setup

The CI for this repository is defined by the workflow files in
[`.woodpecker/`](../../.woodpecker) and runs on
[Woodpecker CI](https://woodpecker-ci.org). This directory holds a local
server/agent stack (`docker-compose.yml`), the activation script
(`activate.sh`) and the operator notes below.

Targeted version: **Woodpecker 3.x** (verified against the 3.18 documentation
and the v3.18.0 source; the syntax also validates against the 2.8 subset it
uses: matrix, labels, steps, workflow-level `when`, `depends_on`, status
filters, cron filters).

## 1. Pipeline model

By default Woodpecker resolves pipeline configs in this order:
`.woodpecker/*.{yaml,yml}` -> `.woodpecker.yaml` -> `.woodpecker.yml`
(project settings -> *Pipeline path* empty). This repository therefore has
**no root `.woodpecker.yml`**: the folder files below are the single source
of truth, and each file is one workflow with its own `when` filter and its own
commit status.

| File (workflow) | Event filter | Storage | Contents |
| --- | --- | --- | --- |
| `.woodpecker/pr.yaml` (`pr`) | `pull_request` | **no volumes at all** | reduced lane set: `storage-policy`, `linux`, `static`, `docs`, `vscode`, `vscode-visual`, `vscode-visual-with-skip`, `jetbrains-build`, `jetbrains-smoke`, then the aggregate `certificate` |
| `.woodpecker/trusted.yaml` (`trusted`) | `push` (any branch) + `tag` (linux); `push` to `main` for darwin/windows | trusted named volumes `faktor-trusted-*` | full linux lane set incl. release `[perf]` + `certificate`; darwin/windows matrix combos + per-platform certificates |
| `.woodpecker/nightly.yaml` (`nightly`) | `cron` job `nightly` | own `faktor-nightly-*` volumes | `[fault]` at scale, longrun, efficiency, economy, coding-benchmark smoke, provider-key real-model run (recorded skip by default), supply-chain, then `certificate-nightly` |
| `.woodpecker/soak.yaml` (`soak`) | `cron` job `soak` | own `faktor-soak-*` volumes | `[soak]` 12h synthetic session + 24h zero-drift wall-clock run, then `certificate-soak` |

`labels: platform: ${platform}` (trusted) or `labels: platform: linux/amd64`
(pr/cron) routes workflows to agents. The deprecated `runs_on` key is **not**
an agent selector in Woodpecker 2.x/3.x: it feeds the run-on-success/failure
status of dependent tasks, and `runs_on: [macos]` would make the workflow
match no status and never run. Agent selection is `labels`.

The darwin and windows combos are gated to `push` on `main` (change
`branch: main` in `.woodpecker/trusted.yaml` if your default branch differs).

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
linux docker agent. It also raises `WOODPECKER_MAX_PIPELINE_TIMEOUT` (default
1560 minutes / 26h) so the `soak` job can be admitted; see §6. Port 9000 must
be reachable from the macOS/Windows agent hosts and should be firewalled
otherwise (only the shared agent secret crosses it). Server state survives
`docker compose down`; use `down -v` only to reset it.

Create a GitHub OAuth application for the forge connection (Settings →
Developer settings → OAuth Apps):

- Homepage URL: `$WOODPECKER_HOST`
- Authorization callback URL: `$WOODPECKER_HOST/authorize`

## 3. Activate the repository

**Forge webhook and commit-status behavior.** Activating a repository in
Woodpecker installs the forge webhook and starts processing its events:
`push`, `tag` and `pull_request` (the webhook for `pull_request` is only
honored while *Allow pull requests* is enabled, which is the default).
Pipelines are triggered per event and each **workflow file** posts its own
commit status; with the default server settings (`WOODPECKER_STATUS_CONTEXT`
= `ci/woodpecker`, `WOODPECKER_STATUS_CONTEXT_FORMAT` =
`{{ .context }}/{{ .event }}/{{ .workflow }}`) the contexts are:

| Event | Context |
| --- | --- |
| pull request | `ci/woodpecker/pr/pr` |
| push to main (any branch) | `ci/woodpecker/push/trusted` |
| tag | `ci/woodpecker/tag/trusted` |
| cron | `ci/woodpecker/cron/nightly`, `ci/woodpecker/cron/soak` |

A workflow excluded by its `when` filter posts nothing. Server overrides of
`WOODPECKER_STATUS_CONTEXT(_FORMAT)` rename every context; use the resulting
strings in branch protection.

### 3a. UI steps (no account scripts required)

1. Log into Woodpecker with an admin/owner account and enable the repository
   (*Repositories → Add*).
2. Leave the **pipeline path** empty: the default resolution finds the
   `.woodpecker/` folder (it takes precedence over any root file; this repo
   has none).
3. Confirm the webhook was installed (forge → repository → Webhooks) and that
   *Allow pull requests* is enabled in the Woodpecker project settings.
4. Continue with §5 (trust) and §6 (cron).

### 3b. Scripted activation (`activate.sh`)

```sh
export WOODPECKER_HOST="https://<your-instance>"     # no trailing slash
export WOODPECKER_TOKEN="<personal access token>"    # Woodpecker UI -> user settings
bash scripts/woodpecker/activate.sh <owner/repo> \
  --timeout-minutes 1560 \
  --trusted                 # instance admin only; only for the named-volume caches
```

The script is idempotent and uses the Woodpecker 3.x API (all paths verified
against v3.18.0):

- `GET /api/user` — token check;
- `POST /api/repos?forge_remote_id=<id>` — activate (409 = already active);
  `<id>` comes from `GET /api/repos/lookup/<owner>/<repo>` when the repo is
  already known, otherwise from `gh api repos/<owner>/<repo> --jq .id` or the
  `FORGE_REMOTE_ID` environment variable;
- `PATCH /api/repos/<repo_id> {"timeout":<minutes>}` — the pipeline timeout
  (needed by `soak`; see §6);
- `PATCH /api/repos/<repo_id> {"trusted":{"volumes":true}}` — trusted volumes
  (instance-admin token only, `--trusted`);
- `POST|PATCH /api/repos/<repo_id>/secrets[/<name>]` — secrets (`--secret
  NAME=VALUE`, repeatable; **none are required by default**);
- `GET|POST|PATCH /api/repos/<repo_id>/cron` — register/patch the `nightly`
  and `soak` cron jobs (§6);
- `--run-now` triggers both jobs once after registration;
- the script prints the branch-protection setup for `ci/woodpecker/pr/pr`
  (§7). Use `--dry-run` to print every call without sending it.

### 3c. Activation status in this checkout

**Activation could not be executed from this machine:** there is no
`WOODPECKER_HOST`/`WOODPECKER_TOKEN` in the environment, no Woodpecker
instance reachable on `localhost:8000`, and no server credentials of any kind
were available. Nothing in this repository has been enabled on an instance by
this change. The exact commands to run somewhere with credentials are:

```sh
export WOODPECKER_HOST="https://<your-instance>"
export WOODPECKER_TOKEN="<personal access token>"
bash scripts/woodpecker/activate.sh <owner/repo> --timeout-minutes 1560 --trusted --run-now
# then verify:
curl -fsS -H "Authorization: Bearer $WOODPECKER_TOKEN" \
  "$WOODPECKER_HOST/api/repos/lookup/<owner>/<repo>" | python3 -m json.tool
curl -fsS -H "Authorization: Bearer $WOODPECKER_TOKEN" \
  "$WOODPECKER_HOST/api/repos/<repo_id>/cron" | python3 -m json.tool
```

If the API path differs on a future Woodpecker minor, use the UI equivalents
(Project settings, Cron Jobs) — §3a, §4 and §6 give the exact fields.

## 4. Secrets

**None are required for the default pipelines.** All gates are deterministic
and offline-capable apart from image/package downloads (Rust crates, npm
packages, Gradle distribution/toolchain downloads, playwright browsers).
Provider-key (real-model) runs are deliberately not part of the PR/trusted
pipelines; the nightly `coding-benchmark-real-model` lane records an explicit
skip marker unless `FAKTOR_BENCH_PROVIDER`, `FAKTOR_BENCH_MODEL` and
`FAKTOR_BENCH_API_KEY` are present in the step environment. Note that a
`from_secret` reference to a secret that does not exist is a config compile
error in Woodpecker, so the workflow does not reference secrets blindly. To
enable real nightly runs, either provide those variables to the agent's step
environment (self-hosted) or register repo secrets with
`activate.sh --secret NAME=VALUE` and add explicit
`environment: {FAKTOR_BENCH_API_KEY: {from_secret: ...}}` entries to the lane
in `.woodpecker/nightly.yaml`.

## 5. Volumes, trust, and the PR/trusted storage policy

Woodpecker only allows `volumes:` when a **server admin marks the repository
Trusted (volumes)** (Project settings → Trusted; the *Trusted* section is
admin-only). Trust is per repository, **not per event**, so the CI layout is
built around that constraint:

- `.woodpecker/pr.yaml` is the only file a `pull_request` event runs and it
  declares **zero volumes** — PR jobs use the ephemeral per-pipeline workspace
  only. The `storage-policy` step and the PR certificate both fail if a
  `volumes:` key appears in that file.
- `.woodpecker/trusted.yaml` is the only workflow that mounts the
  `faktor-trusted-*` caches, and it only matches `push`/`tag` (collaborator
  events). `.woodpecker/nightly.yaml` and `.woodpecker/soak.yaml` mount their
  own `faktor-nightly-*` / `faktor-soak-*` volumes, so a heavy campaign can
  never corrupt the caches a trusted build reuses.
- Each lane gets its own target volume; the registry/git volumes are shared
  only inside one workflow. Named volumes are agent-host scoped and cannot be
  made branch-specific (Woodpecker does not substitute environment variables
  in `volumes:`), which is exactly why the event-class separation above is the
  isolation boundary.
- If the instance cannot grant trusted status, delete every `volumes:` block
  and `CARGO_TARGET_DIR` entry from `trusted.yaml`, `nightly.yaml` and
  `soak.yaml` (everything still runs, just without warm caches). **Never add
  volumes to `pr.yaml`.**

**Residual risk (must stay documented):** because trust is repository-wide,
a PR that is allowed to run and edits its own pipeline config could add
`volumes:` and Woodpecker would honor it once the repo is trusted. Keep
**Require approval for forked repositories** enabled (the Woodpecker default)
and review `.woodpecker/` changes in PRs. Instances that cannot accept that
should either skip trusted status entirely or also require approval for all
pull requests (Project settings → Require approval for → `pull_requests`).

## 6. Cron jobs (`nightly` and `soak`)

`nightly.yaml`/`soak.yaml` only run for a **cron event whose job name matches
`when.cron`**. Register the jobs in the repository:

- UI: Project settings → **Cron Jobs** → *Add*, with
  - `nightly`: schedule `0 3 * * *`, branch `main`, timezone `UTC`;
  - `soak`: schedule `0 4 * * 6`, branch `main`, timezone `UTC`.
- API/script: `bash scripts/woodpecker/activate.sh <owner/repo> --run-now`
  (idempotent; it creates or patches both jobs and can trigger them once),
  equivalent to
  `POST /api/repos/{repo_id}/cron` with
  `{"name":"nightly","schedule":"0 3 * * *","branch":"main","timezone":"UTC","enabled":true}`
  and the `soak` body with `{"name":"soak","schedule":"0 4 * * 6",...}`.

Supported schedule syntax: standard 5-field cron plus `@daily`, `@weekly`,
`@every 5m`, ... (see the Woodpecker Cron doc).

**Timeout:** Woodpecker has no per-step timeout; pipelines are capped by the
repository timeout (default 60 min; the settable maximum defaults to
`WOODPECKER_MAX_PIPELINE_TIMEOUT` = 120 min). The 24h `soak` lane therefore
needs:

```sh
# server/agent environment (docker-compose.yml sets this by default)
WOODPECKER_MAX_PIPELINE_TIMEOUT=1560
# repository setting
bash scripts/woodpecker/activate.sh <owner/repo> --timeout-minutes 1560
```

On hosted `woodpecker-ci.org` the server cap is fixed and cannot be raised by
a user; run `soak` on a self-hosted instance (or accept that the `longrun-24h`
lane is a recorded non-run) until that changes.

## 7. Branch protection / required status checks

Woodpecker reports one commit status per workflow, so the required check for
PRs is the `pr` workflow context:

- GitHub UI: Settings → Branches → Add branch protection rule for `main` →
  *Require status checks to pass* → search for and add `ci/woodpecker/pr/pr`.
- API:

  ```sh
  gh api --method PUT repos/<owner>/<repo>/branches/main/protection --input - <<'JSON'
  {
    "required_status_checks": {
      "strict": true,
      "contexts": ["ci/woodpecker/pr/pr"]
    },
    "enforce_admins": false,
    "required_pull_request_reviews": null,
    "restrictions": null
  }
  JSON
  ```

`activate.sh` prints the same instructions after activation. The push/tag
contexts (`ci/woodpecker/push/trusted`, `ci/woodpecker/tag/trusted`) can be
required too if you want the post-merge trusted evidence checked before other
work lands; the darwin/windows per-platform certificates run inside that same
`trusted` workflow, so their failure also fails `ci/woodpecker/push/trusted`.

## 8. Linux agent

The compose `woodpecker-agent` registers with
`WOODPECKER_AGENT_LABELS=platform=linux/amd64`. Its default labels are
`hostname`, `platform`, `backend`, `repo=*`; a custom `platform` overrides the
backend-derived one, which is exactly what the workflows need.

- Docker backend requirements: Docker Engine on the agent host, the workspace
  needs ~40 GB free disk for the parallel lanes.
- Recommended sizing: **≥ 8 vCPU, 16 GB RAM, 60 GB disk**. The mandatory
  lanes run in parallel inside one workflow; `perf` is serialized after the
  other Rust lanes so release budget assertions do not race a loaded agent.
  Cron lanes use separate target volumes so their parallel cargo processes
  do not contend.
- To run several pipelines at once raise `WOODPECKER_MAX_WORKFLOWS`; the lane
  set is resource-heavy, so keep it low on shared hosts.

## 9. macOS and Windows agents (self-hosted, `local` backend)

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
   `trusted.yaml` matrix entry. Until an agent exists, the darwin/windows
   combos stay queued (only on `push` to `main`) and no linux job is affected;
   if you never plan to add one, remove those matrix entries and the matching
   jobs from the config.

## 10. Certificate and aggregation

Woodpecker workflows are filesystem-isolated, so cross-workflow file markers
are not possible; the aggregate gate therefore lives **inside** each workflow:

- every lane ends by writing
  `target/certification/lanes/<lane>.json`
  (`{"schema":"faktor-woodpecker-lane/v1","lane":...,"status":"passed","commit":...}`);
- each `certificate`/`certificate-nightly`/`certificate-soak` step
  `depends_on` every lane of its workflow, runs with
  `when.status: [success, failure]` (a dependent is otherwise skipped when a
  dependency fails), and fails when any marker is missing, marked failed,
  written for a foreign commit, or unexpected;
- it additionally checks the runtime's own `CI_PIPELINE_STATUS`, which is
  `failure` when any earlier step failed, and writes
  `target/certification/ci-certification.json`
  (`faktor-ci-certification/v1`, with `workflow` recorded).
- The only accepted non-`passed` marker is `skipped` on
  `coding-benchmark-real-model` in the nightly certificate, and only with a
  non-empty `reason` — a silent skip is a failure.

The darwin and windows combos carry their own `certificate-darwin` /
`certificate-windows` steps with the same marker/status rule. Woodpecker has
no cross-matrix summary job (upstream issue #2886), so the workflow-level
contexts are what branch protection requires (§7).

## 11. Cloud tier

[woodpecker-ci.org](https://woodpecker-ci.org) offers a free cloud tier for
open-source repositories; it provides **linux runners only**, and its
`WOODPECKER_MAX_PIPELINE_TIMEOUT` is fixed. The `pr` workflow (including the
required render gate and the aggregate `certificate`) and the `trusted`
linux lanes run there without any agent setup; the darwin/windows combos need
your own agents as in §9 (or removal of those matrix entries), and `soak`
needs a self-hosted instance because of the timeout cap (§6).

## 12. Local/offline certificate

CI covers the platform lanes; `bash scripts/certify-local.sh fast`
(and `full`) remains the **local/offline certificate** for the host and emits
`target/certification/manifest.json`. Neither replaces the other: see
`docs/certification.md` for the levels and the release rule.

## 13. Linting the configs locally

```sh
woodpecker-cli lint .woodpecker/            # one pass over all workflow files
python3 - <<'PY'                            # dependency-light parse gate
import glob, yaml
for f in sorted(glob.glob(".woodpecker/*.yml") + glob.glob(".woodpecker/*.yaml")):
    yaml.safe_load(open(f))
    print("ok", f)
PY
```
