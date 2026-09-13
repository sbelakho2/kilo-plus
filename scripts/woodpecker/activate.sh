#!/usr/bin/env bash
# Activate the Faktor repository on a Woodpecker 3.x instance and register the
# trusted persistence the workflow files expect.
#
# What it does (all calls verified against Woodpecker v3.18.0 source):
#   1. GET  /api/user                                  token check
#   2. POST /api/repos?forge_remote_id=<forge id>       activate (409 = already active)
#      GET  /api/repos/lookup/<owner>/<repo>           resolve repo id/settings
#   3. PATCH /api/repos/<repo_id>                       set the pipeline timeout (opt-in)
#   4. POST|PATCH /api/repos/<repo_id>/secrets[/<name>] set secrets (none by default)
#   5. GET/POST/PATCH /api/repos/<repo_id>/cron         register `nightly` + `soak`
#   6. POST /api/repos/<repo_id>/cron/<cron_id>         optional --run-now
#   7. prints the branch-protection setup for the commit-status contexts
#
# Required environment:
#   WOODPECKER_HOST   base URL of the instance, e.g. https://ci.example.org
#   WOODPECKER_TOKEN  personal access token (Woodpecker UI -> user settings)
#
# Usage:
#   WOODPECKER_HOST=... WOODPECKER_TOKEN=... \
#     bash scripts/woodpecker/activate.sh [owner/repo] [options]
#
# Options:
#   --run-now                 trigger `nightly` and `soak` right after registration
#   --timeout-minutes N       set the repo pipeline timeout (needs server
#                             WOODPECKER_MAX_PIPELINE_TIMEOUT >= N; 0 = leave as is)
#   --trusted                 request trusted.volumes (instance-admin token only;
#                             needed by trusted.yaml/nightly.yaml/soak.yaml caches)
#   --secret NAME=VALUE       create/update a repo secret (repeatable; none required)
#   --dry-run                 print the API calls without sending them
#   -h, --help                this text
#
# Defaults:
#   cron `nightly`: 0 3 * * *   branch main, UTC, enabled
#   cron `soak`:    0 4 * * 6   branch main, UTC, enabled
#
# The UI equivalents are in setup.md (§3 activation, §4 secrets, §6 cron);
# if the API path differs on your instance, setup.md documents the exact UI
# steps that reach the same state.
set -uo pipefail

HOST="${WOODPECKER_HOST:-}"
TOKEN="${WOODPECKER_TOKEN:-}"
DRY_RUN=0
RUN_NOW=0
TRUSTED=0
TIMEOUT_MINUTES=0
REPO_FULL_NAME=""
declare -a SECRETS=()

usage() { sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; }

die() { echo "activate: $*" >&2; exit 1; }
note() { echo "activate: $*"; }
warn() { echo "activate: WARNING: $*" >&2; }

while [ $# -gt 0 ]; do
    case "$1" in
    -h | --help)
        usage
        exit 0
        ;;
    --dry-run)
        DRY_RUN=1
        shift
        ;;
    --run-now)
        RUN_NOW=1
        shift
        ;;
    --trusted)
        TRUSTED=1
        shift
        ;;
    --timeout-minutes)
        [ $# -ge 2 ] || die "--timeout-minutes needs a value"
        TIMEOUT_MINUTES="$2"
        shift 2
        ;;
    --secret)
        [ $# -ge 2 ] || die "--secret needs NAME=VALUE"
        SECRETS+=("$2")
        shift 2
        ;;
    --*)
        die "unknown option: $1 (try --help)"
        ;;
    *)
        [ -z "$REPO_FULL_NAME" ] || die "unexpected argument: $1"
        REPO_FULL_NAME="$1"
        shift
        ;;
    esac
done

[ -n "$HOST" ] || die "WOODPECKER_HOST is required (e.g. https://ci.example.org)"
[ -n "$TOKEN" ] || die "WOODPECKER_TOKEN is required (personal access token)"
HOST="${HOST%/}"
API="${HOST}/api"
command -v curl >/dev/null 2>&1 || die "curl is required"
command -v python3 >/dev/null 2>&1 || die "python3 is required (JSON parsing)"

if [ -z "$REPO_FULL_NAME" ]; then
    remote="$(git remote get-url origin 2>/dev/null || true)"
    case "$remote" in
    *github.com[:/]*)
        REPO_FULL_NAME="$(printf '%s' "$remote" | sed -E 's#.*github\.com[:/]([^/]+/[^/]+?)(\.git)?$#\1#')"
        ;;
    *://*/*|*@*:*)
        host_path="${remote#*://}"
        case "$remote" in
        *@*:*) host_path="${remote#*@}"; host_path="${host_path/:/\/}" ;;
        esac
        REPO_FULL_NAME="$(printf '%s' "$host_path" | sed -E 's#^[^/]+/##; s#\.git$##; s#/+$##')"
        ;;
    esac
fi
[ -n "$REPO_FULL_NAME" ] || die "pass the repository as owner/repo (could not derive it from 'git remote get-url origin')"
case "$REPO_FULL_NAME" in
*/*) ;;
*) die "repository must be owner/repo (got '$REPO_FULL_NAME')" ;;
esac
OWNER="${REPO_FULL_NAME%%/*}"
REPO="${REPO_FULL_NAME##*/}"

case "$TIMEOUT_MINUTES" in
'' | *[!0-9]*) die "--timeout-minutes must be a non-negative integer" ;;
esac

RESPONSE_BODY=""
RESPONSE_STATUS=""
request() {
    local method="$1" path="$2" data="${3:-}" url status body_file
    url="${API}${path}"
    if [ "$DRY_RUN" -eq 1 ]; then
        note "DRY-RUN ${method} ${url}${data:+ ${data}}"
        RESPONSE_STATUS="200"
        RESPONSE_BODY="{}"
        return 0
    fi
    body_file="$(mktemp)"
    if [ -n "$data" ]; then
        status="$(curl -sS -o "$body_file" -w '%{http_code}' -X "$method" \
            -H "Authorization: Bearer ${TOKEN}" -H 'Content-Type: application/json' \
            --data "$data" "$url" || echo "000")"
    else
        status="$(curl -sS -o "$body_file" -w '%{http_code}' -X "$method" \
            -H "Authorization: Bearer ${TOKEN}" "$url" || echo "000")"
    fi
    RESPONSE_BODY="$(cat "$body_file")"
    RESPONSE_STATUS="$status"
    rm -f "$body_file"
    return 0
}

# json_field PATH [DEFAULT] -- reads RESPONSE_BODY.
json_field() {
    local path="$1" default="${2:-}"
    printf '%s' "$RESPONSE_BODY" | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print(sys.argv[2])
    sys.exit(0)
for key in sys.argv[1].split("."):
    if key == "":
        continue
    if isinstance(data, list):
        data = data[int(key)]
    else:
        data = data.get(key)
    if data is None:
        print(sys.argv[2])
        sys.exit(0)
print(data if not isinstance(data, (dict, list)) else json.dumps(data))
' "$path" "$default"
}

# Wait for the forge to list the repository; a fresh instance may not have
# synced it yet.
lookup_repo() {
    request GET "/repos/lookup/${OWNER}/${REPO}"
    [ "$RESPONSE_STATUS" = "200" ]
}

note "instance : ${HOST}"
note "repo     : ${REPO_FULL_NAME}"

request GET "/user"
case "$RESPONSE_STATUS" in
200) note "token    : ok (user $(json_field login unknown))" ;;
401 | 403) die "token rejected by ${HOST} (${RESPONSE_STATUS}); check WOODPECKER_TOKEN" ;;
*) warn "GET /api/user returned ${RESPONSE_STATUS}; continuing" ;;
esac

REPO_ID=""
if [ "$DRY_RUN" -eq 1 ]; then
    lookup_repo || true
    REPO_ID="<repo-id>"
    note "lookup   : DRY-RUN (skipping real id resolution)"
elif lookup_repo; then
    REPO_ID="$(json_field id)"
    note "lookup   : found (id=${REPO_ID}, active=$(json_field active), trusted.volumes=$(json_field trusted.volumes))"
else
    FORGE_REMOTE_ID="${FORGE_REMOTE_ID:-}"
    if [ -z "$FORGE_REMOTE_ID" ] && command -v gh >/dev/null 2>&1; then
        FORGE_REMOTE_ID="$(gh api "repos/${REPO_FULL_NAME}" --jq .id 2>/dev/null || true)"
    fi
    [ -n "$FORGE_REMOTE_ID" ] || die "repo not found in Woodpecker and no forge remote id (set FORGE_REMOTE_ID or install gh); log into ${HOST} once so the forge sync can list the repository, then re-run"
    note "activate : POST /repos?forge_remote_id=${FORGE_REMOTE_ID}"
    request POST "/repos?forge_remote_id=${FORGE_REMOTE_ID}"
    case "$RESPONSE_STATUS" in
    200 | 201) note "activate : ok" ;;
    409) note "activate : already active (409)" ;;
    *) die "activation failed (${RESPONSE_STATUS}): ${RESPONSE_BODY}" ;;
    esac
    lookup_repo || die "repo still not resolvable after activation; check forge access, then re-run"
    REPO_ID="$(json_field id)"
fi
[ -n "$REPO_ID" ] && [ "$REPO_ID" != "None" ] || die "could not resolve the Woodpecker repository id"
note "repo id  : ${REPO_ID}"

# ---------------------------------------------------------------- settings --
if [ "$TIMEOUT_MINUTES" -gt 0 ]; then
    note "settings : timeout=${TIMEOUT_MINUTES} min"
    request PATCH "/repos/${REPO_ID}" "{\"timeout\":${TIMEOUT_MINUTES}}"
    case "$RESPONSE_STATUS" in
    200) note "settings : timeout set" ;;
    403) warn "timeout ${TIMEOUT_MINUTES} refused: server WOODPECKER_MAX_PIPELINE_TIMEOUT caps it (hosted instances are admin-fixed); soak/longrun cannot run at 24h until the cap is raised" ;;
    *) warn "timeout update returned ${RESPONSE_STATUS}: ${RESPONSE_BODY}" ;;
    esac
else
    note "settings : timeout left untouched (pass --timeout-minutes 1560 for the 24h soak)"
fi

if [ "$TRUSTED" -eq 1 ]; then
    note "trusted  : requesting trusted.volumes (needs an instance-admin token)"
    request PATCH "/repos/${REPO_ID}" '{"trusted":{"volumes":true}}'
    case "$RESPONSE_STATUS" in
    200) note "trusted  : granted (named-volume caches usable by trusted.yaml/nightly.yaml/soak.yaml)" ;;
    403) warn "trusted.volumes refused (not an instance admin); either strip volumes: from the trusted/cron workflows or ask a server admin (setup.md §5)" ;;
    *) warn "trusted update returned ${RESPONSE_STATUS}: ${RESPONSE_BODY}" ;;
    esac
else
    note "trusted  : unchanged (pass --trusted to request trusted.volumes as an admin)"
fi

# ----------------------------------------------------------------- secrets --
# None are required by default; provider-key runs stay recorded skips until
# an operator explicitly passes --secret.
if [ "${#SECRETS[@]}" -eq 0 ]; then
    note "secrets  : none passed (correct default; CI requires no secrets)"
else
    for entry in "${SECRETS[@]}"; do
        case "$entry" in
        *=*) ;;
        *) die "--secret expects NAME=VALUE (got '$entry')" ;;
        esac
        name="${entry%%=*}"
        value="${entry#*=}"
        case "$name" in
        '' | *[!A-Za-z0-9._-]*) die "secret name '$name' must match [A-Za-z0-9._-]+" ;;
        esac
        note "secrets  : ${name}"
        request GET "/repos/${REPO_ID}/secrets/${name}"
        if [ "$RESPONSE_STATUS" = "200" ]; then
            request PATCH "/repos/${REPO_ID}/secrets/${name}" "$(python3 -c 'import json,sys; print(json.dumps({"value": sys.argv[1]}))' "$value")"
        else
            request POST "/repos/${REPO_ID}/secrets" "$(python3 -c 'import json,sys; print(json.dumps({"name": sys.argv[1], "value": sys.argv[2]}))' "$name" "$value")"
        fi
        case "$RESPONSE_STATUS" in
        200 | 201) note "secrets  : ${name} set" ;;
        *) warn "secret ${name} returned ${RESPONSE_STATUS}: ${RESPONSE_BODY}" ;;
        esac
    done
fi

# ------------------------------------------------------------------- crons --
# The cron job name must match the `when.cron` filter of the workflow file.
CRON_SPECS=(
    "nightly|0 3 * * *|main|UTC"
    "soak|0 4 * * 6|main|UTC"
)
CRON_IDS=()
for spec in "${CRON_SPECS[@]}"; do
    IFS='|' read -r cron_name cron_schedule cron_branch cron_timezone <<<"$spec"
    cron_body="$(python3 -c 'import json,sys; print(json.dumps({"name": sys.argv[1], "schedule": sys.argv[2], "branch": sys.argv[3], "timezone": sys.argv[4], "enabled": True}))' \
        "$cron_name" "$cron_schedule" "$cron_branch" "$cron_timezone")"
    request GET "/repos/${REPO_ID}/cron"
    cron_id=""
    if [ "$RESPONSE_STATUS" = "200" ]; then
        cron_id="$(printf '%s' "$RESPONSE_BODY" | python3 -c '
import json, sys
name = sys.argv[1]
try:
    jobs = json.load(sys.stdin)
except Exception:
    jobs = []
for job in jobs if isinstance(jobs, list) else []:
    if job.get("name") == name:
        print(job.get("id", ""))
        break
' "$cron_name")"
    fi
    if [ -n "$cron_id" ]; then
        note "cron     : ${cron_name} exists (id=${cron_id}); patching schedule/branch"
        request PATCH "/repos/${REPO_ID}/cron/${cron_id}" "$cron_body"
        case "$RESPONSE_STATUS" in
        200) note "cron     : ${cron_name} updated (${cron_schedule} ${cron_timezone})" ;;
        *) warn "cron ${cron_name} patch returned ${RESPONSE_STATUS}: ${RESPONSE_BODY}" ;;
        esac
    else
        note "cron     : registering ${cron_name} (${cron_schedule} ${cron_timezone})"
        request POST "/repos/${REPO_ID}/cron" "$cron_body"
        case "$RESPONSE_STATUS" in
        200 | 201)
            cron_id="$(json_field id)"
            note "cron     : ${cron_name} registered (id=${cron_id})"
            ;;
        409)
            warn "cron ${cron_name} already exists (409) but was not listed; check the UI"
            ;;
        *) warn "cron ${cron_name} returned ${RESPONSE_STATUS}: ${RESPONSE_BODY}" ;;
        esac
    fi
    [ -n "$cron_id" ] && CRON_IDS+=("${cron_name}=${cron_id}")
done

if [ "$RUN_NOW" -eq 1 ]; then
    for pair in "${CRON_IDS[@]:-}"; do
        [ -n "$pair" ] || continue
        cron_name="${pair%%=*}"
        cron_id="${pair#*=}"
        note "run-now  : ${cron_name}"
        request POST "/repos/${REPO_ID}/cron/${cron_id}"
        case "$RESPONSE_STATUS" in
        200) note "run-now  : ${cron_name} pipeline #$(json_field number '?') created" ;;
        *) warn "run-now ${cron_name} returned ${RESPONSE_STATUS}: ${RESPONSE_BODY}" ;;
        esac
    done
fi

# -------------------------------------------------------- branch protection --
# Woodpecker reports one commit status per workflow; the default context
# format is "{{ .context }}/{{ .event }}/{{ .workflow }}". With the default
# WOODPECKER_STATUS_CONTEXT=ci/woodpecker the contexts are:
#   pull_request  -> ci/woodpecker/pr/pr          (the required PR gate)
#   push main     -> ci/woodpecker/push/trusted   (post-merge evidence)
#   tag           -> ci/woodpecker/tag/trusted
#   cron          -> ci/woodpecker/cron/nightly | ci/woodpecker/cron/soak
cat <<EOF

Next steps (this script does not change branch protection):
  * GitHub UI: Settings -> Branches -> Add branch protection rule for main
    -> Require status checks to pass -> add ci/woodpecker/pr/pr.
  * Or with gh (adjust owner/repo if needed):
      gh api --method PUT repos/${REPO_FULL_NAME}/branches/main/protection --input - <<'JSON'
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
  * Forge webhook: activation installed it; confirm it delivers push, tag and
    pull_request events, and keep "Require approval for forked repositories"
    enabled (Woodpecker default) as documented in setup.md §5.
  * Statuses are per workflow: a failing PR lane fails ci/woodpecker/pr/pr
    because the certificate step inside .woodpecker/pr.yaml fails closed.
    If the server overrides WOODPECKER_STATUS_CONTEXT(_FORMAT), substitute the
    resulting strings.
EOF
note "done"
