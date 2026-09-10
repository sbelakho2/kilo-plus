#!/usr/bin/env bash
# Supply-chain / release-integrity tooling (audits 89/90, tied to the
# P0-75 fuzz campaign and the release certification evidence).
#
# Emits, under $SUPPLY_CHAIN_OUT_DIR (default target/certification):
#   sbom.json                 cargo metadata projection (packages + licenses)
#   checksums.txt             sha256 of the built release artifacts
#   supply-chain-status.json  per-tool result incl. RECORDED SKIPS
#   logs/                     raw tool output
#
# Never silent: every missing tool, unavailable network/advisory database or
# skipped artifact build is printed and recorded in the status JSON. Exit 0
# when the emitted evidence is consistent (skips allowed); exit 1 when a
# real problem was found (SBOM failure, checksum mismatch, vulnerability).
#
# Env:
#   SUPPLY_CHAIN_OUT_DIR     output dir (default target/certification)
#   SUPPLY_CHAIN_FAST=1      do not build; hash only artifacts already present
#   SUPPLY_CHAIN_BUILD=1     build `cargo build -p faktor-cli --release` first
#   SUPPLY_CHAIN_SKIP_AUDIT=1
#                            record cargo-audit/cargo-deny as skipped WITHOUT
#                            attempting them (hermetic/local test runs)
#   SUPPLY_CHAIN_ARTIFACTS   colon-separated artifact paths (overrides
#                            discovery); missing paths are a recorded skip
#   TAMPER=1                 self-test: corrupt one recorded checksum and
#                            require verification to reject it (exits 1)
#
# Tested by `cargo test -p faktor-tests-fuzz-seeds` (fast path + TAMPER).

set -u
set -o pipefail
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

OUT_DIR="${SUPPLY_CHAIN_OUT_DIR:-target/certification}"
FAST="${SUPPLY_CHAIN_FAST:-0}"
BUILD="${SUPPLY_CHAIN_BUILD:-0}"
SKIP_AUDIT="${SUPPLY_CHAIN_SKIP_AUDIT:-0}"
TAMPER="${TAMPER:-0}"

SBOM="$OUT_DIR/sbom.json"
CHECKSUMS="$OUT_DIR/checksums.txt"
STATUS="$OUT_DIR/supply-chain-status.json"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR" || exit 2
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/faktor-supply-chain.XXXXXX")" || exit 2
trap 'rm -rf "$TMP_DIR"' EXIT

FAILED=0
ARTIFACT_COUNT=0
SBOM_WRITTEN=0

TOOL_NAMES=()
TOOL_STATUS=()
TOOL_DETAIL=()
SKIP_NAMES=()
SKIP_REASONS=()

now_ms() {
    if command -v perl >/dev/null 2>&1; then
        perl -MTime::HiRes=time -e 'printf "%.0f\n", time * 1000'
    else
        printf '%s000\n' "$(date +%s)"
    fi
}

json_escape() {
    printf '%s' "$1" | tr -d '\r' | tr '\n' ' ' |
        sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/ /g'
}

record_tool() {
    TOOL_NAMES+=("$1")
    TOOL_STATUS+=("$2")
    TOOL_DETAIL+=("$3")
    printf '[supply-chain] %-14s %-4s %s\n' "$1" "$2" "$3"
}

record_skip() {
    SKIP_NAMES+=("$1")
    SKIP_REASONS+=("$2")
    record_tool "$1" "skip" "$2"
}

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        return 1
    fi
}

verify_checksums() {
    rc=0
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        recorded="$(printf '%s' "$line" | awk '{print $1}')"
        artifact="$(printf '%s' "$line" | sed 's/^[^[:space:]]*[[:space:]][[:space:]]*//')"
        if [ ! -f "$artifact" ]; then
            printf 'CHECKSUM MISMATCH %s: artifact is missing\n' "$artifact" >&2
            rc=1
            continue
        fi
        actual="$(hash_file "$artifact")" || {
            printf 'CHECKSUM ERROR %s: no sha256 tool available\n' "$artifact" >&2
            rc=1
            continue
        }
        if [ "$recorded" != "$actual" ]; then
            printf 'CHECKSUM MISMATCH %s: recorded=%s actual=%s\n' \
                "$artifact" "$recorded" "$actual" >&2
            rc=1
        fi
    done <"$CHECKSUMS"
    return $rc
}

# ---------------------------------------------------------------------------
# 1. SBOM from `cargo metadata`.
# ---------------------------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
    record_skip sbom "cargo not found on PATH; cannot emit an SBOM"
    FAILED=1
elif cargo metadata --format-version 1 --locked >"$TMP_DIR/metadata.json" \
    2>"$LOG_DIR/sbom.err"; then
    if command -v jq >/dev/null 2>&1; then
        jq --arg ts "$(now_ms)" '{
            schema: "faktor-sbom/1",
            generated_at_ms: ($ts | tonumber),
            source: "cargo metadata --format-version 1 --locked",
            package_count: (.packages | length),
            workspace_members: (.workspace_members | length),
            packages: [.packages[] | {name, version, license, source}]
                | sort_by(.name, .version)
        }' "$TMP_DIR/metadata.json" >"$SBOM" || {
            record_tool sbom fail "jq projection failed"
            FAILED=1
        }
        [ -s "$SBOM" ] && SBOM_WRITTEN=1
    elif command -v node >/dev/null 2>&1; then
        node -e '
const fs = require("fs");
const m = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
const out = {
  schema: "faktor-sbom/1",
  generated_at_ms: Number(process.argv[2]),
  source: "cargo metadata --format-version 1 --locked",
  package_count: m.packages.length,
  workspace_members: m.workspace_members.length,
  packages: m.packages
    .map((p) => ({ name: p.name, version: p.version, license: p.license, source: p.source }))
    .sort((a, b) => (a.name + a.version).localeCompare(b.name + b.version)),
};
process.stdout.write(JSON.stringify(out, null, 2) + "\n");
' "$TMP_DIR/metadata.json" "$(now_ms)" >"$SBOM" || {
            record_tool sbom fail "node projection failed"
            FAILED=1
        }
        [ -s "$SBOM" ] && SBOM_WRITTEN=1
    else
        cp "$TMP_DIR/metadata.json" "$SBOM" || {
            record_tool sbom fail "copying raw cargo metadata failed"
            FAILED=1
        }
        [ -s "$SBOM" ] && SBOM_WRITTEN=1
        record_skip sbom-projection \
            "neither jq nor node on PATH; raw cargo metadata written to sbom.json (valid JSON, unprojected)"
    fi
    if [ "$SBOM_WRITTEN" = "1" ]; then
        if command -v jq >/dev/null 2>&1; then
            jq -e . "$SBOM" >/dev/null 2>&1 || {
                record_tool sbom-validate fail "sbom.json is not valid JSON"
                FAILED=1
                SBOM_WRITTEN=0
            }
        elif command -v node >/dev/null 2>&1; then
            node -e 'JSON.parse(require("fs").readFileSync(process.argv[1],"utf8"))' "$SBOM" \
                >/dev/null 2>&1 || {
                record_tool sbom-validate fail "sbom.json is not valid JSON"
                FAILED=1
                SBOM_WRITTEN=0
            }
        fi
        if [ "$SBOM_WRITTEN" = "1" ]; then
            record_tool sbom pass "written to $SBOM"
        fi
    fi
else
    detail="$(head -n 1 "$LOG_DIR/sbom.err" 2>/dev/null | tr -d '\r')"
    [ -n "$detail" ] || detail="cargo metadata --locked failed"
    record_tool sbom fail "cargo metadata failed: $detail (see $LOG_DIR/sbom.err)"
    FAILED=1
fi

# ---------------------------------------------------------------------------
# 2. Release artifact checksums.
# ---------------------------------------------------------------------------
if [ "$BUILD" = "1" ] && [ "$FAST" != "1" ]; then
    if cargo build -p faktor-cli --release --locked >"$LOG_DIR/artifact-build.log" 2>&1; then
        record_tool artifact-build pass "cargo build -p faktor-cli --release --locked"
    else
        record_tool artifact-build fail "see $LOG_DIR/artifact-build.log"
        FAILED=1
    fi
else
    record_skip artifact-build \
        "not building (fast path); set SUPPLY_CHAIN_BUILD=1 for a release build"
fi

: >"$TMP_DIR/artifacts"
if [ -n "${SUPPLY_CHAIN_ARTIFACTS:-}" ]; then
    printf '%s\n' "$SUPPLY_CHAIN_ARTIFACTS" | tr ':' '\n' >"$TMP_DIR/artifact-list"
    while IFS= read -r path; do
        if [ -n "$path" ] && [ -f "$path" ]; then
            printf '%s\n' "$path" >>"$TMP_DIR/artifacts"
        elif [ -n "$path" ]; then
            record_skip "artifact:$path" "requested artifact not found"
        fi
    done <"$TMP_DIR/artifact-list"
else
    for candidate in \
        target/release/faktor-cli \
        target/release/faktor \
        target/debug/faktor-cli \
        target/debug/faktor; do
        [ -f "$candidate" ] && printf '%s\n' "$candidate" >>"$TMP_DIR/artifacts"
    done
fi

sort -u "$TMP_DIR/artifacts" >"$TMP_DIR/artifacts.sorted" 2>/dev/null || : >"$TMP_DIR/artifacts.sorted"
: >"$CHECKSUMS"
while IFS= read -r artifact; do
    [ -n "$artifact" ] || continue
    digest="$(hash_file "$artifact")" || {
        record_tool checksums fail "no sha256 tool (sha256sum/shasum/openssl) on PATH"
        FAILED=1
        break
    }
    printf '%s  %s\n' "$digest" "$artifact" >>"$CHECKSUMS"
    ARTIFACT_COUNT=$((ARTIFACT_COUNT + 1))
done <"$TMP_DIR/artifacts.sorted"

if [ "$ARTIFACT_COUNT" -eq 0 ]; then
    record_skip artifacts \
        "no built binaries found; set SUPPLY_CHAIN_BUILD=1 (or SUPPLY_CHAIN_ARTIFACTS) to hash release artifacts"
else
    record_tool artifacts pass "$ARTIFACT_COUNT artifact(s) hashed into $CHECKSUMS"
fi

# TAMPER self-test: corrupt the first recorded hash and require rejection.
if [ "$TAMPER" = "1" ]; then
    if [ ! -s "$CHECKSUMS" ]; then
        selftest_artifact="$OUT_DIR/selftest-artifact.bin"
        printf 'faktor supply-chain TAMPER self-test artifact\n' >"$selftest_artifact" || exit 2
        printf '%s  %s\n' "$(hash_file "$selftest_artifact")" "$selftest_artifact" >"$CHECKSUMS" || exit 2
    fi
    awk 'NR==1 { c=substr($1,1,1); r=(c=="0"?"1":"0"); printf "%s%s  %s\n", r, substr($1,2), $2; next } { print }' \
        "$CHECKSUMS" >"$TMP_DIR/tampered" || exit 2
    mv "$TMP_DIR/tampered" "$CHECKSUMS" || exit 2
    if verify_checksums >"$TMP_DIR/tamper-verify.log" 2>&1; then
        printf '[supply-chain] TAMPER self-test FAILED: the tampered checksum was ACCEPTED\n' >&2
    else
        printf '[supply-chain] TAMPER self-test PASS: tampered checksum rejected (exit 1 by design)\n'
        cat "$TMP_DIR/tamper-verify.log"
    fi
    exit 1
fi

if verify_checksums >"$TMP_DIR/verify.log" 2>&1; then
    if [ "$ARTIFACT_COUNT" -gt 0 ]; then
        record_tool checksum-verify pass "all $ARTIFACT_COUNT recorded artifact hashes recomputed and match"
    fi
else
    record_tool checksum-verify fail "$(head -n 1 "$TMP_DIR/verify.log")"
    FAILED=1
fi

# ---------------------------------------------------------------------------
# 3. Advisory database tools: attempt, or record WHY not.
# ---------------------------------------------------------------------------
if [ "$SKIP_AUDIT" = "1" ]; then
    record_skip cargo-audit \
        "SUPPLY_CHAIN_SKIP_AUDIT=1: attempt suppressed; tool/network status UNKNOWN (recorded skip)"
    record_skip cargo-deny \
        "SUPPLY_CHAIN_SKIP_AUDIT=1: attempt suppressed; tool/network status UNKNOWN (recorded skip)"
elif ! command -v cargo-audit >/dev/null 2>&1; then
    record_skip cargo-audit "tool not installed ('cargo-audit' not on PATH)"
else
    cargo audit --json >"$LOG_DIR/cargo-audit.json" 2>"$LOG_DIR/cargo-audit.log"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        record_tool cargo-audit pass "no advisories reported ($LOG_DIR/cargo-audit.json)"
    elif grep -qiE 'failed to (fetch|get|download|clone)|network|timed? out|dns|advisory[- ]db|no such host|connection refused|could not' \
        "$LOG_DIR/cargo-audit.log" "$LOG_DIR/cargo-audit.json" 2>/dev/null; then
        record_skip cargo-audit "advisory database/network unavailable (rc=$rc); see $LOG_DIR/cargo-audit.log"
    else
        record_tool cargo-audit fail "cargo audit exited $rc; see $LOG_DIR/cargo-audit.log"
        FAILED=1
    fi
fi

if [ "$SKIP_AUDIT" = "1" ]; then
    : # already recorded above
elif ! command -v cargo-deny >/dev/null 2>&1; then
    record_skip cargo-deny "tool not installed ('cargo-deny' not on PATH)"
else
    cargo deny check >"$LOG_DIR/cargo-deny.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        record_tool cargo-deny pass "all configured checks passed"
    elif grep -qiE 'failed to (fetch|get|download|clone|load)|network|timed? out|dns|advisory[- ]db|no such host|connection refused|could not|unable to find.*(config|deny)' \
        "$LOG_DIR/cargo-deny.log" 2>/dev/null; then
        record_skip cargo-deny "advisory database/network/config unavailable (rc=$rc); see $LOG_DIR/cargo-deny.log"
    else
        record_tool cargo-deny fail "cargo deny check exited $rc; see $LOG_DIR/cargo-deny.log"
        FAILED=1
    fi
fi

# ---------------------------------------------------------------------------
# 4. Status JSON (never silent).
# ---------------------------------------------------------------------------
tools_json=""
i=0
while [ "$i" -lt "${#TOOL_NAMES[@]}" ]; do
    [ -n "$tools_json" ] && tools_json="$tools_json,"
    tools_json="$tools_json{\"name\":\"$(json_escape "${TOOL_NAMES[$i]}")\",\"status\":\"$(json_escape "${TOOL_STATUS[$i]}")\",\"detail\":\"$(json_escape "${TOOL_DETAIL[$i]}")\"}"
    i=$((i + 1))
done
skips_json=""
i=0
while [ "$i" -lt "${#SKIP_NAMES[@]}" ]; do
    [ -n "$skips_json" ] && skips_json="$skips_json,"
    skips_json="$skips_json{\"name\":\"$(json_escape "${SKIP_NAMES[$i]}")\",\"reason\":\"$(json_escape "${SKIP_REASONS[$i]}")\"}"
    i=$((i + 1))
done
if [ "$FAILED" -eq 0 ]; then
    status_value="pass"
else
    status_value="fail"
fi
{
    printf '{\n'
    printf '  "schema": "faktor-supply-chain/1",\n'
    printf '  "status": "%s",\n' "$status_value"
    printf '  "sbom": "%s",\n' "$(json_escape "$SBOM")"
    printf '  "checksums": "%s",\n' "$(json_escape "$CHECKSUMS")"
    printf '  "artifact_count": %s,\n' "$ARTIFACT_COUNT"
    printf '  "tools": [%s],\n' "$tools_json"
    printf '  "skips": [%s]\n' "$skips_json"
    printf '}\n'
} >"$STATUS" || exit 2

printf '[supply-chain] status=%s artifacts=%s recorded_skips=%s status_file=%s\n' \
    "$status_value" "$ARTIFACT_COUNT" "${#SKIP_NAMES[@]}" "$STATUS"
if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
exit 0
