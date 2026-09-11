#!/usr/bin/env bash
# Cross-target compile certification (Windows MSVC compile-only lane).
#
# `cargo check --target x86_64-pc-windows-msvc` for every crate that can be
# compiled WITHOUT a C cross-toolchain on this host. Nothing is linked and no
# Windows SDK is needed for `check`; only build scripts that compile C/asm
# for the target are a hard limit here. Those crates are recorded as SKIP
# with the exact tool error — never silently dropped, never a fake pass —
# while a genuine Rust compile error is a FAILURE and exits non-zero.
#
# Per-crate statuses are emitted to target/certification/cross-target.json:
#   {"crate": ..., "status": "pass|skip|fail", "detail": ...}
# Exit 0 when there is no real compile error (recorded skips allowed),
# exit 1 when a crate fails for a reason other than the missing C
# cross-toolchain / missing target.
#
# blake3 workaround (verified by reading the vendored build script):
# blake3 1.8.7 build.rs `is_pure()` returns true when the environment
# variable CARGO_FEATURE_PURE is defined, in which case it selects the
# Rust-intrinsics path and compiles NO C/asm. Without it, the msvc target
# fails with `error occurred in cc-rs: failed to find tool "ml64.exe"` on
# hosts with no MASM. The variable only affects blake3's build script; other
# native deps (zstd-sys, libsqlite3-sys) still require a C cross-toolchain
# and are recorded as skips with their exact error.
#
# Env:
#   CROSS_TARGET          target triple (default x86_64-pc-windows-msvc)
#   CERTIFY_OUT_DIR       report dir (default target/certification)
#   CROSS_TARGET_SELFTEST=classify  prove the status classifier on synthetic
#                         logs (real Rust error => fail, ml64/missing header
#                         => skip) without invoking cargo.
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

TARGET="${CROSS_TARGET:-x86_64-pc-windows-msvc}"
OUT_DIR="${CERTIFY_OUT_DIR:-target/certification}"
LOG_DIR="${CROSS_TARGET_LOG_DIR:-$OUT_DIR/logs}"
REPORT="${CROSS_TARGET_REPORT:-$OUT_DIR/cross-target.json}"

# crate package : source directory (a missing directory is recorded skipped).
CANDIDATES=(
    "faktor-core:core"
    "faktor-fs:fs"
    "faktor-pty:pty"
    "faktor-winjob:winjob"
    "faktor-terminal:terminal"
    "faktor-evidence:evidence"
    "faktor-context:context"
    "faktor-semantic:semantic"
    "faktor-verify:verify"
)

# A missing C cross-toolchain or target, not a Rust compile error.
TOOLCHAIN_RE='failed to find tool|error occurred in cc-rs|fatal error:.*file not found|ml64|link\.exe.*not found|lld-link.*not found'
TARGET_RE="can't find crate for .std.|may not be installed|Error loading target specification|could not find.*(std|core) for"

# classifies a captured cargo log: pass | skip | fail.
classify() {
    local log="$1" rc="$2"
    if [ "$rc" -eq 0 ]; then
        printf 'pass'
    elif grep -aE "$TOOLCHAIN_RE|$TARGET_RE" "$log" >/dev/null 2>&1; then
        printf 'skip'
    else
        printf 'fail'
    fi
}

skip_detail() {
    local log="$1" line
    line="$(grep -a -m1 -E 'failed to find tool' "$log" 2>/dev/null)"
    [ -n "$line" ] || line="$(grep -a -m1 -E "fatal error:.*file not found" "$log" 2>/dev/null)"
    [ -n "$line" ] || line="$(grep -a -m1 -E 'error occurred in cc-rs' "$log" 2>/dev/null)"
    [ -n "$line" ] || line="$(grep -a -m1 -E "$TARGET_RE" "$log" 2>/dev/null)"
    [ -n "$line" ] || line="$(grep -a -m1 -E "$TOOLCHAIN_RE" "$log" 2>/dev/null)"
    [ -n "$line" ] || line="requires a C cross-toolchain for $TARGET"
    printf '%s' "$line" | sed -e 's/^cargo:warning=//' -e 's/^warning: [^ ]*: //' | cut -c1-240
}

first_error_line() {
    local log="$1" line
    line="$(grep -a -m1 -E '(^error(\[E[0-9]+\])?:|^error:|panicked at|FAILED)' "$log" 2>/dev/null | head -n1)"
    [ -n "$line" ] || line="$(grep -a -m1 -iE 'error|fail' "$log" 2>/dev/null | head -n1)"
    [ -n "$line" ] || line="$(grep -a -m1 -v '^[[:space:]]*$' "$log" 2>/dev/null | head -n1)"
    printf '%s' "$line" | cut -c1-240
}

json_escape() {
    printf '%s' "$1" | tr -d '\r' | tr '\n' ' ' |
        sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/ /g'
}

# ---------------------------------------------------------------------------
# Classifier selftest (no cargo): proves real errors fail and toolchain
# limits skip.
# ---------------------------------------------------------------------------
if [ "${CROSS_TARGET_SELFTEST:-}" = "classify" ]; then
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/kp-cross-selftest.XXXXXX")" || exit 2
    failures=0
    expect_status() {
        local want="$1" rc="$2" body="$3" got
        printf '%s\n' "$body" >"$tmp/log"
        got="$(classify "$tmp/log" "$rc")"
        if [ "$got" != "$want" ]; then
            printf 'cross-target selftest: %s => %s (expected %s)\n' "$body" "$got" "$want" >&2
            failures=$((failures + 1))
        fi
    }
    expect_status pass 0 'Finished `dev` profile'
    expect_status fail 1 'error[E0308]: mismatched types
error: could not compile `faktor-x` (lib) due to 1 previous error'
    expect_status skip 1 'error occurred in cc-rs: failed to find tool "ml64.exe": No such file or directory'
    expect_status skip 1 "fatal error: 'string.h' file not found"
    expect_status skip 1 "error[E0463]: can't find crate for \`std\` target may not be installed"
    rm -rf "$tmp"
    if [ "$failures" -eq 0 ]; then
        printf 'cross-target selftest: PASS (pass/skip/fail classification)\n'
        exit 0
    fi
    printf 'cross-target selftest: FAIL (%s assertion(s))\n' "$failures" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$OUT_DIR" || exit 2

HAVE_CARGO=1
command -v cargo >/dev/null 2>&1 || HAVE_CARGO=0
TARGET_INSTALLED=0
if [ "$HAVE_CARGO" -eq 1 ] && command -v rustup >/dev/null 2>&1; then
    if rustup target list --installed 2>/dev/null | grep -qx "$TARGET"; then
        TARGET_INSTALLED=1
    fi
fi

CRATE_NAME=()
CRATE_STATUS=()
CRATE_DETAIL=()
FAILED=0
PASSES=0
SKIPS=0

printf 'cross-target check: target=%s host=%s\n' \
    "$TARGET" "$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || printf unknown)"

for entry in "${CANDIDATES[@]}"; do
    name="${entry%%:*}"
    dir="${entry##*:}"
    if [ ! -d "crates/$dir" ]; then
        status="skip"
        detail="crate directory crates/$dir absent from this workspace"
    elif [ "$HAVE_CARGO" -eq 0 ]; then
        status="skip"
        detail="cargo is unavailable on this host"
    elif [ "$TARGET_INSTALLED" -eq 0 ] && command -v rustup >/dev/null 2>&1; then
        status="skip"
        detail="target $TARGET not installed (rustup target add $TARGET)"
    else
        log="$LOG_DIR/cross-target-$name.log"
        CARGO_FEATURE_PURE=1 cargo check --target "$TARGET" -p "$name" >"$log" 2>&1
        rc=$?
        status="$(classify "$log" "$rc")"
        if [ "$status" = "pass" ]; then
            detail="ok"
        elif [ "$status" = "skip" ]; then
            detail="needs C cross-toolchain: $(skip_detail "$log")"
        else
            detail="$(first_error_line "$log")"
        fi
    fi
    CRATE_NAME+=("$name")
    CRATE_STATUS+=("$status")
    CRATE_DETAIL+=("$detail")
    case "$status" in
        pass) PASSES=$((PASSES + 1)) ;;
        skip) SKIPS=$((SKIPS + 1)) ;;
        fail) FAILED=1 ;;
    esac
    printf '  [%s] %-18s %s\n' "$status" "$name" "$detail"
done

if [ "$FAILED" -eq 0 ]; then
    OVERALL="pass"
else
    OVERALL="fail"
fi

mkdir -p "$OUT_DIR"
{
    printf '{\n'
    printf '  "schema": "faktor-cross-target-report/v1",\n'
    printf '  "target": "%s",\n' "$(json_escape "$TARGET")"
    printf '  "host": "%s",\n' "$(json_escape "$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || printf unknown)")"
    printf '  "workaround": "CARGO_FEATURE_PURE=1 (blake3 1.8.7 build.rs is_pure() skips C/asm; verified empirically: without it crates/fs fails with ml64.exe, with it passes)",\n'
    printf '  "status": "%s",\n' "$OVERALL"
    printf '  "crates": ['
    n=${#CRATE_NAME[@]}
    for ((k = 0; k < n; k++)); do
        if [ "$k" -gt 0 ]; then
            printf ','
        fi
        printf '\n    {"crate": "%s", "status": "%s", "detail": "%s"}' \
            "$(json_escape "${CRATE_NAME[$k]}")" \
            "${CRATE_STATUS[$k]}" \
            "$(json_escape "${CRATE_DETAIL[$k]}")"
    done
    if [ "$n" -gt 0 ]; then
        printf '\n  '
    fi
    printf ']\n}\n'
} >"$REPORT.tmp" && mv "$REPORT.tmp" "$REPORT"

printf 'cross-target: %s (%s pass, %s recorded skip(s), %s real error(s))\n' \
    "$(printf '%s' "$OVERALL" | tr '[:lower:]' '[:upper:]')" "$PASSES" "$SKIPS" "$((n - PASSES - SKIPS))"
printf 'report: %s\n' "$REPORT"

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
exit 0
