#!/usr/bin/env bash
# Branding scan (normative gate, see docs/specs/branding.md).
#
# Exits nonzero when any forbidden legacy wordmark token appears in the
# scanned material. Tokens (matched literally, case-insensitive):
#   Kilo+ | Kilo Plus | kilo-plus | kilop | kilo server listening
#   | FAKTOR_PLUS | KilopClient
#
# Modes:
#   scripts/branding-scan.sh               source mode (default): scans
#     crates/ tests/ apps/ docs/ scripts/ .github/ plus README.md,
#     Cargo.toml, AGENTS.md at the repo root.
#   scripts/branding-scan.sh --artifacts DIR
#     artifact mode: scans compiled/public assets under DIR (vsix, plugin
#     jars, cargo artifacts, tarballs). Binary payloads are matched
#     byte-level (grep -a), so packaged binaries must carry no token.
#
# Allowlist (by path, recursive; entries that do not exist are tolerated):
#   * paths under compat/ and vendor/  — frozen compatibility fixtures and
#     upstream sources (e.g. compat/upstream-kilo-v756, vendor/upstream-kilo)
#   * apps/jetbrains/                  — frozen JetBrains 7.1.2 legacy IDE
#     shell; retains old forms by design (docs/specs/branding.md)
#   * crates/protocol/src/v756/        — frozen v7.5.6 wire mirror of the
#     compat/kilo-v756 fixtures; the retained legacy handshake prefix lives
#     here so the daemon can reject the old handshake loudly
#   * crates/server/src/api.rs         — the frozen v756 auth/legacy-handshake
#     tests assert the legacy forms (which the server still must not emit)
#   * scripts/check-docs-sync.sh       — the docs-drift guard must spell the
#     forbidden identifiers to scan docs/architecture.md for them (same
#     self-reference as this script)
#   * this script itself (it must spell the tokens to scan for)
# Build outputs (node_modules/, target/, .git/, tsc out/) are never scanned
# in source mode; compiled artifacts belong to --artifacts mode.
#
# No external dependencies beyond find/grep. Run from anywhere; the repo
# root is derived from the script location.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

TOKENS=(
  'Kilo+'
  'Kilo Plus'
  'kilo-plus'
  'kilop'
  'kilo server listening'
  'FAKTOR_PLUS'
  'KilopClient'
)

# Path fragments that mark an allowlisted path. Matching is case-insensitive
# on the path relative to the scan root.
ALLOWLIST_FRAGMENTS=(
  '/compat/'
  '/vendor/'
  '/apps/jetbrains/'
  '/crates/protocol/src/v756/'
  '/crates/server/src/api.rs'
  '/scripts/check-docs-sync.sh'
)

# Fragments for paths that are never scanned in source mode.
SKIP_FRAGMENTS=(
  '/node_modules/'
  '/target/'
  '/.git/'
  '/out/'
  "/scripts/branding-scan.sh"
)

usage() {
  echo "usage: $0 [--artifacts DIR]" >&2
  exit 2
}

MODE=source
ARTIFACT_DIR=""
if [ "${1:-}" = "--artifacts" ]; then
  MODE=artifacts
  ARTIFACT_DIR="${2:-}"
  if [ -z "$ARTIFACT_DIR" ]; then
    usage
  fi
  if [ ! -d "$ARTIFACT_DIR" ]; then
    echo "error: --artifacts directory does not exist: $ARTIFACT_DIR" >&2
    exit 2
  fi
elif [ "$#" -gt 0 ]; then
  usage
fi

is_skipped() {
  local path="$1"
  local frag
  for frag in "${SKIP_FRAGMENTS[@]}" "${ALLOWLIST_FRAGMENTS[@]}"; do
    if printf '%s' "$path" | grep -qi -F "$frag"; then
      return 0
    fi
  done
  return 1
}

collect_files() {
  if [ "$MODE" = artifacts ]; then
    find "$ARTIFACT_DIR" -type f 2>/dev/null
  else
    local d
    for d in crates tests apps docs scripts .github; do
      if [ -d "$ROOT/$d" ]; then
        find "$ROOT/$d" -type f 2>/dev/null
      fi
    done
    local f
    for f in README.md Cargo.toml AGENTS.md; do
      if [ -f "$ROOT/$f" ]; then
        printf '%s\n' "$ROOT/$f"
      fi
    done
  fi
}

hits=0
while IFS= read -r file; do
  [ -n "$file" ] || continue
  if is_skipped "$file"; then
    continue
  fi
  if grep -a -n -H -i -F -e 'Kilo+' -e 'Kilo Plus' -e 'kilo-plus' -e 'kilop' \
      -e 'kilo server listening' -e 'FAKTOR_PLUS' -e 'KilopClient' -- "$file" 2>/dev/null; then
    hits=$((hits + 1))
  fi
done < <(collect_files)

if [ "$hits" -gt 0 ]; then
  echo "branding scan: $hits file(s) contain legacy wordmark tokens outside the allowlist" >&2
  exit 1
fi

echo "branding scan: clean ($MODE mode, no legacy wordmark tokens outside the allowlist)"
exit 0
