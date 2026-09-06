#!/usr/bin/env bash
# Branding scan (normative gate; product name: Faktor).
#
# Exits nonzero when any forbidden legacy wordmark token appears in the
# scanned material, or when package/manifest/CLI metadata still carries the
# legacy names. Tokens (matched literally, case-insensitive):
#   Kilo+ | Kilo Plus | kilo-plus | kilop | kilo server listening
#   | KilopClient | FAKTOR_PLUS
#
# Modes:
#   scripts/branding-scan.sh               source mode (default): scans
#     crates/ tests/ apps/ docs/ scripts/ .github/ plus README.md,
#     Cargo.toml, AGENTS.md at the repo root, plus metadata checks:
#     - package.json "name"/"displayName"/"publisher" fields
#       (any value carrying kilo/faktor-plus is a hit)
#     - Clap command metadata and Cargo.toml package/[[bin]] name fields
#       (source-level `name = "faktor-plus"` / `name = "kilo...` hits;
#       a legacy Clap name or binary name is a hit)
#     - default daemon data-dir constants in crates/cli
#       (`default_value = "...faktor-plus..."` / `"...kilo..."` hits;
#       the data dir itself is `~/.faktor`)
#   scripts/branding-scan.sh --artifacts DIR
#     artifact mode: scans compiled/public assets under DIR (vsix, plugin
#     jars, cargo artifacts, tarballs). Binary payloads are matched
#     byte-level (grep -a), so packaged binaries must carry no token.
#
# Exemption policy — legacy wordmark tokens survive ONLY in frozen
# compatibility material and in tooling that must spell the tokens. Whole
# application/IDE trees (apps/vscode, apps/jetbrains, crates, tests, docs)
# are NEVER exempt: they are scanned like everything else.
#   * paths under compat/ and vendor/ and third-party/  — frozen
#     compatibility fixtures and upstream sources (e.g.
#     compat/kilo-v756, vendor/upstream-kilo); entries that do not exist
#     are tolerated
#   * crates/protocol/src/v756/  — frozen v7.5.6 wire mirror of the
#     compat/kilo-v756 fixtures; the retained legacy handshake prefix
#     lives here so the daemon can reject the old handshake loudly
#   * crates/server/src/api.rs    — the frozen v756 auth/legacy-handshake
#     tests assert the legacy forms (which the server still must not emit)
#   * scripts/check-docs-sync.sh  — the docs-drift guard must spell the
#     forbidden identifiers to scan docs/architecture.md for them (same
#     self-reference as this script)
#   * this script itself (it must spell the tokens to scan for)
# Build outputs (node_modules/, target/, .git/, build/, .gradle/, tsc
# out/) are never scanned in source mode; compiled artifacts belong to
# --artifacts mode.
#
# Note: the GitHub repository name/description are EXTERNAL metadata and
# cannot be renamed from inside this repository; in-repo package/manifest
# metadata is authoritative and is what this scan enforces.
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

# Path fragments that mark an exempt path. Matching is case-insensitive on
# the path relative to the scan root. Precise exemptions only: compat/
# vendor/ third-party/ trees, the two frozen legacy mirrors (v7.5.6 wire
# mirror in the protocol crate, frozen v756 tests in server/src/api.rs)
# and self-referential tooling. No whole application/IDE trees.
ALLOWLIST_FRAGMENTS=(
  '/compat/'
  '/vendor/'
  '/third-party/'
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
  '/build/'
  '/.gradle/'
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

# Wordmark token scan (both modes).
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

# Metadata scan (source mode only): manifest/package fields, Clap command
# names, Cargo.toml [[bin]] name fields, and default data-dir constants in
# crates/cli. The frozen wire mirrors keep legacy FORMS only (never
# package/manifest names), so this pass has no exemptions of its own beyond
# is_skipped.
meta_hits=0
if [ "$MODE" = source ]; then
  while IFS= read -r file; do
    [ -n "$file" ] || continue
    if is_skipped "$file"; then
      continue
    fi
    case "$file" in
      *.rs | *.toml | *.gradle.kts | *.kt)
        # Clap #[command(name = ...)] / Cargo [package]/[[bin]] name fields
        # / Gradle project names carrying the legacy identifiers.
        if grep -a -n -H -i -E 'name[[:space:]]*=[[:space:]]*"(faktor-plus|kilo)' \
            -- "$file" 2>/dev/null; then
          meta_hits=$((meta_hits + 1))
        fi
        ;;
    esac
    case "$file" in
      */package.json)
        # VS Code (and any other) extension manifest: name/displayName/
        # publisher must carry no kilo/faktor-plus form.
        if grep -a -n -H -i -E \
            '"(name|displayName|publisher)"[[:space:]]*:[[:space:]]*"[^"]*(kilo|faktor-plus)' \
            -- "$file" 2>/dev/null; then
          meta_hits=$((meta_hits + 1))
        fi
        ;;
    esac
    case "$file" in
      */crates/cli/*.rs)
        # Default daemon data-dir constants in cli code must point at the
        # Faktor data dir (~/.faktor), never a legacy dir name.
        if grep -a -n -H -i -E \
            'default_value[[:space:]]*=[[:space:]]*"[^"]*(faktor-plus|kilo)' \
            -- "$file" 2>/dev/null; then
          meta_hits=$((meta_hits + 1))
        fi
        ;;
    esac
  done < <(collect_files)
fi

if [ "$hits" -gt 0 ]; then
  echo "branding scan: $hits file(s) contain legacy wordmark tokens outside the exemption set" >&2
  exit 1
fi
if [ "$meta_hits" -gt 0 ]; then
  echo "branding scan: $meta_hits file(s) carry legacy package/CLI/data-dir metadata" >&2
  exit 1
fi

echo "branding scan: clean ($MODE mode; no legacy wordmark tokens or metadata outside the exemption set)"
if [ "$MODE" = source ]; then
  echo "note: the GitHub repository name/description are external metadata and cannot be changed"
  echo "      from this repository; in-repo package/manifest metadata is authoritative here."
fi
exit 0
