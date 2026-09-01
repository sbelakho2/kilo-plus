#!/usr/bin/env bash
# Run the Kilo+ master integration harness (apps/vscode/harness/client.mjs).
# Builds the CLI when the binary is missing, sets KILO_PLUS_BIN, runs the
# harness, and exits with the harness's exit code.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="$REPO_ROOT/apps/vscode/harness/client.mjs"

BIN="${KILO_PLUS_BIN:-}"
if [[ -z "$BIN" ]]; then
  for candidate in \
    "$REPO_ROOT/target/debug/kilop-cli" \
    "$REPO_ROOT/target/release/kilop-cli"; do
    if [[ -x "$candidate" ]]; then
      BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$BIN" ]]; then
  echo "kilop-cli binary not found; building it..." >&2
  (cd "$REPO_ROOT" && cargo build -p kilop-cli)
  BIN="$REPO_ROOT/target/debug/kilop-cli"
fi

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN is not executable" >&2
  exit 1
fi

export KILO_PLUS_BIN="$BIN"
exec node "$HARNESS"
