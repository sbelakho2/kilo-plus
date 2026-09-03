#!/usr/bin/env bash
# Run the Faktor master integration harness (apps/vscode/harness/client.mjs).
# Builds the CLI when the binary is missing, sets FAKTOR_PLUS_BIN, runs the
# harness, and exits with the harness's exit code.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="$REPO_ROOT/apps/vscode/harness/client.mjs"

BIN="${FAKTOR_PLUS_BIN:-}"
if [[ -z "$BIN" ]]; then
  for candidate in \
    "$REPO_ROOT/target/debug/faktor-cli" \
    "$REPO_ROOT/target/release/faktor-cli"; do
    if [[ -x "$candidate" ]]; then
      BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$BIN" ]]; then
  echo "faktor-cli binary not found; building it..." >&2
  (cd "$REPO_ROOT" && cargo build -p faktor-cli)
  BIN="$REPO_ROOT/target/debug/faktor-cli"
fi

if [[ ! -x "$BIN" ]]; then
  echo "error: $BIN is not executable" >&2
  exit 1
fi

export FAKTOR_PLUS_BIN="$BIN"
exec node "$HARNESS"
