#!/usr/bin/env bash
# Vendor the frozen upstream Kilo v7.5.6 UI paths into ui/, then verify hashes.
#
#   scripts/vendor-upstream.sh          fetch pinned commit, re-copy, re-hash, verify
#   scripts/vendor-upstream.sh --check  verify the existing vendored tree only (offline)
#
# Fetch protocol (exactly what this script does, and what ui/upstream.json pins):
#   git clone --filter=blob:none --no-checkout --depth 1 --branch v7.5.6 \
#       https://github.com/Kilo-Org/kilocode <tmp>/repo
#   git -C <tmp>/repo sparse-checkout set \
#       packages/kilo-vscode/webview-ui packages/kilo-ui
#   git -C <tmp>/repo checkout fa02955bfa17b60e57e0d7406d200a73337472ee
#   rsync -a --delete <tmp>/repo/packages/kilo-vscode/webview-ui/ ui/kilo-v756-webview/
#   rsync -a --delete <tmp>/repo/packages/kilo-ui/ ui/kilo-ui/
#   node scripts/verify-upstream.mjs --write && node scripts/verify-upstream.mjs --verify
#
# Offline behavior: when the fetch fails (no network/DNS/registry), this script
# exits nonzero and prints the protocol above. It never fabricates vendored
# content; a blocked run leaves ui/ untouched.

set -euo pipefail

REPO="https://github.com/Kilo-Org/kilocode"
TAG="v7.5.6"
COMMIT="fa02955bfa17b60e57e0d7406d200a73337472ee"
UPSTREAM_WEBVIEW="packages/kilo-vscode/webview-ui"
UPSTREAM_KILO_UI="packages/kilo-ui"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

print_protocol() {
  cat <<'EOF'
fetch protocol:
  git clone --filter=blob:none --no-checkout --depth 1 --branch v7.5.6 \
      https://github.com/Kilo-Org/kilocode <tmp>/repo
  git -C <tmp>/repo sparse-checkout set packages/kilo-vscode/webview-ui packages/kilo-ui
  git -C <tmp>/repo checkout fa02955bfa17b60e57e0d7406d200a73337472ee
  rsync -a --delete <tmp>/repo/packages/kilo-vscode/webview-ui/ ui/kilo-v756-webview/
  rsync -a --delete <tmp>/repo/packages/kilo-ui/ ui/kilo-ui/
  node scripts/verify-upstream.mjs --write
  node scripts/verify-upstream.mjs --verify
EOF
}

case "${1:-}" in
  --check)
    node scripts/verify-upstream.mjs --verify
    exit $?
    ;;
  --help|-h)
    sed -n '2,23p' "$0"
    exit 0
    ;;
  "") ;;
  *)
    echo "unknown argument: $1" >&2
    exit 2
    ;;
esac

if ! command -v git >/dev/null 2>&1; then
  echo "BLOCKED: git not found on PATH" >&2
  print_protocol
  exit 2
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/faktor-vendor-upstream.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

echo "== fetching $REPO $TAG ($COMMIT)"
if ! git clone --filter=blob:none --no-checkout --depth 1 --branch "$TAG" "$REPO" "$TMP/repo"; then
  echo "BLOCKED: network fetch failed; no vendored content was written or modified" >&2
  print_protocol
  exit 2
fi

git -C "$TMP/repo" sparse-checkout set "$UPSTREAM_WEBVIEW" "$UPSTREAM_KILO_UI"
if ! git -C "$TMP/repo" checkout "$COMMIT"; then
  echo "BLOCKED: checkout of pinned commit failed; no vendored content was written" >&2
  exit 2
fi

ACTUAL="$(git -C "$TMP/repo" rev-parse HEAD)"
if [ "$ACTUAL" != "$COMMIT" ]; then
  echo "REFUSING: fetched HEAD $ACTUAL != pinned $COMMIT" >&2
  exit 2
fi

echo "== vendoring verbatim (no edits)"
mkdir -p ui/kilo-v756-webview ui/kilo-ui ui/LICENSES
rsync -a --delete "$TMP/repo/$UPSTREAM_WEBVIEW/" ui/kilo-v756-webview/
rsync -a --delete "$TMP/repo/$UPSTREAM_KILO_UI/" ui/kilo-ui/
cp "$TMP/repo/LICENSE" ui/LICENSES/kilocode-LICENSE.txt
cp "$TMP/repo/packages/kilo-vscode/LICENSE" ui/LICENSES/kilo-vscode-LICENSE.txt
# The third-party notice directory is outside the sparse checkout; read the
# exact pinned blobs instead of widening the fetch.
rm -rf ui/LICENSES/kilo-vscode-THIRD_PARTY_LICENSES
while IFS= read -r file; do
  rel="${file#"packages/kilo-vscode/THIRD_PARTY_LICENSES/"}"
  mkdir -p "ui/LICENSES/kilo-vscode-THIRD_PARTY_LICENSES/$(dirname "$rel")"
  git -C "$TMP/repo" show "$COMMIT:$file" > "ui/LICENSES/kilo-vscode-THIRD_PARTY_LICENSES/$rel"
done < <(git -C "$TMP/repo" ls-tree -r --name-only "$COMMIT" packages/kilo-vscode/THIRD_PARTY_LICENSES)

echo "== hashing"
node scripts/verify-upstream.mjs --write
echo "== verifying"
node scripts/verify-upstream.mjs --verify
echo "VENDOR OK: $COMMIT"
