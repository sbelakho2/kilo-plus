#!/usr/bin/env bash
# CI guard for the vendored-webview visual gate (job: vscode-visual-with-skip).
#
# Contract:
#   1. always run the offline self-test of the visual gate itself
#      (tampered/missing/extra dist assets must fail the self-test);
#   2. always run the structural + manifest gate over ui/kilo-v756-webview;
#   3. try to install playwright + chromium when no usable headless browser
#      is present, then run the full gate so the render fingerprint is
#      actually exercised;
#   4. when the browser cannot be installed, record an explicit SKIP with
#      `--no-headless` (writes dist/visual-report.json) — a skip is never a
#      silent pass, and a real mismatch still exits non-zero.
#
# Env:
#   VISUAL_PLAYWRIGHT_DIR  install location (default target/visual-playwright)
#   FAKTOR_PLAYWRIGHT      honoured by scripts/webview-visual-check.mjs
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

echo "== visual gate self-test (offline) =="
node scripts/webview-visual-check.mjs --self-test || exit $?

PW_DIR="${VISUAL_PLAYWRIGHT_DIR:-$ROOT/target/visual-playwright}"
mkdir -p "$PW_DIR"

browser_ready() {
  node -e '
    const spec = process.env.FAKTOR_PLAYWRIGHT || "playwright";
    import(spec).then((m) => {
      const mod = m.chromium ?? m.default?.chromium;
      if (typeof mod?.executablePath !== "function") process.exit(1);
      const fs = require("node:fs");
      const exe = mod.executablePath();
      process.exit(exe && fs.existsSync(exe) ? 0 : 1);
    }).catch(() => process.exit(1));
  ' >/dev/null 2>&1
}

if browser_ready; then
  echo "== headless browser present; running the full visual gate =="
  node scripts/webview-visual-check.mjs
  exit $?
fi

echo "== no usable playwright chromium; attempting install into $PW_DIR =="
if npm install --prefix "$PW_DIR" --no-save playwright >"$PW_DIR/install.log" 2>&1 &&
  "$PW_DIR/node_modules/.bin/playwright" install chromium >>"$PW_DIR/install.log" 2>&1; then
  export FAKTOR_PLAYWRIGHT="$PW_DIR/node_modules/playwright"
  echo "== chromium installed; running the full visual gate =="
  node scripts/webview-visual-check.mjs
  exit $?
fi

tail -n 40 "$PW_DIR/install.log" 2>/dev/null || true
echo "== playwright/chromium unavailable; recording an explicit skip =="
node scripts/webview-visual-check.mjs --no-headless
exit $?
