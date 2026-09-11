#!/usr/bin/env node
// Machine-readable capability manifest (certification truth audit).
//
// Every status is DERIVED from repository files and scripts — never from
// prose. The generator probes the pinned vendored UI tree, the extension
// sources, the JetBrains backend/frontend sources, the frozen compat
// fixtures and the ACP subset, then writes:
//
//   target/certification/capabilities.json
//
// and verifies that the capability table in docs/certification.md carries
// the exact same status for every surface (`--check`, the drift test). A
// mismatch — a stale prose label, a missing row, or a status that no longer
// matches the tree — exits non-zero.
//
// Usage:
//   node scripts/capabilities-manifest.mjs                 generate + drift check
//   node scripts/capabilities-manifest.mjs --generate-only  skip the docs check
//
// Env:
//   CAPABILITIES_OUT_DIR  output directory (default target/certification)

import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = process.env.CAPABILITIES_OUT_DIR || 'target/certification';
const MANIFEST_PATH = resolve(ROOT, OUT_DIR, 'capabilities.json');
const DOC_PATH = resolve(ROOT, 'docs/certification.md');
const GENERATE_ONLY = process.argv.includes('--generate-only');

const file = (rel) => existsSync(resolve(ROOT, rel));
const dir = (rel) => existsSync(resolve(ROOT, rel)) && !file(rel);

const VENDORED_WEBVIEW_ROOT = 'ui/kilo-v756-webview';
const JETBRAINS_712_ROOTS = ['ui/kilo-jetbrains-712', 'compat/jetbrains-712'];

// ------------------------------------------------------------- derivations

function vscodeWebviewStatus() {
  if (!file('apps/vscode/src/webview.ts') || !file('apps/vscode/src/kilo-bridge.ts')) {
    return 'ABSENT';
  }
  if (
    file(`${VENDORED_WEBVIEW_ROOT}/dist/webview.js`) &&
    file(`${VENDORED_WEBVIEW_ROOT}/dist/webview.css`)
  ) {
    // The pinned v7.5.6 webview bundle is vendored AND built; the visual
    // baseline proves the gate ran on this tree.
    return file(`${VENDORED_WEBVIEW_ROOT}/dist/visual-baseline.json`)
      ? 'IMPLEMENTED'
      : 'PARTIAL';
  }
  if (dir(VENDORED_WEBVIEW_ROOT)) {
    return 'PARTIAL';
  }
  // Built-in fallback shell only: the upstream assets are absent.
  return 'BLOCKED_EXTERNAL';
}

function jetbrains712Vendored() {
  return JETBRAINS_712_ROOTS.some((root) => dir(root));
}

const SURFACES = {
  vscode_native_client: () => ({
    status:
      file('apps/vscode/src/nativeClient.ts') && file('apps/vscode/scripts/selftest.mjs')
        ? 'IMPLEMENTED'
        : 'ABSENT',
    evidence: ['apps/vscode/src/nativeClient.ts', 'apps/vscode/scripts/selftest.mjs'],
  }),
  vscode_webview: () => ({
    status: vscodeWebviewStatus(),
    evidence: [
      'apps/vscode/src/webview.ts',
      'apps/vscode/src/kilo-bridge.ts',
      `${VENDORED_WEBVIEW_ROOT}/dist/webview.js`,
      `${VENDORED_WEBVIEW_ROOT}/dist/visual-baseline.json`,
      'ui/upstream.json',
    ],
  }),
  jetbrains_native_bridge: () => ({
    status:
      file('apps/jetbrains/backend/src/main/kotlin/dev/faktor/backend/NativeClient.kt') &&
      file('apps/jetbrains/backend/src/main/kotlin/dev/faktor/backend/NativeEventStream.kt') &&
      file('apps/jetbrains/compile-and-smoke.sh')
        ? 'IMPLEMENTED'
        : 'ABSENT',
    evidence: [
      'apps/jetbrains/backend/src/main/kotlin/dev/faktor/backend/NativeClient.kt',
      'apps/jetbrains/backend/src/main/kotlin/dev/faktor/backend/NativeEventStream.kt',
      'apps/jetbrains/compile-and-smoke.sh',
    ],
  }),
  jetbrains_frontend: () => ({
    status:
      file('apps/jetbrains/frontend/src/main/kotlin/dev/faktor/frontend/FaktorChatPanel.kt') &&
      file('apps/jetbrains/frontend/src/main/resources/META-INF/plugin.xml') &&
      file('apps/jetbrains/frontend/build.gradle.kts')
        ? 'PARTIAL'
        : 'ABSENT',
    evidence: [
      'apps/jetbrains/frontend/src/main/kotlin/dev/faktor/frontend/FaktorChatPanel.kt',
      'apps/jetbrains/frontend/src/main/resources/META-INF/plugin.xml',
      'apps/jetbrains/frontend/build.gradle.kts',
    ],
  }),
  compat_v756: () => ({
    status:
      file('compat/kilo-v756/startup_line.json') &&
      file('compat/kilo-v756/sse_frames.json') &&
      file('tests/compat/Cargo.toml')
        ? 'IMPLEMENTED'
        : 'ABSENT',
    evidence: [
      'compat/kilo-v756',
      'compat/kilo-v756/startup_line.json',
      'compat/kilo-v756/sse_frames.json',
      'tests/compat',
    ],
  }),
  ui_parity: () => {
    const vscode = vscodeWebviewStatus();
    const jetbrains712 = jetbrains712Vendored();
    let status = 'BLOCKED_EXTERNAL';
    if (vscode === 'IMPLEMENTED' && jetbrains712) {
      status = 'IMPLEMENTED';
    } else if (vscode !== 'BLOCKED_EXTERNAL' || jetbrains712) {
      status = 'PARTIAL';
    }
    return {
      status,
      evidence: [
        `${VENDORED_WEBVIEW_ROOT}/dist/visual-baseline.json`,
        'ui/upstream.json',
        ...JETBRAINS_712_ROOTS,
      ],
    };
  },
  acp_subset: () => ({
    status:
      file('crates/acp/src/lib.rs') &&
      file('crates/acp/src/protocol.rs') &&
      file('crates/acp/tests/interop.rs') &&
      file('tests/acp-official/Cargo.toml')
        ? 'IMPLEMENTED'
        : 'ABSENT',
    evidence: [
      'crates/acp/src/lib.rs',
      'crates/acp/src/protocol.rs',
      'crates/acp/tests/interop.rs',
      'tests/acp-official',
    ],
  }),
};

function buildManifest() {
  let commit = 'unknown';
  try {
    commit = execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: ROOT,
      encoding: 'utf8',
    }).trim();
  } catch {
    // A tree without git metadata still yields a usable (if unbound) manifest.
  }
  const surfaces = {};
  for (const [key, derive] of Object.entries(SURFACES)) {
    const { status, evidence } = derive();
    surfaces[key] = {
      status,
      evidence: evidence.filter((rel) => file(rel) || dir(rel)),
    };
  }
  return {
    schema: 'faktor-capabilities-manifest/v1',
    commit,
    generated_from: 'repository files/scripts (scripts/capabilities-manifest.mjs)',
    surfaces,
  };
}

// ---------------------------------------------------------- docs drift check

// The doc's capability table rows carry a backticked key and an uppercase
// status: `| \`vscode_webview\` | IMPLEMENTED | ... |`.
const DOC_ROW = /^\|\s*`([a-z0-9_]+)`\s*\|\s*([A-Z_]+)\s*\|/;

function parseDocTable() {
  if (!existsSync(DOC_PATH)) {
    throw new Error(`${DOC_PATH} does not exist`);
  }
  const rows = new Map();
  for (const line of readFileSync(DOC_PATH, 'utf8').split('\n')) {
    const match = DOC_ROW.exec(line);
    if (!match) {
      continue;
    }
    const [, key, status] = match;
    if (rows.has(key)) {
      throw new Error(`docs/certification.md lists capability '${key}' twice`);
    }
    rows.set(key, status);
  }
  return rows;
}

function checkDocs(manifest) {
  const rows = parseDocTable();
  const errors = [];
  for (const [key, surface] of Object.entries(manifest.surfaces)) {
    if (!rows.has(key)) {
      errors.push(`docs/certification.md is missing a capability row for '${key}'`);
    } else if (rows.get(key) !== surface.status) {
      errors.push(
        `docs/certification.md says ${key}=${rows.get(key)} but the tree derives ${surface.status}`,
      );
    }
  }
  for (const key of rows.keys()) {
    if (!(key in manifest.surfaces)) {
      errors.push(`docs/certification.md lists unknown capability '${key}'`);
    }
  }
  if (errors.length > 0) {
    for (const error of errors) {
      console.error(`capabilities drift: ${error}`);
    }
    console.error(
      'fix docs/certification.md (or the derivation) so the table matches target/certification/capabilities.json',
    );
    process.exit(1);
  }
}

// --------------------------------------------------------------------- main

const manifest = buildManifest();
mkdirSync(resolve(ROOT, OUT_DIR), { recursive: true });
writeFileSync(MANIFEST_PATH, `${JSON.stringify(manifest, null, 2)}\n`);

if (!GENERATE_ONLY) {
  checkDocs(manifest);
}

const summary = Object.entries(manifest.surfaces)
  .map(([key, surface]) => `${key}=${surface.status}`)
  .join(' ');
console.log(`capabilities manifest: ${MANIFEST_PATH}`);
console.log(`surfaces: ${summary}`);
if (!GENERATE_ONLY) {
  console.log('docs/certification.md capability table in sync.');
}
