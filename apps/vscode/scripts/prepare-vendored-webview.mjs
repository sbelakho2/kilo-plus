#!/usr/bin/env node
// Stage the pinned Kilo v7.5.6 chat-webview build closure inside the
// extension so the VSIX is self-contained.
//
// Runtime resolution (src/webview.ts `vendoredRoot`) only ever looks inside
// the installed extension:
//
//   <extensionUri>/media/kilo-v756-webview/dist/{webview.js,webview.css,...}
//
// This script turns the repo checkout's pinned vendored tree
// (ui/kilo-v756-webview/dist, hashed by ui/upstream.json and
// dist/build-manifest.json) into exactly that layout. It reuses the pinned
// toolchain (ui/kilo-v756-webview/dist/toolchain/build-webview.mjs) for the
// closure enumeration and scripts/verify-upstream.mjs for the source pin,
// and it fails loudly when the vendored source is missing or diverged.
//
// Usage:
//   node scripts/prepare-vendored-webview.mjs
//
// Repo development only: the extension additionally honors the explicit
// FAKTOR_UI_BUNDLE environment variable at runtime to point at a bundle
// checkout. That override is never inferred from the checkout layout and is
// not used by this script.

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
} from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = resolve(SCRIPT_DIR, '..');
const REPO_ROOT = resolve(APP_ROOT, '..', '..');
const SOURCE_ROOT = join(REPO_ROOT, 'ui', 'kilo-v756-webview');
const SOURCE_DIST = join(SOURCE_ROOT, 'dist');
const TOOLCHAIN = join(SOURCE_DIST, 'toolchain', 'build-webview.mjs');
const BUILD_MANIFEST = join(SOURCE_DIST, 'build-manifest.json');
const VERIFY_UPSTREAM = join(REPO_ROOT, 'scripts', 'verify-upstream.mjs');
const TARGET_ROOT = join(APP_ROOT, 'media', 'kilo-v756-webview');

function blocked(message) {
  console.error(`BLOCKED: ${message}`);
  process.exit(2);
}

function run(script, args, label) {
  console.log(`[vendored-webview] ${label}`);
  try {
    execFileSync(process.execPath, [script, ...args], { stdio: 'inherit' });
  } catch {
    blocked(`${label} failed; refusing to stage an unverified bundle`);
  }
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

async function loadClosure() {
  const toolchainUrl = pathToFileURL(TOOLCHAIN).href;
  const toolchain = await import(toolchainUrl);
  const icons = join(SOURCE_DIST, 'assets', 'icons');
  return toolchain.buildClosure(SOURCE_DIST, existsSync(icons) ? icons : null);
}

function assertPinned(closure) {
  const manifest = JSON.parse(readFileSync(BUILD_MANIFEST, 'utf8'));
  const pinned = new Map(
    (manifest.vendored ?? []).map((file) => [file.path, { sha256: file.sha256, size: file.size }]),
  );
  const staged = new Map(closure.map((file) => [`dist/${file.path}`, file]));
  const missing = [...pinned.keys()].filter((path) => !staged.has(path));
  const extra = [...staged.keys()].filter((path) => !pinned.has(path));
  if (missing.length > 0 || extra.length > 0) {
    blocked(
      `closure diverges from ${BUILD_MANIFEST}: missing=[${missing.join(', ')}] extra=[${extra.join(', ')}]`,
    );
  }
  for (const [path, want] of pinned) {
    const got = staged.get(path);
    if (got.sha256 !== want.sha256 || got.size !== want.size) {
      blocked(
        `closure diverges from ${BUILD_MANIFEST} at ${path}: expected ${want.sha256}/${want.size}, got ${got.sha256}/${got.size}`,
      );
    }
  }
  return pinned;
}

async function main() {
  if (!existsSync(SOURCE_ROOT) || !existsSync(SOURCE_DIST)) {
    blocked(
      `vendored webview source missing at ${SOURCE_ROOT} (pinned v7.5.6); ` +
        'restore it with `bash scripts/vendor-upstream.sh`',
    );
  }
  if (!existsSync(TOOLCHAIN) || !existsSync(BUILD_MANIFEST)) {
    blocked(
      `pinned webview build closure missing under ${SOURCE_DIST}; ` +
        'rebuild it with `node ui/kilo-v756-webview/dist/toolchain/build-webview.mjs`',
    );
  }

  run(VERIFY_UPSTREAM, ['--verify'], 'verify-upstream --verify (pinned source tree)');
  run(TOOLCHAIN, ['--verify'], 'build-webview --verify (pinned dist closure)');

  const closure = await loadClosure();
  const pinned = assertPinned(closure);

  // Stage next to the app (same filesystem) so the final swap is a rename and
  // a crash can never leave half a bundle inside media/.
  const staging = mkdtempSync(join(APP_ROOT, '.kilo-v756-webview-stage-'));
  let bytes = 0;
  try {
    for (const file of closure) {
      const relative = `dist/${file.path}`;
      const target = join(staging, ...relative.split('/'));
      mkdirSync(dirname(target), { recursive: true });
      copyFileSync(file.abs, target);
      const copied = readFileSync(target);
      const want = pinned.get(relative);
      if (copied.length !== want.size || sha256(copied) !== want.sha256) {
        throw new Error(`staged copy diverges from the pin at ${relative}; refusing to ship it`);
      }
      bytes += copied.length;
    }
    rmSync(TARGET_ROOT, { recursive: true, force: true });
    renameSync(staging, TARGET_ROOT);
  } catch (error) {
    rmSync(staging, { recursive: true, force: true });
    throw error;
  }

  const tag = JSON.parse(readFileSync(BUILD_MANIFEST, 'utf8')).pinned?.tag ?? 'unknown';
  console.log(
    `[vendored-webview] staged ${closure.length} pinned files (${bytes} bytes) at ` +
      `${TARGET_ROOT} (pin ${tag})`,
  );
  console.log('[vendored-webview] OK');
}

main().catch((error) => {
  blocked(error && error.stack ? error.stack : String(error));
});
