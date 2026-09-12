#!/usr/bin/env node
// Stage the pinned Kilo v7.5.6 chat-webview build closure inside the
// extension so the VSIX is self-contained, and merge the additive Faktor
// companion overlay on top.
//
// Runtime resolution (src/webview.ts `vendoredRoot`) only ever looks inside
// the installed extension:
//
//   <extensionUri>/media/kilo-v756-webview/dist/{webview.js,webview.css,...}
//   <extensionUri>/media/kilo-v756-webview/dist/overlay/faktor-companion.{js,css}
//
// This script turns the repo checkout's pinned vendored tree
// (ui/kilo-v756-webview/dist, hashed by ui/upstream.json and
// dist/build-manifest.json) into exactly that layout. It reuses the pinned
// toolchain (ui/kilo-v756-webview/dist/toolchain/build-webview.mjs) for the
// closure enumeration and scripts/verify-upstream.mjs for the source pin,
// and it fails loudly when the vendored source is missing or diverged.
//
// ---------------------------------------------------------------- overlay
//
// The Faktor companion panel lives in the overlay directory
//
//   ui/kilo-v756-webview/dist/toolchain/overlay/
//     overlay-manifest.json      pins every overlay file (sha256 + size)
//     faktor-companion.js        the panel (plain browser JS)
//     faktor-companion.css       the panel styles
//
// and is merged to `dist/overlay/` in the staged extension at this step.
// Rationale for the exact location:
//   - upstream source must stay byte-identical: the overlay is NEVER placed
//     beside the frozen sources, only inside generated `dist/` output;
//   - the pinned toolchain's `clearDist()` preserves `dist/toolchain/`, so
//     a pinned bundle rebuild cannot delete the overlay;
//   - scripts/verify-upstream.mjs skips generated `dist/` trees and
//     scripts/webview-visual-check.mjs already allows `dist/toolchain/**`,
//     so both existing gates stay green without weakening them;
//   - the overlay has its own hash manifest and the staged
//     `dist/build-manifest.json` records a `faktorOverlay` section with the
//     merged file hashes, so verify-vsix can prove what shipped.
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
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
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
const OVERLAY_DIR = join(SOURCE_DIST, 'toolchain', 'overlay');
const OVERLAY_MANIFEST = join(OVERLAY_DIR, 'overlay-manifest.json');
const OVERLAY_MANIFEST_NAME = 'overlay-manifest.json';
/** Where the overlay lands inside the staged/installed extension. */
const STAGED_OVERLAY_DIR = 'dist/overlay';

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

function sha256File(path) {
  return sha256(readFileSync(path));
}

// ------------------------------------------------------------------ overlay

/** Every regular file under `root` (POSIX-relative); symlinks are refused. */
function walkFiles(root, prefix = '') {
  const out = [];
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    const rel = prefix.length > 0 ? `${prefix}/${entry.name}` : entry.name;
    const abs = join(root, entry.name);
    if (entry.isSymbolicLink()) {
      out.push({ rel, abs, symlink: true });
      continue;
    }
    if (entry.isDirectory()) {
      out.push(...walkFiles(abs, rel));
      continue;
    }
    if (entry.isFile()) {
      out.push({ rel, abs, symlink: false });
    }
  }
  return out;
}

function overlayPathProblem(path) {
  if (typeof path !== 'string' || path.length === 0 || path.length > 256) {
    return 'overlay file path must be a bounded non-empty string';
  }
  if (path.includes('\\') || path.includes('\0') || path.startsWith('/')) {
    return `overlay file path ${JSON.stringify(path)} must be relative and POSIX`;
  }
  if (path.split('/').some((segment) => segment === '' || segment === '.' || segment === '..')) {
    return `overlay file path ${JSON.stringify(path)} carries traversal segments`;
  }
  return null;
}

export function loadOverlayManifest(path = OVERLAY_MANIFEST) {
  if (!existsSync(path)) {
    throw new Error(`overlay manifest missing at ${path}`);
  }
  const manifest = JSON.parse(readFileSync(path, 'utf8'));
  if (manifest.schema !== 'faktor.webview-overlay-manifest/1') {
    throw new Error(`unknown overlay manifest schema ${JSON.stringify(manifest.schema)}`);
  }
  if (manifest.target !== STAGED_OVERLAY_DIR) {
    throw new Error(`overlay manifest target must be ${STAGED_OVERLAY_DIR}`);
  }
  if (!Array.isArray(manifest.files) || manifest.files.length === 0) {
    throw new Error('overlay manifest lists no files');
  }
  const seen = new Set();
  for (const file of manifest.files) {
    const problem = overlayPathProblem(file.path);
    if (problem !== null) {
      throw new Error(problem);
    }
    if (seen.has(file.path)) {
      throw new Error(`duplicate overlay file entry ${file.path}`);
    }
    seen.add(file.path);
    if (typeof file.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(file.sha256)) {
      throw new Error(`overlay file ${file.path} has no sha256 hash`);
    }
    if (!Number.isInteger(file.size) || file.size <= 0) {
      throw new Error(`overlay file ${file.path} has no integer size`);
    }
  }
  return manifest;
}

/**
 * Verify the overlay directory against its manifest: every pinned file must
 * exist with the exact bytes, and no unexpected file (or symlink) may be
 * present. The manifest itself is the only meta file allowed.
 */
export function verifyOverlay(root = OVERLAY_DIR) {
  const errors = [];
  let checked = 0;
  let manifest;
  try {
    manifest = loadOverlayManifest(join(root, OVERLAY_MANIFEST_NAME));
  } catch (error) {
    return { ok: false, checked: 0, errors: [String(error && error.message ? error.message : error)] };
  }
  for (const file of manifest.files) {
    const abs = join(root, ...file.path.split('/'));
    if (!existsSync(abs)) {
      errors.push(`missing: ${file.path}`);
      continue;
    }
    const stat = statSync(abs);
    if (!stat.isFile()) {
      errors.push(`not a regular file: ${file.path}`);
      continue;
    }
    const got = sha256File(abs);
    if (got !== file.sha256 || stat.size !== file.size) {
      errors.push(
        `hash mismatch: ${file.path} (expected ${file.sha256}/${file.size}, got ${got}/${stat.size})`,
      );
      continue;
    }
    checked += 1;
  }
  const expected = new Set(manifest.files.map((file) => file.path));
  for (const entry of walkFiles(root)) {
    if (entry.symlink) {
      errors.push(`symlink not allowed in the overlay: ${entry.rel}`);
      continue;
    }
    if (!expected.has(entry.rel) && entry.rel !== OVERLAY_MANIFEST_NAME) {
      errors.push(`unexpected overlay file: ${entry.rel}`);
    }
  }
  return { ok: errors.length === 0, checked, errors };
}

/**
 * Merge the verified overlay into a staging root and record the merged
 * hashes in the staged build manifest (additive `faktorOverlay` section;
 * the pinned `vendored` list is copied byte-identically and never edited).
 */
export function applyOverlay(stagingRoot, root = OVERLAY_DIR) {
  const manifest = loadOverlayManifest(join(root, OVERLAY_MANIFEST_NAME));
  const merged = [];
  for (const file of manifest.files) {
    const source = join(root, ...file.path.split('/'));
    const targetRel = `${STAGED_OVERLAY_DIR}/${file.path}`;
    const target = join(stagingRoot, ...targetRel.split('/'));
    mkdirSync(dirname(target), { recursive: true });
    copyFileSync(source, target);
    const copied = readFileSync(target);
    if (copied.length !== file.size || sha256(copied) !== file.sha256) {
      throw new Error(`staged overlay copy diverges from the pin at ${targetRel}`);
    }
    merged.push({ path: targetRel, sha256: file.sha256, size: file.size });
  }
  const manifestPath = join(stagingRoot, 'dist', 'build-manifest.json');
  const sourceManifest = JSON.parse(readFileSync(BUILD_MANIFEST, 'utf8'));
  const stagedManifest = {
    ...sourceManifest,
    faktorOverlay: {
      schema: manifest.schema,
      target: manifest.target,
      upstream: manifest.upstream ?? null,
      files: merged,
    },
  };
  writeFileSync(manifestPath, `${JSON.stringify(stagedManifest, null, 2)}\n`);
  return merged;
}

// ----------------------------------------------------------------- staging

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
    throw new Error(
      `closure diverges from ${BUILD_MANIFEST}: missing=[${missing.join(', ')}] extra=[${extra.join(', ')}]`,
    );
  }
  for (const [path, want] of pinned) {
    const got = staged.get(path);
    if (got.sha256 !== want.sha256 || got.size !== want.size) {
      throw new Error(
        `closure diverges from ${BUILD_MANIFEST} at ${path}: expected ${want.sha256}/${want.size}, got ${got.sha256}/${got.size}`,
      );
    }
  }
  return pinned;
}

/**
 * Stage the pinned closure plus the verified overlay into `targetRoot`.
 * Returns counts for the caller; throws (never process.exit) so the
 * selftest can drive the exact production merge into a temp directory.
 */
export async function stageBundle(targetRoot) {
  const overlay = verifyOverlay();
  if (!overlay.ok) {
    throw new Error(`overlay verification failed: ${overlay.errors.join('; ')}`);
  }
  const closure = await loadClosure();
  const pinned = assertPinned(closure);
  let bytes = 0;
  for (const file of closure) {
    const relative = `dist/${file.path}`;
    const target = join(targetRoot, ...relative.split('/'));
    mkdirSync(dirname(target), { recursive: true });
    copyFileSync(file.abs, target);
    const copied = readFileSync(target);
    const want = pinned.get(relative);
    if (copied.length !== want.size || sha256(copied) !== want.sha256) {
      throw new Error(`staged copy diverges from the pin at ${relative}; refusing to ship it`);
    }
    bytes += copied.length;
  }
  const mergedOverlay = applyOverlay(targetRoot);
  return { files: closure.length, overlayFiles: mergedOverlay.length, bytes };
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
  if (!existsSync(OVERLAY_MANIFEST)) {
    blocked(
      `Faktor companion overlay manifest missing at ${OVERLAY_MANIFEST}; ` +
        'the additive panel must be pinned before staging',
    );
  }

  run(VERIFY_UPSTREAM, ['--verify'], 'verify-upstream --verify (pinned source tree)');
  run(TOOLCHAIN, ['--verify'], 'build-webview --verify (pinned dist closure)');

  const overlay = verifyOverlay();
  if (!overlay.ok) {
    blocked(`overlay verification failed: ${overlay.errors.join('; ')}`);
  }
  console.log(
    `[vendored-webview] overlay verified: ${overlay.checked} pinned panel file(s)`,
  );

  // Stage next to the app (same filesystem) so the final swap is a rename and
  // a crash can never leave half a bundle inside media/.
  const staging = mkdtempSync(join(APP_ROOT, '.kilo-v756-webview-stage-'));
  let stats;
  try {
    stats = await stageBundle(staging);
    rmSync(TARGET_ROOT, { recursive: true, force: true });
    renameSync(staging, TARGET_ROOT);
  } catch (error) {
    rmSync(staging, { recursive: true, force: true });
    throw error;
  }

  const tag = JSON.parse(readFileSync(BUILD_MANIFEST, 'utf8')).pinned?.tag ?? 'unknown';
  console.log(
    `[vendored-webview] staged ${stats.files} pinned files + ${stats.overlayFiles} overlay ` +
      `file(s) (${stats.bytes} pinned bytes) at ${TARGET_ROOT} (pin ${tag})`,
  );
  console.log('[vendored-webview] OK');
}

if (process.argv[1] !== undefined && pathToFileURL(process.argv[1]).href === import.meta.url) {
  main().catch((error) => {
    blocked(error && error.stack ? error.stack : String(error));
  });
}
