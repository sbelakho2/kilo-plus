#!/usr/bin/env node
// Frozen upstream UI manifest verifier.
//
//   node scripts/verify-upstream.mjs            verify ui/upstream.json hashes
//   node scripts/verify-upstream.mjs --verify   same (explicit)
//   node scripts/verify-upstream.mjs --write    re-hash ui/ and rewrite the manifest
//   node scripts/verify-upstream.mjs --self-test  tamper a sandbox copy -> must fail
//
// The manifest (`ui/upstream.json`) pins the exact upstream commit and the
// SHA-256 of every vendored file. Divergence is a hard failure: a missing
// file, a modified byte, or an unexpected file inside a vendored tree
// (except generated `dist/` bundles, which are build output, never vendored
// source) exits nonzero. There is no "skip" mode on purpose.

import { createHash } from 'node:crypto';
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, '..');
const UI_ROOT = join(REPO_ROOT, 'ui');
const MANIFEST_PATH = join(UI_ROOT, 'upstream.json');

/** Directories that hold generated build output, not vendored source. */
const GENERATED_DIRS = new Set(['dist']);

const PIN_DEFAULTS = {
  repository: 'https://github.com/Kilo-Org/kilocode',
  tag: 'v7.5.6',
  commit: 'fa02955bfa17b60e57e0d7406d200a73337472ee',
  paths: {
    'packages/kilo-vscode/webview-ui': 'ui/kilo-v756-webview',
    'packages/kilo-ui': 'ui/kilo-ui',
  },
  fetch:
    'git clone --filter=blob:none --no-checkout --depth 1 --branch v7.5.6 ' +
    'https://github.com/Kilo-Org/kilocode <tmp>/repo && ' +
    'git -C <tmp>/repo sparse-checkout set packages/kilo-vscode/webview-ui packages/kilo-ui && ' +
    'git -C <tmp>/repo checkout fa02955bfa17b60e57e0d7406d200a73337472ee && ' +
    'rsync -a --delete <tmp>/repo/packages/kilo-vscode/webview-ui/ ui/kilo-v756-webview/ && ' +
    'rsync -a --delete <tmp>/repo/packages/kilo-ui/ ui/kilo-ui/',
  hashAlgorithm: 'sha256',
};

const PIN_FIELDS = ['repository', 'tag', 'commit', 'paths', 'fetch', 'hashAlgorithm'];

export function sha256File(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

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
      if (GENERATED_DIRS.has(entry.name)) {
        continue;
      }
      if (prefix === '' && entry.name === 'LICENSES') {
        continue;
      }
      out.push(...walkFiles(abs, rel));
      continue;
    }
    if (entry.isFile()) {
      if (prefix === '' && entry.name === 'upstream.json') {
        continue;
      }
      out.push({ rel, abs, symlink: false });
    }
  }
  return out;
}

/** Compute the full manifest (sorted, POSIX separators) for the ui/ tree. */
export function buildManifest() {
  const manifest = {};
  for (const entry of walkFiles(UI_ROOT)) {
    const hash = sha256File(entry.abs);
    manifest[entry.rel.split(sep).join('/')] = hash;
  }
  return Object.fromEntries(Object.entries(manifest).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
}

/**
 * Verify a manifest against a filesystem root.
 *
 * @param {string} root absolute directory the manifest paths are relative to
 * @param {Record<string, string>} fileHashes vendored-relative path -> sha256
 * @param {{ strictExtras?: boolean }} [options]
 * @returns {{ ok: boolean, checked: number, errors: string[] }}
 */
export function verifyTree(root, fileHashes, options = {}) {
  const errors = [];
  const expected = new Set(Object.keys(fileHashes));
  let checked = 0;
  for (const [rel, want] of Object.entries(fileHashes)) {
    const abs = join(root, ...rel.split('/'));
    if (!existsSync(abs)) {
      errors.push(`missing: ${rel}`);
      continue;
    }
    let stat;
    try {
      stat = statSync(abs);
    } catch (error) {
      errors.push(`unreadable: ${rel} (${error && error.message ? error.message : error})`);
      continue;
    }
    if (!stat.isFile()) {
      errors.push(`not a regular file: ${rel}`);
      continue;
    }
    const got = sha256File(abs);
    if (got !== want) {
      errors.push(`hash mismatch: ${rel} (expected ${want}, got ${got})`);
      continue;
    }
    checked += 1;
  }
  if (options.strictExtras) {
    for (const entry of walkFiles(root)) {
      if (entry.symlink) {
        errors.push(`symlink not allowed in vendored tree: ${entry.rel}`);
        continue;
      }
      if (!expected.has(entry.rel)) {
        errors.push(`unexpected file: ${entry.rel}`);
      }
    }
  }
  return { ok: errors.length === 0, checked, errors };
}

export function loadManifest() {
  if (!existsSync(MANIFEST_PATH)) {
    throw new Error(`${MANIFEST_PATH} does not exist; run scripts/vendor-upstream.sh`);
  }
  return JSON.parse(readFileSync(MANIFEST_PATH, 'utf8'));
}

function writeManifest(previous) {
  const manifest = buildManifest();
  let totalBytes = 0;
  for (const rel of Object.keys(manifest)) {
    totalBytes += statSync(join(UI_ROOT, ...rel.split('/'))).size;
  }
  const source = previous ?? PIN_DEFAULTS;
  const next = {};
  for (const field of PIN_FIELDS) {
    if (field === 'hashAlgorithm') {
      next.hashAlgorithm = 'sha256';
    } else if (source[field] !== undefined) {
      next[field] = source[field];
    }
  }
  next.fileCount = Object.keys(manifest).length;
  next.totalBytes = totalBytes;
  next.file_hashes = manifest;
  writeFileSync(MANIFEST_PATH, `${JSON.stringify(next, null, 2)}\n`);
  return { fileCount: next.fileCount, totalBytes };
}

function verifyCommand() {
  const manifest = loadManifest();
  if (manifest.hashAlgorithm !== 'sha256') {
    console.error(`unknown hashAlgorithm: ${manifest.hashAlgorithm}`);
    return 1;
  }
  const hashes = manifest.file_hashes ?? {};
  const { ok, checked, errors } = verifyTree(UI_ROOT, hashes, { strictExtras: true });
  if (!ok) {
    for (const error of errors) {
      console.error(`DIVERGENCE ${error}`);
    }
    console.error(`${errors.length} divergence(s); upstream tree is not the pinned commit`);
    return 1;
  }
  if (checked !== manifest.fileCount) {
    console.error(`manifest fileCount ${manifest.fileCount} !== verified ${checked}`);
    return 1;
  }
  let bytes = 0;
  for (const rel of Object.keys(hashes)) {
    bytes += statSync(join(UI_ROOT, ...rel.split('/'))).size;
  }
  console.log(`upstream OK: ${manifest.repository} ${manifest.tag} ${manifest.commit}`);
  console.log(`${checked} files, ${bytes} bytes, sha256 verified`);
  return 0;
}

function selfTest() {
  const manifest = loadManifest();
  const entries = Object.entries(manifest.file_hashes ?? {});
  if (entries.length === 0) {
    console.error('SELFTEST FAIL: manifest has no file hashes');
    return 1;
  }
  const [rel, hash] = entries[0];
  const sandbox = mkdtempSync(join(tmpdir(), 'faktor-upstream-selftest-'));
  const sandboxFile = join(sandbox, ...rel.split('/'));
  mkdirSync(dirname(sandboxFile), { recursive: true });
  cpSync(join(UI_ROOT, ...rel.split('/')), sandboxFile);
  const sandboxManifest = { [rel]: hash };
  let failures = 0;

  const clean = verifyTree(sandbox, sandboxManifest, { strictExtras: true });
  if (clean.ok && clean.checked === 1) {
    console.log(`PASS  clean copy verifies (${rel})`);
  } else {
    failures += 1;
    console.error(`FAIL  clean copy rejected: ${clean.errors.join('; ')}`);
  }

  writeFileSync(sandboxFile, `${readFileSync(sandboxFile, 'utf8')}tampered`);
  const tampered = verifyTree(sandbox, sandboxManifest, { strictExtras: true });
  if (!tampered.ok && tampered.errors.some((error) => error.startsWith('hash mismatch'))) {
    console.log(`PASS  tampered file fails (${relative(sandbox, sandboxFile)})`);
  } else {
    failures += 1;
    console.error('FAIL  tampered file was accepted');
  }

  rmSync(sandboxFile);
  const missing = verifyTree(sandbox, sandboxManifest, { strictExtras: true });
  if (!missing.ok && missing.errors.some((error) => error.startsWith('missing'))) {
    console.log('PASS  missing file fails');
  } else {
    failures += 1;
    console.error('FAIL  missing file was accepted');
  }

  writeFileSync(sandboxFile, 'not the pinned bytes');
  writeFileSync(join(sandbox, 'intruder.txt'), 'extra');
  const extra = verifyTree(sandbox, { [rel]: sha256File(sandboxFile) }, { strictExtras: true });
  if (!extra.ok && extra.errors.some((error) => error.startsWith('unexpected file'))) {
    console.log('PASS  unexpected file fails');
  } else {
    failures += 1;
    console.error('FAIL  unexpected file was accepted');
  }

  rmSync(sandbox, { recursive: true, force: true });
  if (failures > 0) {
    console.error(`SELFTEST FAIL: ${failures} check(s) failed`);
    return 1;
  }
  console.log('SELFTEST OK');
  return 0;
}

function main(argv) {
  const mode = argv[0] ?? '--verify';
  try {
    if (mode === '--write') {
      const previous = existsSync(MANIFEST_PATH) ? loadManifest() : null;
      const result = writeManifest(previous);
      console.log(`wrote ${MANIFEST_PATH}: ${result.fileCount} files, ${result.totalBytes} bytes`);
      return 0;
    }
    if (mode === '--verify') {
      return verifyCommand();
    }
    if (mode === '--self-test') {
      return selfTest();
    }
    console.error(`usage: verify-upstream.mjs [--verify|--write|--self-test]`);
    return 2;
  } catch (error) {
    console.error(`FATAL: ${error && error.stack ? error.stack : error}`);
    return 1;
  }
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  process.exit(main(process.argv.slice(2)));
}
