#!/usr/bin/env node
// Assert that a packaged VSIX is self-contained: the vendored webview closure
// staged by `npm run prepackage:vsix` must be inside the archive, and when an
// extraction directory is given, every packaged file must match the pinned
// `ui/kilo-v756-webview/dist/build-manifest.json` hashes byte for byte.
//
// Usage:
//   node scripts/verify-vsix.mjs <vsix> [--min-webview-files N]
//   node scripts/verify-vsix.mjs <vsix> --extract-dir <extension-dir>
//   node scripts/verify-vsix.mjs <vsix> --ide-load
//
// --ide-load installs the VSIX into an isolated VS Code extensions directory
// when a `code` CLI harness is available. When none is available it records an
// explicit skip (never a silent pass) in
// target/certification/vsix-ide-load.json and exits 0.

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const APP_ROOT = resolve(SCRIPT_DIR, '..');
const REPO_ROOT = resolve(APP_ROOT, '..', '..');
const BUILD_MANIFEST = join(REPO_ROOT, 'ui', 'kilo-v756-webview', 'dist', 'build-manifest.json');
const OVERLAY_MANIFEST = join(
  REPO_ROOT,
  'ui',
  'kilo-v756-webview',
  'dist',
  'toolchain',
  'overlay',
  'overlay-manifest.json',
);
const APP_MANIFEST = join(APP_ROOT, 'package.json');
const WEBVIEW_PREFIX = 'extension/media/kilo-v756-webview/';
const OVERLAY_PREFIX = `${WEBVIEW_PREFIX}dist/overlay/`;
const DEFAULT_MIN_WEBVIEW_FILES = 25;

/** Additive panel markers: the panel is useless if any of these vanished. */
const COMPANION_MARKERS = [
  'faktorTaskState',
  'faktorAgents',
  'faktorCockpit',
  'faktorTournament',
  'faktorEvidence',
  'faktorBoardState',
  'faktorAgentAction',
  'faktorTournamentAction',
  'faktorEvidenceExpand',
  'faktorBoardAction',
];

let passed = 0;
let failed = 0;

function check(condition, label) {
  if (condition) {
    passed += 1;
    console.log(`PASS  ${label}`);
  } else {
    failed += 1;
    console.error(`FAIL  ${label}`);
  }
}

function sha256File(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function unzip(vsix, args) {
  return execFileSync('unzip', args, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
}

function listEntries(vsix) {
  const listing = unzip(vsix, ['-l', vsix]);
  console.log(listing.trimEnd());
  const entries = [];
  for (const line of listing.split(/\r?\n/)) {
    const match = line.match(/^\s*(\d+)\s+\S+\s+\S+\s+(.+?)\s*$/);
    if (match) {
      entries.push({ size: Number(match[1]), name: match[2] });
    }
  }
  return entries;
}

function archiveChecks(vsix, minWebviewFiles) {
  const entries = listEntries(vsix);
  const names = new Set(entries.map((entry) => entry.name));
  check(names.has('extension/out/extension.js'), 'VSIX contains extension/out/extension.js');
  check(
    names.has(`${WEBVIEW_PREFIX}dist/webview.js`),
    'VSIX contains the vendored entry bundle extension/media/kilo-v756-webview/dist/webview.js',
  );
  check(
    names.has(`${WEBVIEW_PREFIX}dist/webview.css`),
    'VSIX contains the vendored stylesheet extension/media/kilo-v756-webview/dist/webview.css',
  );
  check(
    names.has(`${WEBVIEW_PREFIX}dist/shiki-worker.js`),
    'VSIX contains the vendored shiki worker',
  );
  check(
    names.has(`${OVERLAY_PREFIX}faktor-companion.js`),
    'VSIX contains the Faktor companion panel (dist/overlay/faktor-companion.js)',
  );
  check(
    names.has(`${OVERLAY_PREFIX}faktor-companion.css`),
    'VSIX contains the Faktor companion styles (dist/overlay/faktor-companion.css)',
  );
  const webviewFiles = entries.filter((entry) => entry.name.startsWith(WEBVIEW_PREFIX));
  check(
    webviewFiles.length >= minWebviewFiles,
    `VSIX has >= ${minWebviewFiles} files under ${WEBVIEW_PREFIX} (found ${webviewFiles.length})`,
  );
  return webviewFiles.length;
}

function extractedChecks(extensionDir) {
  const required = [
    'out/extension.js',
    'out/webview.js',
    'out/kilo-bridge.js',
    'media/chat.js',
    'media/chat.css',
    'media/composer-state.js',
    'media/faktor.svg',
    'media/kilo-v756-webview/dist/webview.js',
    'media/kilo-v756-webview/dist/webview.css',
    'media/kilo-v756-webview/dist/shiki-worker.js',
    'media/kilo-v756-webview/dist/overlay/faktor-companion.js',
    'media/kilo-v756-webview/dist/overlay/faktor-companion.css',
    'media/kilo-v756-webview/dist/build-manifest.json',
  ];
  for (const relative of required) {
    check(existsSync(join(extensionDir, relative)), `packaged extension has ${relative}`);
  }

  const built = join(extensionDir, 'out', 'webview.js');
  if (existsSync(built)) {
    const source = readFileSync(built, 'utf8');
    check(
      source.includes("'media', 'kilo-v756-webview'"),
      'compiled webview resolves <extensionUri>/media/kilo-v756-webview',
    );
    check(
      !/'\.\.',\s*'\.\.'/.test(source) && !source.includes("'ui', 'kilo-v756-webview'"),
      'compiled webview has no checkout-relative ../.. escape',
    );
  }

  if (!existsSync(BUILD_MANIFEST)) {
    check(false, `pinned ${BUILD_MANIFEST} is present for byte verification`);
    return;
  }
  const manifest = JSON.parse(readFileSync(BUILD_MANIFEST, 'utf8'));
  let verified = 0;
  for (const file of manifest.vendored ?? []) {
    const packaged = join(extensionDir, 'media', 'kilo-v756-webview', ...file.path.split('/'));
    if (!existsSync(packaged)) {
      check(false, `packaged vendored file missing: ${file.path}`);
      continue;
    }
    const stat = statSync(packaged);
    const sha = sha256File(packaged);
    if (stat.size !== file.size || sha !== file.sha256) {
      check(
        false,
        `packaged vendored file diverges: ${file.path} (expected ${file.sha256}/${file.size}, got ${sha}/${stat.size})`,
      );
      continue;
    }
    verified += 1;
  }
  check(
    verified === (manifest.vendored ?? []).length && verified > 0,
    `all ${verified}/${(manifest.vendored ?? []).length} packaged vendored files match the pinned manifest`,
  );

  // Additive Faktor overlay: pinned by its own manifest and recorded in the
  // staged build manifest (`faktorOverlay`); the pinned `vendored` list is
  // untouched, so the upstream closure stays byte-identical.
  const overlayRoot = join(extensionDir, 'media', 'kilo-v756-webview', 'dist', 'overlay');
  if (!existsSync(OVERLAY_MANIFEST)) {
    check(false, `pinned ${OVERLAY_MANIFEST} is present for overlay verification`);
  } else {
    const overlayManifest = JSON.parse(readFileSync(OVERLAY_MANIFEST, 'utf8'));
    let overlayVerified = 0;
    for (const file of overlayManifest.files ?? []) {
      const packaged = join(overlayRoot, ...file.path.split('/'));
      if (!existsSync(packaged)) {
        check(false, `packaged overlay file missing: ${file.path}`);
        continue;
      }
      const stat = statSync(packaged);
      const sha = sha256File(packaged);
      if (stat.size !== file.size || sha !== file.sha256) {
        check(
          false,
          `packaged overlay file diverges: ${file.path} (expected ${file.sha256}/${file.size}, got ${sha}/${stat.size})`,
        );
        continue;
      }
      overlayVerified += 1;
    }
    check(
      overlayVerified === (overlayManifest.files ?? []).length && overlayVerified > 0,
      `all ${overlayVerified}/${(overlayManifest.files ?? []).length} packaged overlay files match the pinned overlay manifest`,
    );

    const stagedPath = join(
      extensionDir,
      'media',
      'kilo-v756-webview',
      'dist',
      'build-manifest.json',
    );
    if (!existsSync(stagedPath)) {
      check(false, 'packaged dist/build-manifest.json records the merged overlay hashes');
    } else {
      const staged = JSON.parse(readFileSync(stagedPath, 'utf8'));
      const merged = staged.faktorOverlay?.files ?? [];
      let mergedVerified = 0;
      for (const entry of merged) {
        const packaged = join(extensionDir, 'media', 'kilo-v756-webview', ...entry.path.split('/'));
        if (!existsSync(packaged)) {
          check(false, `staged manifest names a missing overlay file: ${entry.path}`);
          continue;
        }
        const stat = statSync(packaged);
        const sha = sha256File(packaged);
        if (stat.size !== entry.size || sha !== entry.sha256) {
          check(false, `staged overlay hash diverges: ${entry.path}`);
          continue;
        }
        mergedVerified += 1;
      }
      check(
        mergedVerified === (overlayManifest.files ?? []).length &&
          merged.length === (overlayManifest.files ?? []).length,
        `staged build manifest records all ${mergedVerified} merged overlay hashes`,
      );
      check(
        JSON.stringify(staged.vendored) === JSON.stringify(manifest.vendored),
        'staged build manifest keeps the pinned vendored list byte-identical',
      );
    }
  }

  const companion = join(overlayRoot, 'faktor-companion.js');
  if (existsSync(companion)) {
    const source = readFileSync(companion, 'utf8');
    const missing = COMPANION_MARKERS.filter((marker) => !source.includes(marker));
    check(
      missing.length === 0,
      `companion panel consumes the additive messages and actions (missing: ${missing.join(', ') || 'none'})`,
    );
  } else {
    check(false, 'companion panel source is present for marker verification');
  }
}

function writeIdeRecord(record) {
  const outDir = join(REPO_ROOT, 'target', 'certification');
  mkdirSync(outDir, { recursive: true });
  const path = join(outDir, 'vsix-ide-load.json');
  writeFileSync(path, `${JSON.stringify(record, null, 2)}\n`);
  console.log(`[ide-load] recorded ${path}: ${record.status}`);
}

function findCodeCli() {
  if (process.env.VSCODE_CLI !== undefined && process.env.VSCODE_CLI.length > 0) {
    return existsSync(process.env.VSCODE_CLI) ? process.env.VSCODE_CLI : null;
  }
  try {
    const found = execFileSync('sh', ['-c', 'command -v code'], { encoding: 'utf8' }).trim();
    return found.length > 0 ? found : null;
  } catch {
    return null;
  }
}

function ideLoad(vsix) {
  const code = findCodeCli();
  if (code === null) {
    const reason =
      'no `code` CLI on PATH (VS Code harness unavailable on this runner); ' +
      'the required job still ran the copied-extension assertions';
    console.log(`SKIP  IDE launch: ${reason}`);
    writeIdeRecord({ status: 'skipped', reason, vsix: resolve(vsix) });
    return;
  }
  const work = mkdtempSync(join(tmpdir(), 'faktor-vsix-ide-'));
  const extensionsDir = join(work, 'extensions');
  const userDir = join(work, 'user-data');
  try {
    execFileSync(code, ['--version'], { encoding: 'utf8' });
    execFileSync(
      code,
      [
        '--extensions-dir',
        extensionsDir,
        '--user-data-dir',
        userDir,
        '--install-extension',
        resolve(vsix),
        '--force',
      ],
      { encoding: 'utf8' },
    );
    const listed = execFileSync(
      code,
      ['--extensions-dir', extensionsDir, '--user-data-dir', userDir, '--list-extensions', '--show-versions'],
      { encoding: 'utf8' },
    );
    console.log(listed.trimEnd());
    const manifest = JSON.parse(readFileSync(APP_MANIFEST, 'utf8'));
    const extensionId = `${manifest.publisher}.${manifest.name}`.replace(/\./g, '\\.');
    check(
      new RegExp(`^${extensionId}@`, 'm').test(listed),
      `VS Code harness loaded the packaged extension ${manifest.publisher}.${manifest.name} (${code})`,
    );
    writeIdeRecord({ status: failed === 0 ? 'loaded' : 'failed', reason: '', vsix: resolve(vsix) });
  } catch (error) {
    writeIdeRecord({
      status: 'failed',
      reason: String(error && error.message ? error.message : error),
      vsix: resolve(vsix),
    });
    console.error(`FAIL  IDE launch via ${code}: ${error && error.message ? error.message : error}`);
    failed += 1;
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
}

function parseArgs(argv) {
  const options = { minWebviewFiles: DEFAULT_MIN_WEBVIEW_FILES, extractDir: null, ideLoad: false, vsix: null };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--extract-dir') {
      options.extractDir = resolve(argv[(i += 1)] ?? '');
    } else if (arg === '--min-webview-files') {
      options.minWebviewFiles = Number(argv[(i += 1)] ?? '');
    } else if (arg === '--ide-load') {
      options.ideLoad = true;
    } else if (!arg.startsWith('--')) {
      options.vsix = resolve(arg);
    }
  }
  return options;
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.vsix === null || !existsSync(options.vsix)) {
    console.error('usage: node scripts/verify-vsix.mjs <vsix> [--min-webview-files N] [--extract-dir <dir>] [--ide-load]');
    return 2;
  }
  if (!Number.isInteger(options.minWebviewFiles) || options.minWebviewFiles < 1) {
    console.error(`invalid --min-webview-files: ${options.minWebviewFiles}`);
    return 2;
  }
  console.log(`[verify-vsix] ${options.vsix}`);
  archiveChecks(options.vsix, options.minWebviewFiles);
  if (options.extractDir !== null) {
    check(existsSync(options.extractDir), `extracted extension dir exists: ${options.extractDir}`);
    if (existsSync(options.extractDir)) {
      extractedChecks(options.extractDir);
    }
  }
  if (options.ideLoad) {
    ideLoad(options.vsix);
  }
  console.log(`\n[verify-vsix] ${passed} passed, ${failed} failed`);
  return failed > 0 ? 1 : 0;
}

process.exit(main());
