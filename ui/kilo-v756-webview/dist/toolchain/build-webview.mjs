#!/usr/bin/env node
// Reproducible build + vendor step for the pinned Kilo v7.5.6 chat webview.
//
// The Faktor vendored tree (`ui/kilo-v756-webview`) is a sparse checkout of
// upstream `packages/kilo-vscode/webview-ui` plus `packages/kilo-ui`; it
// deliberately carries no package.json / lockfile / build config (those live
// outside the sparse paths), so `npm ci` / `npm install` cannot run there:
//
//   npm error enoent Could not read package.json .../ui/kilo-v756-webview/package.json
//
// This script reconstructs the exact upstream build from the pinned commit
// (Kilo-Org/kilocode v7.5.6, fa02955bfa17b60e57e0d7406d200a73337472ee) with
// the upstream toolchain (bun + packages/kilo-vscode/esbuild.js), then vendors
// only the chat-webview closure into this directory and writes:
//
//   dist/build-manifest.json   full upstream dist listing + vendored hashes
//   dist/visual-baseline.json  visual-parity baseline (hashes/sizes + DOM/CSS
//                              artifacts; `render` filled by the gate)
//
// No upstream source is edited. Usage:
//
//   node dist/toolchain/build-webview.mjs
//       full build: clone pin -> bun install --ignore-scripts -> esbuild
//   node dist/toolchain/build-webview.mjs --from <upstream-dist> [--icons <dir>]
//       vendor an already-built upstream dist (offline path)
//   node dist/toolchain/build-webview.mjs --verify
//       re-verify the vendored closure against build-manifest.json
//
// Environment: BUN=/path/to/bun overrides binary discovery; TMPDIR is honored.

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import {
  existsSync,
  mkdtempSync,
  mkdirSync,
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
const WEBVIEW_ROOT = resolve(SCRIPT_DIR, '..', '..');
const DIST = join(WEBVIEW_ROOT, 'dist');
const REPO_ROOT = resolve(WEBVIEW_ROOT, '..', '..');

export const PIN = {
  repository: 'https://github.com/Kilo-Org/kilocode',
  tag: 'v7.5.6',
  commit: 'fa02955bfa17b60e57e0d7406d200a73337472ee',
  upstreamPackage: 'packages/kilo-vscode',
  upstreamDist: 'packages/kilo-vscode/dist',
  upstreamIcons: 'packages/kilo-vscode/assets/icons',
};

/** Files esbuild.js emits for the chat webview entry (`webview-ui/src/index.tsx`). */
const ENTRY_FILES = ['webview.js', 'webview.css', 'shiki-worker.js', 'markdown-shiki-worker.js'];

/** Optional upstream entry shapes; included when produced by a future build. */
const OPTIONAL_ENTRY_FILES = ['index.html'];

/** Markers that must survive bundling: the frozen bridge ABI and the CSS surface. */
const DOM_CSS_ARTIFACTS = {
  js: [
    { path: 'dist/webview.js', marker: 'webviewReady', min: 1 },
    { path: 'dist/webview.js', marker: 'KILO_SHIKI_WORKER_URI', min: 1 },
    { path: 'dist/webview.js', marker: 'sendMessage', min: 1 },
    { path: 'dist/webview.js', marker: 'loadMessages', min: 1 },
    { path: 'dist/webview.js', marker: 'messagesLoaded', min: 1 },
    { path: 'dist/webview.js', marker: 'ICONS_BASE_URI', min: 1 },
  ],
  css: [
    { path: 'dist/webview.css', marker: '--vscode-', min: 100 },
    { path: 'dist/webview.css', marker: 'prompt-input', min: 10 },
    { path: 'dist/webview.css', marker: 'data-theme', min: 1 },
  ],
};

export function sha256File(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

function sortByPath(entries) {
  return entries.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
}

/** Every file under `root`, path-relative, POSIX separators; refuses symlinks. */
function walk(root) {
  const out = [];
  const visit = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const rel = prefix.length > 0 ? `${prefix}/${entry.name}` : entry.name;
      const abs = join(dir, entry.name);
      if (entry.isSymbolicLink()) {
        throw new Error(`refusing symlink in build output: ${rel}`);
      }
      if (entry.isDirectory()) {
        visit(abs, rel);
      } else if (entry.isFile()) {
        out.push({ path: rel, abs, size: statSync(abs).size, sha256: sha256File(abs) });
      }
    }
  };
  visit(root, '');
  return sortByPath(out);
}

/** Local assets referenced from a CSS bundle (`url(...)`, query stripped). */
function cssAssetRefs(cssPath) {
  const css = readFileSync(cssPath, 'utf8');
  const refs = new Set();
  for (const match of css.matchAll(/url\(([^)]*)\)/g)) {
    let ref = match[1].trim().replace(/^['"]|['"]$/g, '');
    ref = ref.split(/[?#]/)[0];
    if (ref.length === 0 || /^[a-z][a-z0-9+.-]*:/i.test(ref) || ref.startsWith('//')) {
      continue;
    }
    refs.add(ref.replace(/^\.\//, ''));
  }
  return [...refs].sort();
}

/** Entry + worker + referenced fonts from an upstream (or previously built) dist. */
export function buildClosure(upstreamDist, iconsDir) {
  const files = [];
  const missing = [];
  for (const name of [...ENTRY_FILES, ...OPTIONAL_ENTRY_FILES]) {
    const abs = join(upstreamDist, name);
    if (existsSync(abs)) {
      files.push({ rel: name, abs, size: statSync(abs).size });
    } else if (!OPTIONAL_ENTRY_FILES.includes(name)) {
      missing.push(name);
    }
  }
  if (missing.length > 0) {
    throw new Error(`upstream dist is not a chat-webview build; missing ${missing.join(', ')}`);
  }
  const css = files.find((file) => file.rel === 'webview.css');
  if (css === undefined) {
    throw new Error('upstream dist has no webview.css to derive font assets from');
  }
  for (const ref of cssAssetRefs(css.abs)) {
    const abs = join(upstreamDist, ref);
    if (!existsSync(abs) || !statSync(abs).isFile()) {
      throw new Error(`webview.css references a missing asset: ${ref}`);
    }
    files.push({ rel: ref, abs, size: statSync(abs).size });
  }
  if (iconsDir !== null && existsSync(iconsDir)) {
    for (const icon of walk(iconsDir)) {
      files.push({ rel: `assets/icons/${icon.path}`, abs: icon.abs, size: icon.size });
    }
  }
  return sortByPath(
    files.map(({ rel, abs, size }) => ({ path: rel, abs, size, sha256: sha256File(abs) })),
  );
}

function toolVersion(bin, args) {
  try {
    return execFileSync(bin, args, { encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
}

function findBun() {
  if (process.env.BUN !== undefined && process.env.BUN.length > 0) {
    return process.env.BUN;
  }
  try {
    const path = execFileSync('sh', ['-c', 'command -v bun'], { encoding: 'utf8' }).trim();
    return path.length > 0 ? path : null;
  } catch {
    return null;
  }
}

function fullBuild() {
  const bun = findBun();
  if (bun === null) {
    throw new Error(
      'bun not found on PATH (set BUN=/path/to/bun). The upstream build script is `bun esbuild.js`; ' +
        'install bun 1.3.14 (e.g. `npm install -g bun@1.3.14`).',
    );
  }
  const tmp = mkdtempSync(join(tmpdir(), 'faktor-webview-build-'));
  try {
    const repo = join(tmp, 'repo');
    execFileSync(
      'git',
      [
        'clone',
        '--filter=blob:none',
        '--no-checkout',
        '--depth',
        '1',
        '--branch',
        PIN.tag,
        PIN.repository,
        repo,
      ],
      { stdio: 'inherit' },
    );
    execFileSync('git', ['-C', repo, 'checkout', PIN.commit], { stdio: 'inherit' });
    const head = execFileSync('git', ['-C', repo, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
    if (head !== PIN.commit) {
      throw new Error(`fetched HEAD ${head} != pinned ${PIN.commit}`);
    }
    const pkgDir = join(repo, PIN.upstreamPackage);
    execFileSync(bun, ['install', '--ignore-scripts'], { cwd: repo, stdio: 'inherit' });
    execFileSync(bun, ['esbuild.js', '--production'], { cwd: pkgDir, stdio: 'inherit' });
    return {
      tmp,
      dist: join(repo, PIN.upstreamDist),
      icons: join(repo, PIN.upstreamIcons),
      bunVersion: toolVersion(bun, ['--version']),
      esbuildVersion: toolVersion(join(repo, 'node_modules', '.bin', 'esbuild'), ['--version']),
    };
  } catch (error) {
    rmSync(tmp, { recursive: true, force: true });
    throw error;
  }
}

function writeManifests(closure, upstreamListing, build) {
  const pin = { repository: PIN.repository, tag: PIN.tag, commit: PIN.commit };
  const vendored = closure.map(({ path, sha256, size }) => ({ path: `dist/${path}`, sha256, size }));
  const buildManifest = {
    schema: 'faktor.webview-build-manifest/1',
    pinned: pin,
    upstreamPackage: PIN.upstreamPackage,
    command: 'bun esbuild.js --production',
    toolchain: {
      node: process.version,
      bun: build.bunVersion ?? null,
      esbuild: build.esbuildVersion ?? null,
    },
    upstreamDist: {
      fileCount: upstreamListing.length,
      totalBytes: upstreamListing.reduce((sum, file) => sum + file.size, 0),
      files: upstreamListing.map(({ path, sha256, size }) => ({ path, sha256, size })),
    },
    vendored,
  };
  const visualBaseline = {
    schema: 'faktor.webview-visual-baseline/1',
    pinned: pin,
    files: Object.fromEntries(
      closure.map(({ path, sha256, size }) => [`dist/${path}`, { sha256, size }]),
    ),
    artifacts: DOM_CSS_ARTIFACTS,
    render: null,
  };
  writeFileSync(join(DIST, 'build-manifest.json'), `${JSON.stringify(buildManifest, null, 2)}\n`);
  writeFileSync(join(DIST, 'visual-baseline.json'), `${JSON.stringify(visualBaseline, null, 2)}\n`);
  return { vendored: vendored.length, upstream: upstreamListing.length };
}

function clearDist() {
  for (const entry of readdirSync(DIST, { withFileTypes: true })) {
    if (entry.name === 'toolchain') {
      continue; // this script + its package metadata are tooling, not output
    }
    rmSync(join(DIST, entry.name), { recursive: true, force: true });
  }
}

function vendor({ upstreamDist, iconsDir, bunVersion, esbuildVersion }) {
  const closure = buildClosure(upstreamDist, iconsDir);
  const upstreamListing = walk(upstreamDist);
  clearDist();
  for (const file of closure) {
    const target = join(DIST, ...file.path.split('/'));
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, readFileSync(file.abs));
  }
  const stats = writeManifests(closure, upstreamListing, { bunVersion, esbuildVersion });
  const bytes = closure.reduce((sum, file) => sum + file.size, 0);
  console.log(
    `vendored ${stats.vendored} chat-webview files (${bytes} bytes) from ` +
      `${stats.upstream} upstream dist files`,
  );
  for (const file of closure) {
    console.log(`  ${file.sha256.slice(0, 12)}  ${String(file.size).padStart(9)}  ${file.path}`);
  }
}

function verifyVendored() {
  const manifestPath = join(DIST, 'build-manifest.json');
  if (!existsSync(manifestPath)) {
    throw new Error(`${manifestPath} is missing; run build-webview.mjs first`);
  }
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  let failures = 0;
  for (const file of manifest.vendored ?? []) {
    const abs = join(WEBVIEW_ROOT, ...file.path.split('/'));
    if (!existsSync(abs)) {
      console.error(`MISSING ${file.path}`);
      failures += 1;
      continue;
    }
    const sha = sha256File(abs);
    if (sha !== file.sha256 || statSync(abs).size !== file.size) {
      console.error(`DIVERGENCE ${file.path}: expected ${file.sha256}/${file.size}, got ${sha}/${statSync(abs).size}`);
      failures += 1;
    }
  }
  if (failures > 0) {
    console.error(`${failures} divergence(s) between dist/ and build-manifest.json`);
    return 1;
  }
  console.log(`dist OK: ${(manifest.vendored ?? []).length} vendored files match build-manifest.json`);
  return 0;
}

function main(argv) {
  const fromIndex = argv.indexOf('--from');
  if (argv[0] === '--verify') {
    return verifyVendored();
  }
  try {
    if (fromIndex !== -1) {
      const upstreamDist = resolve(argv[fromIndex + 1] ?? '');
      const iconsIndex = argv.indexOf('--icons');
      const iconsDir = iconsIndex !== -1 ? resolve(argv[iconsIndex + 1] ?? '') : join(upstreamDist, '..', 'assets', 'icons');
      if (!existsSync(upstreamDist)) {
        throw new Error(`--from directory does not exist: ${upstreamDist}`);
      }
      vendor({ upstreamDist, iconsDir });
      return 0;
    }
    const built = fullBuild();
    try {
      vendor({
        upstreamDist: built.dist,
        iconsDir: built.icons,
        bunVersion: built.bunVersion,
        esbuildVersion: built.esbuildVersion,
      });
    } finally {
      rmSync(built.tmp, { recursive: true, force: true });
    }
    return 0;
  } catch (error) {
    console.error(`BLOCKED: ${error && error.message ? error.message : error}`);
    return 2;
  }
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
  process.exit(main(process.argv.slice(2)));
}
