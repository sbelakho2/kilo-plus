#!/usr/bin/env node
// Lightweight visual/static check for the frozen webview surface.
//
// The Rust visual suite (tests/visual) renders screenshot fixtures; this
// script is the offline, dependency-free counterpart for the vendored UI
// wiring: it asserts the shell the extension serves for the pinned bundle is
// structurally sound (nonce, default-deny CSP, local-only resources, root
// element, worker globals), that the vendored tree and its manifest agree,
// and that the built-in fallback media exists for the no-bundle case.
//
//   node scripts/webview-visual-check.mjs

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildVendoredWebviewHtml, vendoredCsp } from '../apps/vscode/src/kilo-bridge.ts';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
let failed = 0;

function check(label, fn) {
  try {
    fn();
    console.log(`PASS  ${label}`);
  } catch (error) {
    failed += 1;
    console.error(`FAIL  ${label} — ${error && error.message ? error.message : error}`);
  }
}

const webviewRoot = join(REPO_ROOT, 'ui', 'kilo-v756-webview');
const kiloUiRoot = join(REPO_ROOT, 'ui', 'kilo-ui');

check('vendored source trees are present at the pinned paths', () => {
  for (const path of [
    join(webviewRoot, 'src', 'index.tsx'),
    join(webviewRoot, 'src', 'App.tsx'),
    join(webviewRoot, 'tsconfig.json'),
    join(kiloUiRoot, 'package.json'),
    join(kiloUiRoot, 'src', 'index.ts'),
  ]) {
    if (!existsSync(path)) {
      throw new Error(`missing ${path}`);
    }
  }
});

check('upstream manifest agrees with the vendored directories', () => {
  const manifest = JSON.parse(readFileSync(join(REPO_ROOT, 'ui', 'upstream.json'), 'utf8'));
  if (!manifest.paths['packages/kilo-vscode/webview-ui']?.endsWith('kilo-v756-webview')) {
    throw new Error('manifest does not map webview-ui to kilo-v756-webview');
  }
  if (!manifest.paths['packages/kilo-ui']?.endsWith('kilo-ui')) {
    throw new Error('manifest does not map kilo-ui to kilo-ui');
  }
  if (manifest.hashAlgorithm !== 'sha256' || manifest.fileCount !== Object.keys(manifest.file_hashes).length) {
    throw new Error('manifest hash bookkeeping is inconsistent');
  }
});

check('bundle detection contract matches what is committed', () => {
  const script = join(webviewRoot, 'dist', 'webview.js');
  const style = join(webviewRoot, 'dist', 'webview.css');
  const scriptPresent = existsSync(script);
  if (scriptPresent !== existsSync(style)) {
    throw new Error('dist/webview.js and dist/webview.css must be present or absent together');
  }
  if (!scriptPresent) {
    // No bundle committed: the extension must fall back to media/chat.js.
    for (const path of [
      join(REPO_ROOT, 'apps', 'vscode', 'media', 'chat.js'),
      join(REPO_ROOT, 'apps', 'vscode', 'media', 'chat.css'),
    ]) {
      if (!existsSync(path)) {
        throw new Error(`no vendored bundle and no fallback ${path}`);
      }
    }
  }
});

check('vendored shell is default-deny, nonce-only and local-only', () => {
  const nonce = 'visualcheck';
  const html = buildVendoredWebviewHtml({
    cspSource: 'vscode-webview://visual',
    nonce,
    scriptUri: 'vscode-webview://visual/dist/webview.js',
    styleUri: 'vscode-webview://visual/dist/webview.css',
    iconsBaseUri: 'vscode-webview://visual/assets/icons',
    workerUri: 'vscode-webview://visual/dist/shiki-worker.js',
    title: 'Faktor',
    sidebar: '',
    topBar: false,
  });
  const csp = vendoredCsp('vscode-webview://visual', nonce);
  if (!csp.startsWith("default-src 'none'")) throw new Error('CSP is not default-deny');
  if (!csp.includes("connect-src vscode-webview://visual")) throw new Error('CSP connect-src must stay local');
  if (csp.includes('http://') || csp.includes('https://')) throw new Error('CSP must not allow remote origins');
  if (/(src|href)="https?:/.test(html)) throw new Error('shell must not reference remote resources');
  if (!html.includes('id="root"')) throw new Error('shell lacks the Solid mount point');
  if (!html.includes('KILO_SHIKI_WORKER_URI')) throw new Error('shell lacks the shiki worker bootstrap');
  const nonceCount = (html.match(new RegExp(`nonce="${nonce}"`, 'g')) ?? []).length;
  if (nonceCount !== 2) throw new Error(`expected 2 nonce scripts, found ${nonceCount}`);
});

check('vendored source never loads remote code', () => {
  const index = readFileSync(join(webviewRoot, 'src', 'index.tsx'), 'utf8');
  if (/https?:\/\//.test(index)) {
    throw new Error('src/index.tsx references a remote origin');
  }
});

console.log(`\n${failed === 0 ? 'VISUAL CHECK OK' : `${failed} visual check(s) failed`}`);
process.exit(failed > 0 ? 1 : 0);
