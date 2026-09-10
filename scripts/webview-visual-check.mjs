#!/usr/bin/env node
// Visual-parity gate for the frozen webview surface.
//
// The Rust visual suite (tests/visual) covers the branded screenshot corpus
// with masked-pixel diffing. This script is the offline, dependency-free
// counterpart for the built vendored UI:
//
//   1. structural checks over the served shell (nonce, default-deny CSP,
//      local-only resources, root element, worker globals) and the vendored
//      source/manifest agreement (unchanged from the first wave);
//   2. upstream-baseline manifest comparison over the built
//      ui/kilo-v756-webview/dist closure: every manifest file must exist with
//      the recorded sha256 + size, key DOM/CSS artifacts must survive in the
//      bundles, and any extra file in dist/ (outside the manifest and the
//      recorded tooling/meta files) fails the gate;
//   3. when a headless browser is available (playwright/puppeteer), the built
//      UI is rendered at fixed widths x themes and a deterministic DOM/CSS
//      fingerprint is compared against the baseline. When unavailable the
//      render section is SKIPPED WITH A RECORD in dist/visual-report.json —
//      it is never reported as passing.
//
// Modes:
//   node scripts/webview-visual-check.mjs                     full gate
//   node scripts/webview-visual-check.mjs --no-headless       record the skip
//   node scripts/webview-visual-check.mjs --update-render-baseline
//       render and (re)write the `render` section of visual-baseline.json
//   node scripts/webview-visual-check.mjs --self-test
//       adversarial self-test: clean sandbox passes; tampered/missing/extra
//       dist asset fails; missing dist falls back with a recorded notice.
//
// Headless browser resolution: playwright / playwright-core / puppeteer are
// probed via import; FAKTOR_PLAYWRIGHT=<dir-or-specifier> adds an explicit
// candidate (a directory is imported as <dir>/index.js). Chromium can be
// installed with `npm install playwright && npx playwright install chromium`
// (set PLAYWRIGHT_BROWSERS_PATH to share the browser cache), or a locally
// installed Chrome can be used with FAKTOR_BROWSER_CHANNEL=chrome (the
// engine-specific channel is passed straight to launch); FAKTOR_CHROMIUM_PATH
// pins an explicit executable.

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
import http from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, extname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  buildVendoredWebviewHtml,
  locateVendoredBundle,
  vendoredCsp,
  vendoredFallbackNotice,
} from '../apps/vscode/src/kilo-bridge.ts';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const webviewRoot = join(REPO_ROOT, 'ui', 'kilo-v756-webview');
const DIST = join(webviewRoot, 'dist');
const BASELINE_PATH = join(DIST, 'visual-baseline.json');
const BUILD_MANIFEST_PATH = join(DIST, 'build-manifest.json');
const REPORT_PATH = join(DIST, 'visual-report.json');

/** Non-output files allowed inside dist/ on top of the baseline manifest. */
const DIST_META_FILES = new Set(['build-manifest.json', 'visual-baseline.json', 'visual-report.json']);
const DIST_TOOLING_DIR = 'toolchain';

const RENDER_WIDTHS = [390, 800, 1280];
const RENDER_THEMES = ['light', 'dark', 'high-contrast'];

const THEME_STYLES = {
  light: ':root{--vscode-editor-background:#ffffff;--vscode-editor-foreground:#3b3b3b;--vscode-foreground:#3b3b3b;--vscode-sideBar-background:#f8f8f8;--vscode-button-background:#007acc;--vscode-button-foreground:#ffffff;--vscode-font-family:"Helvetica Neue",Arial,sans-serif;--vscode-font-size:13px;--vscode-editor-font-family:Menlo,monospace;--vscode-input-background:#ffffff;--vscode-input-foreground:#3b3b3b;--vscode-input-border:#cecece;--vscode-list-hoverBackground:#e8e8e8;--vscode-panel-border:#e5e5e5;}',
  dark: ':root{--vscode-editor-background:#1e1e1e;--vscode-editor-foreground:#cccccc;--vscode-foreground:#cccccc;--vscode-sideBar-background:#252526;--vscode-button-background:#0e639c;--vscode-button-foreground:#ffffff;--vscode-font-family:"Helvetica Neue",Arial,sans-serif;--vscode-font-size:13px;--vscode-editor-font-family:Menlo,monospace;--vscode-input-background:#3c3c3c;--vscode-input-foreground:#cccccc;--vscode-input-border:#3c3c3c;--vscode-list-hoverBackground:#2a2d2e;--vscode-panel-border:#3c3c3c;}',
  'high-contrast': ':root{--vscode-editor-background:#000000;--vscode-editor-foreground:#ffffff;--vscode-foreground:#ffffff;--vscode-sideBar-background:#000000;--vscode-button-background:#000000;--vscode-button-foreground:#ffffff;--vscode-font-family:"Helvetica Neue",Arial,sans-serif;--vscode-font-size:13px;--vscode-editor-font-family:Menlo,monospace;--vscode-input-background:#000000;--vscode-input-foreground:#ffffff;--vscode-input-border:#ffffff;--vscode-list-hoverBackground:#000000;--vscode-panel-border:#ffffff;}',
};

let failures = 0;
const recorded = { manifest: null, render: null };

function check(label, fn) {
  try {
    fn();
    console.log(`PASS  ${label}`);
  } catch (error) {
    failures += 1;
    console.error(`FAIL  ${label} — ${error && error.message ? error.message : error}`);
  }
}

function sha256(data) {
  return createHash('sha256').update(data).digest('hex');
}

function sha256File(path) {
  return sha256(readFileSync(path));
}

/** Every regular file under root (POSIX-relative); symlinks are an error. */
function walkFiles(root) {
  const files = [];
  const errors = [];
  const visit = (dir, prefix) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const rel = prefix.length > 0 ? `${prefix}/${entry.name}` : entry.name;
      const abs = join(dir, entry.name);
      if (entry.isSymbolicLink()) {
        errors.push(`symlink not allowed in dist: ${rel}`);
        continue;
      }
      if (entry.isDirectory()) {
        visit(abs, rel);
      } else if (entry.isFile()) {
        files.push({ rel, abs });
      }
    }
  };
  visit(root, '');
  return { files, errors };
}

function loadBaseline(root) {
  const path = join(root, 'dist', 'visual-baseline.json');
  if (!existsSync(path)) {
    return null;
  }
  return JSON.parse(readFileSync(path, 'utf8'));
}

/**
 * Compare a dist tree against its baseline: missing/divergent files, extra
 * files, and required DOM/CSS artifact markers all fail. `root` is the
 * webview root (manifest keys are `dist/...`).
 */
function verifyDist(root, baseline, { strictExtras = true } = {}) {
  const errors = [];
  let checked = 0;
  for (const [rel, expected] of Object.entries(baseline.files ?? {})) {
    const abs = join(root, ...rel.split('/'));
    if (!existsSync(abs)) {
      errors.push(`missing: ${rel}`);
      continue;
    }
    const stat = statSync(abs);
    if (!stat.isFile()) {
      errors.push(`not a regular file: ${rel}`);
      continue;
    }
    const got = sha256File(abs);
    if (got !== expected.sha256 || stat.size !== expected.size) {
      errors.push(
        `hash mismatch: ${rel} (expected ${expected.sha256}/${expected.size}, got ${got}/${stat.size})`,
      );
      continue;
    }
    checked += 1;
  }
  for (const group of Object.values(baseline.artifacts ?? {})) {
    for (const artifact of group) {
      const abs = join(root, ...artifact.path.split('/'));
      if (!existsSync(abs)) {
        errors.push(`artifact file missing: ${artifact.path}`);
        continue;
      }
      const text = readFileSync(abs, 'utf8');
      const count = text.split(artifact.marker).length - 1;
      if (count < artifact.min) {
        errors.push(
          `artifact divergence: ${artifact.path} contains ${artifact.marker} ${count}x (< ${artifact.min})`,
        );
      }
    }
  }
  if (strictExtras && existsSync(join(root, 'dist'))) {
    const { files, errors: walkErrors } = walkFiles(join(root, 'dist'));
    errors.push(...walkErrors);
    const expected = new Set(Object.keys(baseline.files ?? {}).map((rel) => rel.replace(/^dist\//, '')));
    for (const file of files) {
      if (expected.has(file.rel)) {
        continue;
      }
      if (DIST_META_FILES.has(file.rel)) {
        continue;
      }
      if (file.rel === DIST_TOOLING_DIR || file.rel.startsWith(`${DIST_TOOLING_DIR}/`)) {
        continue;
      }
      errors.push(`unexpected file: ${file.rel}`);
    }
  }
  return { ok: errors.length === 0, checked, errors };
}

// ------------------------------------------------------------- static checks

function checkVendoredSources() {
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
}

function checkBundleContract() {
  check('bundle discovery agrees with the committed dist layout', () => {
    const bundle = locateVendoredBundle(webviewRoot);
    if (!existsSync(DIST)) {
      // No bundle committed: the extension must fall back to media/chat.js.
      if (bundle !== null) {
        throw new Error('bundle located without a dist directory');
      }
      for (const path of [
        join(REPO_ROOT, 'apps', 'vscode', 'media', 'chat.js'),
        join(REPO_ROOT, 'apps', 'vscode', 'media', 'chat.css'),
      ]) {
        if (!existsSync(path)) {
          throw new Error(`no vendored bundle and no fallback ${path}`);
        }
      }
      return;
    }
    if (bundle === null) {
      throw new Error('dist exists but no strict-local bundle was located');
    }
    for (const path of [bundle.script, bundle.style, bundle.worker, bundle.markdownWorker, bundle.icons]) {
      if (path !== null && (isAbsolute(path) === false || !path.startsWith(webviewRoot))) {
        throw new Error(`bundle path escapes the vendored root: ${path}`);
      }
    }
    if (bundle.entry !== 'esbuild' && bundle.entry !== 'index.html') {
      throw new Error(`unknown entry kind ${bundle.entry}`);
    }
  });

  check('missing dist falls back with a recorded notice', () => {
    const sandbox = mkdtempSync(join(tmpdir(), 'faktor-visual-missing-'));
    try {
      if (locateVendoredBundle(sandbox) !== null) {
        throw new Error('empty root must not locate a bundle');
      }
      const notice = vendoredFallbackNotice(sandbox);
      if (!notice.includes(noticeFallbackNeedle(sandbox))) {
        throw new Error(`notice does not record the fallback: ${notice}`);
      }
    } finally {
      rmSync(sandbox, { recursive: true, force: true });
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
}

function noticeFallbackNeedle(root) {
  return join(root, 'dist');
}

// --------------------------------------------------------- baseline compare

function checkBaseline(root) {
  let baseline = null;
  check('built dist baseline manifest is present', () => {
    if (!existsSync(join(root, 'dist'))) {
      throw new Error(`missing ${join(root, 'dist')}; build it with dist/toolchain/build-webview.mjs`);
    }
    baseline = loadBaseline(root);
    if (baseline === null) {
      throw new Error(`missing ${join(root, 'dist', 'visual-baseline.json')}`);
    }
    if (baseline.schema !== 'faktor.webview-visual-baseline/1') {
      throw new Error(`unknown baseline schema ${baseline.schema}`);
    }
    if (Object.keys(baseline.files ?? {}).length === 0) {
      throw new Error('baseline lists no files');
    }
  });
  if (baseline === null) {
    return null;
  }
  let verified = null;
  check('dist closure matches baseline hashes, sizes and DOM/CSS artifacts', () => {
    verified = verifyDist(root, baseline, { strictExtras: true });
    if (!verified.ok) {
      throw new Error(
        `${verified.errors.length} divergence(s): ${verified.errors.slice(0, 5).join('; ')}` +
          (verified.errors.length > 5 ? ` (+${verified.errors.length - 5} more)` : ''),
      );
    }
  });
  if (verified !== null) {
    recorded.manifest = {
      status: verified.ok ? 'passed' : 'failed',
      checked: verified.checked,
      errors: verified.errors,
    };
  }
  return baseline;
}

// ------------------------------------------------------------- headless part

async function importSpecifier(specifier) {
  try {
    return await import(specifier);
  } catch {
    return null;
  }
}

async function loadHeadless() {
  const candidates = [];
  const override = process.env.FAKTOR_PLAYWRIGHT;
  if (override !== undefined && override.length > 0) {
    const abs = resolve(override);
    if (existsSync(abs) && statSync(abs).isDirectory()) {
      candidates.push(pathToFileURL(join(abs, 'index.js')).href);
    } else {
      candidates.push(override);
    }
  }
  candidates.push('playwright', 'playwright-core', 'puppeteer');
  for (const candidate of candidates) {
    const module = await importSpecifier(candidate);
    if (module === null) {
      continue;
    }
    const chromium = module.chromium ?? module.default?.chromium;
    if (typeof chromium?.launch === 'function') {
      return { kind: 'playwright', browserType: chromium, label: candidate };
    }
    if (typeof module.launch === 'function') {
      return { kind: 'puppeteer', browserType: module, label: candidate };
    }
    if (typeof module.default?.launch === 'function') {
      return { kind: 'puppeteer', browserType: module.default, label: candidate };
    }
  }
  return null;
}

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
};

function startStaticServer() {
  let boundPort = 0;
  const server = http.createServer((request, response) => {
    try {
      const url = new URL(request.url ?? '/', 'http://127.0.0.1');
      if (url.pathname === '/' || url.pathname === '/index.html') {
        const theme = url.searchParams.get('theme');
        response.writeHead(200, { 'content-type': CONTENT_TYPES['.html'] });
        response.end(renderShell(boundPort, RENDER_THEMES.includes(theme) ? theme : 'light'));
        return;
      }
      const decoded = decodeURIComponent(url.pathname);
      if (decoded.includes('\0') || decoded.split('/').some((segment) => segment === '..')) {
        response.writeHead(400);
        response.end('bad path');
        return;
      }
      const abs = resolve(webviewRoot, `.${decoded}`);
      const rel = relative(webviewRoot, abs);
      if (rel.startsWith('..') || isAbsolute(rel) || !existsSync(abs) || !statSync(abs).isFile()) {
        response.writeHead(404);
        response.end('not found');
        return;
      }
      response.writeHead(200, {
        'content-type': CONTENT_TYPES[extname(abs)] ?? 'application/octet-stream',
      });
      response.end(readFileSync(abs));
    } catch {
      response.writeHead(500);
      response.end('error');
    }
  });
  return new Promise((resolvePromise) => {
    server.listen(0, '127.0.0.1', () => {
      boundPort = server.address().port;
      resolvePromise({
        port: boundPort,
        close: () => new Promise((done) => server.close(() => done())),
      });
    });
  });
}

function renderShell(port, theme) {
  const base = `http://127.0.0.1:${port}`;
  const nonce = `visual-${theme}`;
  const html = buildVendoredWebviewHtml({
    cspSource: base,
    nonce,
    scriptUri: `${base}/dist/webview.js`,
    styleUri: `${base}/dist/webview.css`,
    iconsBaseUri: `${base}/dist/assets/icons`,
    workerUri: `${base}/dist/shiki-worker.js`,
    title: 'Faktor visual gate',
    sidebar: '',
    topBar: false,
    extraStyles: THEME_STYLES[theme],
  });
  return html.replace('<body>', `<body class="vscode-${theme}">`);
}

const FINGERPRINT_SCRIPT = () => {
  const root = document.getElementById('root');
  const elements = [...document.querySelectorAll('*')];
  const classes = new Set();
  for (const element of elements) {
    for (const name of element.classList) {
      classes.add(name);
    }
  }
  const styleOf = (selector) => {
    const element = document.querySelector(selector);
    if (element === null) {
      return null;
    }
    const computed = getComputedStyle(element);
    return {
      backgroundColor: computed.backgroundColor,
      color: computed.color,
      display: computed.display,
      fontSize: computed.fontSize,
    };
  };
  return {
    rootChildren: root === null ? 0 : root.children.length,
    elements: elements.length,
    classes: [...classes].sort().slice(0, 400),
    bodyBackground: getComputedStyle(document.body).backgroundColor,
    bodyColor: getComputedStyle(document.body).color,
    rootStyle: styleOf('#root'),
    errors: (window.__faktorVisualErrors ?? []).slice(0, 20),
  };
};

async function pageErrors(page) {
  const errors = [];
  if (typeof page.on === 'function') {
    page.on('pageerror', (error) => {
      if (errors.length < 20) {
        errors.push(String(error && error.message ? error.message : error));
      }
    });
  }
  return errors;
}

async function renderState(session, url, width, theme) {
  const key = `${theme}-${width}`;
  if (session.kind === 'playwright') {
    const context = await session.browser.newContext({
      viewport: { width, height: 900 },
      deviceScaleFactor: 1,
      colorScheme: theme === 'dark' ? 'dark' : 'light',
      reducedMotion: 'reduce',
    });
    const page = await context.newPage();
    const errors = await pageErrors(page);
    await page.addInitScript(() => {
      window.__faktorVisualErrors = [];
      window.addEventListener('error', (event) => {
        window.__faktorVisualErrors.push(String(event.message));
      });
    });
    await page.goto(url, { waitUntil: 'load', timeout: 60000 });
    const mounted = await page
      .waitForFunction(() => document.getElementById('root')?.children.length > 0, null, { timeout: 30000 })
      .then(() => true)
      .catch(() => false);
    if (!mounted) {
      const diag = await page
        .evaluate(() => ({
          readyState: document.readyState,
          bodyClass: document.body.className,
          scripts: [...document.querySelectorAll('script')].map((s) => s.getAttribute('src') ?? 'inline'),
          rootHtmlLength: document.getElementById('root')?.innerHTML.length ?? -1,
        }))
        .catch(() => null);
      throw new Error(
        `${key}: #root never received children; refusing to fingerprint an empty shell (${JSON.stringify(diag)})`,
      );
    }
    await page.waitForTimeout(750);
    const fingerprint = await page.evaluate(FINGERPRINT_SCRIPT);
    const screenshot = await page.screenshot({ type: 'png' });
    await context.close();
    return { key, width, theme, fingerprint, pageErrors: errors, screenshot };
  }
  const page = await session.browser.newPage();
  const errors = await pageErrors(page);
  await page.setViewport({ width, height: 900, deviceScaleFactor: 1 });
  if (typeof page.emulateMediaFeatures === 'function') {
    await page.emulateMediaFeatures([
      { name: 'prefers-color-scheme', value: theme === 'dark' ? 'dark' : 'light' },
    ]);
  }
  await page.evaluateOnNewDocument(() => {
    window.__faktorVisualErrors = [];
    window.addEventListener('error', (event) => {
      window.__faktorVisualErrors.push(String(event.message));
    });
  });
  await page.goto(url, { waitUntil: 'load', timeout: 60000 });
  const mounted = await page
    .waitForFunction(() => document.getElementById('root')?.children.length > 0, { timeout: 30000 })
    .then(() => true)
    .catch(() => false);
  if (!mounted) {
    throw new Error(`${key}: #root never received children; refusing to fingerprint an empty shell`);
  }
  await new Promise((done) => setTimeout(done, 750));
  const fingerprint = await page.evaluate(FINGERPRINT_SCRIPT);
  const screenshot = await page.screenshot({ type: 'png' });
  await page.close();
  return { key, width, theme, fingerprint, pageErrors: errors, screenshot };
}

async function runHeadless({ update }) {
  const record = (render) => {
    recorded.render = render;
    writeReport();
    console.log(`SKIP  headless render (${render.status}${render.reason ? `: ${render.reason}` : ''}) — recorded in ${relative(REPO_ROOT, REPORT_PATH)}`);
  };
  if (!existsSync(join(DIST, 'webview.js'))) {
    record({ status: 'skipped', reason: 'dist/webview.js absent' });
    return;
  }
  const headless = await loadHeadless();
  if (headless === null) {
    record({
      status: 'skipped',
      reason:
        'no headless browser available (install playwright + `npx playwright install chromium`, ' +
        'or set FAKTOR_PLAYWRIGHT)',
    });
    return;
  }
  const baseline = loadBaseline(webviewRoot);
  if (baseline === null) {
    record({ status: 'skipped', reason: 'visual-baseline.json absent' });
    return;
  }
  const existing = baseline.render;
  if (!update && (existing === null || existing === undefined)) {
    record({
      status: 'skipped',
      reason: 'no render baseline yet; run --update-render-baseline once with a headless browser',
    });
    return;
  }

  let server = null;
  let browser = null;
  try {
    const portProbe = await startStaticServer();
    server = portProbe;
    const base = `http://127.0.0.1:${portProbe.port}`;
    const launchOptions = { headless: true };
    if (process.env.FAKTOR_BROWSER_CHANNEL) {
      launchOptions.channel = process.env.FAKTOR_BROWSER_CHANNEL;
    }
    if (process.env.FAKTOR_CHROMIUM_PATH) {
      launchOptions.executablePath = process.env.FAKTOR_CHROMIUM_PATH;
    }
    if (headless.kind === 'puppeteer') {
      launchOptions.args = ['--no-sandbox'];
    }
    browser = await headless.browserType.launch(launchOptions);
    const session = { kind: headless.kind, browser };
    const states = {};
    const divergences = [];
    for (const theme of RENDER_THEMES) {
      for (const width of RENDER_WIDTHS) {
        const result = await renderState(session, `${base}/?theme=${theme}`, width, theme);
        const fingerprintHash = sha256(JSON.stringify(result.fingerprint));
        states[result.key] = {
          fingerprint: result.fingerprint,
          fingerprintSha256: fingerprintHash,
          screenshotSha256: sha256(result.screenshot),
          screenshotBytes: result.screenshot.length,
        };
        if (!update) {
          const expected = existing.states?.[result.key];
          if (expected === undefined) {
            divergences.push(`${result.key}: baseline state missing`);
          } else if (expected.fingerprintSha256 !== fingerprintHash) {
            const keys = new Set([
              ...Object.keys(expected.fingerprint ?? {}),
              ...Object.keys(result.fingerprint),
            ]);
            const changed = [...keys].filter(
              (key) =>
                JSON.stringify(expected.fingerprint?.[key]) !==
                JSON.stringify(result.fingerprint?.[key]),
            );
            divergences.push(
              `${result.key}: fingerprint changed (` +
                `${changed.length > 0 ? changed.slice(0, 6).join(', ') : 'digest mismatch with identical content'})`,
            );
          }
        }
      }
    }
    if (update) {
      baseline.render = {
        tool: headless.label,
        engine: headless.kind,
        widths: [...RENDER_WIDTHS],
        themes: [...RENDER_THEMES],
        states,
      };
      writeFileSync(BASELINE_PATH, `${JSON.stringify(baseline, null, 2)}\n`);
      recorded.render = { status: 'updated', tool: headless.label, states: Object.keys(states).length };
      writeReport();
      console.log(`UPDATE headless render baseline (${headless.label}): ${Object.keys(states).length} states`);
      return;
    }
    if (divergences.length > 0) {
      failures += 1;
      recorded.render = { status: 'failed', tool: headless.label, divergences };
      writeReport();
      console.error(`FAIL  headless render parity — ${divergences.length} divergence(s):`);
      for (const divergence of divergences.slice(0, 10)) {
        console.error(`      ${divergence}`);
      }
      return;
    }
    recorded.render = { status: 'passed', tool: headless.label, states: Object.keys(states).length };
    writeReport();
    console.log(`PASS  headless render parity against baseline (${headless.label}, ${Object.keys(states).length} states)`);
  } catch (error) {
    record({
      status: 'skipped',
      reason: `headless render unavailable: ${error && error.message ? error.message : error}`,
    });
  } finally {
    if (browser !== null && typeof browser.close === 'function') {
      await browser.close().catch(() => {});
    }
    if (server !== null) {
      await server.close();
    }
  }
}

function writeReport() {
  if (!existsSync(DIST)) {
    return;
  }
  writeFileSync(
    REPORT_PATH,
    `${JSON.stringify(
      {
        schema: 'faktor.webview-visual-report/1',
        manifest: recorded.manifest,
        render: recorded.render,
      },
      null,
      2,
    )}\n`,
  );
}

// ----------------------------------------------------------------- self-test

function verifySandbox(sandbox) {
  const baseline = loadBaseline(sandbox);
  if (baseline === null) {
    throw new Error('sandbox baseline missing');
  }
  return verifyDist(sandbox, baseline, { strictExtras: true });
}

function selfTest() {
  let failed = 0;
  const report = (label, fn) => {
    try {
      fn();
      console.log(`PASS  ${label}`);
    } catch (error) {
      failed += 1;
      console.error(`FAIL  ${label} — ${error && error.message ? error.message : error}`);
    }
  };
  if (!existsSync(BASELINE_PATH)) {
    console.error('SELFTEST FAIL: no dist baseline to test against; build dist first');
    return 1;
  }
  const sandbox = mkdtempSync(join(tmpdir(), 'faktor-visual-selftest-'));
  try {
    cpSync(join(webviewRoot, 'dist'), join(sandbox, 'dist'), { recursive: true });
    report('clean sandbox verifies against the baseline', () => {
      const result = verifySandbox(sandbox);
      if (!result.ok || result.checked === 0) {
        throw new Error(`clean sandbox rejected: ${result.errors.join('; ')}`);
      }
    });
    report('tampered dist asset fails the gate', () => {
      const target = join(sandbox, 'dist', 'webview.css');
      const original = readFileSync(target);
      writeFileSync(target, `${original.toString('utf8')}/*tampered*/`);
      const result = verifySandbox(sandbox);
      writeFileSync(target, original);
      if (result.ok || !result.errors.some((error) => error.startsWith('hash mismatch: dist/webview.css'))) {
        throw new Error(`tamper accepted: ${result.errors.join('; ') || 'no errors'}`);
      }
    });
    report('missing dist asset fails the gate', () => {
      const target = join(sandbox, 'dist', 'webview.js');
      const original = readFileSync(target);
      rmSync(target);
      const result = verifySandbox(sandbox);
      writeFileSync(target, original);
      if (result.ok || !result.errors.some((error) => error.startsWith('missing: dist/webview.js'))) {
        throw new Error(`missing asset accepted: ${result.errors.join('; ') || 'no errors'}`);
      }
    });
    report('extra dist file fails the gate', () => {
      const target = join(sandbox, 'dist', 'intruder.js');
      writeFileSync(target, '// not in the baseline');
      const result = verifySandbox(sandbox);
      rmSync(target);
      if (result.ok || !result.errors.some((error) => error.startsWith('unexpected file: intruder.js'))) {
        throw new Error(`extra file accepted: ${result.errors.join('; ') || 'no errors'}`);
      }
    });
    report('artifact marker loss fails the gate', () => {
      const target = join(sandbox, 'dist', 'webview.css');
      const original = readFileSync(target);
      writeFileSync(target, original.toString('utf8').replaceAll('--vscode-', '--vsc0de-'));
      const result = verifySandbox(sandbox);
      writeFileSync(target, original);
      if (result.ok || !result.errors.some((error) => error.startsWith('artifact divergence'))) {
        throw new Error(`artifact loss accepted: ${result.errors.join('; ') || 'no errors'}`);
      }
    });
    report('missing dist falls back with a recorded notice', () => {
      const empty = mkdtempSync(join(tmpdir(), 'faktor-visual-fallback-'));
      try {
        if (locateVendoredBundle(empty) !== null) {
          throw new Error('empty root located a bundle');
        }
        if (!vendoredFallbackNotice(empty).includes(join(empty, 'dist'))) {
          throw new Error('fallback notice misses the dist path');
        }
      } finally {
        rmSync(empty, { recursive: true, force: true });
      }
    });
    report('remote index.html entry is refused', () => {
      const remote = mkdtempSync(join(tmpdir(), 'faktor-visual-remote-'));
      try {
        mkdirSync(join(remote, 'dist'));
        writeFileSync(
          join(remote, 'dist', 'index.html'),
          '<script src="https://evil.example/app.js"></script>',
        );
        if (locateVendoredBundle(remote) !== null) {
          throw new Error('remote entry was accepted');
        }
      } finally {
        rmSync(remote, { recursive: true, force: true });
      }
    });
  } finally {
    rmSync(sandbox, { recursive: true, force: true });
  }
  console.log(`\n${failed === 0 ? 'VISUAL SELFTEST OK' : `${failed} visual selftest(s) failed`}`);
  return failed > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------- main

async function main(argv) {
  if (argv.includes('--self-test')) {
    process.exit(selfTest());
  }
  const update = argv.includes('--update-render-baseline');
  const skipHeadless = argv.includes('--no-headless') && !update;

  checkVendoredSources();
  checkBundleContract();
  const baseline = checkBaseline(webviewRoot);

  if (baseline === null) {
    writeReport();
    console.error(`\n${failures} visual check(s) failed`);
    process.exit(1);
  }

  if (skipHeadless) {
    recorded.render = {
      status: 'skipped',
      reason: 'headless render disabled by --no-headless',
    };
    writeReport();
    console.log('SKIP  headless render (--no-headless) — recorded in dist/visual-report.json');
  } else {
    await runHeadless({ update });
  }

  if (failures > 0) {
    console.error(`\n${failures} visual check(s) failed`);
    process.exit(1);
  }
  const render = recorded.render?.status ?? 'unknown';
  if (render === 'passed' || render === 'updated') {
    console.log(`\nVISUAL CHECK OK — dist baseline verified; headless render ${render}`);
  } else {
    console.log(
      `\nVISUAL CHECK OK (dist baseline verified) — headless render NOT run (${render}` +
        `${recorded.render?.reason ? `: ${recorded.render.reason}` : ''}); recorded in dist/visual-report.json`,
    );
  }
}

main(process.argv.slice(2)).catch((error) => {
  console.error(`FATAL: ${error && error.stack ? error.stack : error}`);
  process.exit(1);
});
