# Built webview dist and visual-parity gate

The frozen Kilo v7.5.6 UI source is vendored under
`ui/kilo-v756-webview/` (sparse checkout of
`packages/kilo-vscode/webview-ui` + `packages/kilo-ui`). That sparse tree has
no `package.json`, lockfile, build config or README, so the upstream build
cannot be driven from it directly:

```text
$ npm ci --prefix ui/kilo-v756-webview
npm error could not read package.json .../ui/kilo-v756-webview/package.json
$ npm install --prefix ui/kilo-v756-webview
npm error enoent Could not read package.json: ... ENOENT
```

The chat webview is instead built with the upstream's own toolchain from the
pinned monorepo commit, and only the resulting closure is placed in
`ui/kilo-v756-webview/dist/`.

## Provenance

| Field | Value |
|---|---|
| Repository | `https://github.com/Kilo-Org/kilocode` |
| Tag / commit | `v7.5.6` / `fa02955bfa17b60e57e0d7406d200a73337472ee` |
| Build script | upstream `packages/kilo-vscode/esbuild.js` (`--production`) |
| Toolchain | bun `1.3.14`, esbuild `0.27.x`, Node 26 |
| Command | `bun install --ignore-scripts && bun esbuild.js --production` |

`dist/build-manifest.json` records the full upstream dist listing (78 files,
140,463,231 bytes, sha256 + size each) and the 71-file vendored closure
(33,232,986 bytes): `webview.js`, `webview.css`, `shiki-worker.js`,
`markdown-shiki-worker.js`, the KaTeX/codicon fonts referenced by
`webview.css`, and `assets/icons/`. No upstream source is edited.

Reproduce with network + bun:

```bash
node ui/kilo-v756-webview/dist/toolchain/build-webview.mjs
```

Offline, vendor an existing upstream build (the copy used for this dist was
produced this way after a real upstream build):

```bash
node ui/kilo-v756-webview/dist/toolchain/build-webview.mjs \
  --from <upstream>/packages/kilo-vscode/dist \
  --icons <upstream>/packages/kilo-vscode/assets/icons
node ui/kilo-v756-webview/dist/toolchain/build-webview.mjs --verify
```

## Gate

```bash
node scripts/webview-visual-check.mjs                     # full gate
node scripts/webview-visual-check.mjs --no-headless       # record the skip
node scripts/webview-visual-check.mjs --update-render-baseline
node scripts/webview-visual-check.mjs --self-test         # adversarial
```

1. **Static checks** (unchanged first wave): vendored source presence,
   `ui/upstream.json` agreement, strict-local bundle discovery, nonce-only
   default-deny shell CSP, no remote code in `src/index.tsx`.
2. **Baseline manifest comparison**: every file in
   `dist/visual-baseline.json` must exist in `dist/` with the recorded sha256
   and size; required DOM/CSS artifact markers (`webviewReady`,
   `KILO_SHIKI_WORKER_URI`, `sendMessage`, `loadMessages`, `messagesLoaded`,
   `ICONS_BASE_URI`, `--vscode-`, `prompt-input`, `data-theme`) must survive
   bundling; and any extra file in `dist/` outside the manifest, the recorded
   meta files and `dist/toolchain/**` fails the gate. A tampered, missing or
   marker-stripped asset therefore fails; `--self-test` proves each of these
   paths against a sandbox copy and must print `VISUAL SELFTEST OK`.
3. **Headless render parity**: when playwright/puppeteer is importable (or
   `FAKTOR_PLAYWRIGHT` points at one), the built UI is served over a
   loopback-only static server (never `file://`) and rendered at widths
   `[390, 800, 1280] x themes [light, dark, high-contrast]`. A deterministic
   DOM/CSS fingerprint (root children, element count, class set, computed
   body/root styles, bounded page errors) is compared against
   `dist/visual-baseline.json` `render`. Screenshot hashes are recorded but
   not compared across machines (platform font rendering is not byte-stable).
   When no browser is available the render section is **skipped with a
   record** in `dist/visual-report.json` and the gate never reports it as
   passing; the final line distinguishes "headless render passed" from
   "dist baseline verified — headless render NOT run".

Headless baseline generation (explicit opt-in; the committed baseline `render`
section was generated with playwright 1.57 driving the locally installed
Google Chrome 153 through `channel: "chrome"`):

```bash
FAKTOR_PLAYWRIGHT=/path/to/playwright FAKTOR_BROWSER_CHANNEL=chrome \
  node scripts/webview-visual-check.mjs --update-render-baseline
# or, for a downloaded chromium: PLAYWRIGHT_BROWSERS_PATH=... (after
# `npx playwright install chromium`); FAKTOR_CHROMIUM_PATH pins an explicit
# executable instead.
```

A render result where `#root` never receives children is refused, not
recorded: an empty shell can never become a passing baseline. The committed
baseline records 750 mounted elements / 39 classes at each of the nine
width x theme states with zero page errors; changing any of them is a
divergence until the baseline is deliberately regenerated.

`dist/visual-baseline.json` is trusted gate input: regenerating it is a
reviewed diff, and the build script only ever writes `render: null` so a
rebuild cannot silently bless changed rendering.

## Extension wiring

`locateVendoredBundle(root)` (`apps/vscode/src/kilo-bridge.ts`) prefers a
built `dist/index.html` when present, extracting its script/style only if
every reference is local, inside `dist/` and exists; otherwise it uses
`dist/webview.js` + `dist/webview.css`. Workers (`shiki-worker.js`,
`markdown-shiki-worker.js`) and `dist/assets/icons` are attached when
present. `webview.ts` serves the strict nonce CSP shell with
`localResourceRoots` limited to the extension `media/` directory and the
bundle root; when no bundle is located it logs the recorded fallback notice
and serves the built-in panel unchanged.
