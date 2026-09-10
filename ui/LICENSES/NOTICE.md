# Vendored upstream UI notices

Source: <https://github.com/Kilo-Org/kilocode>

| Field | Value |
|---|---|
| Tag | `v7.5.6` |
| Commit | `fa02955bfa17b60e57e0d7406d200a73337472ee` |
| `packages/kilo-vscode/webview-ui` | vendored verbatim as `ui/kilo-v756-webview/` |
| `packages/kilo-ui` | vendored verbatim as `ui/kilo-ui/` |

These trees are frozen byte-for-byte. `ui/upstream.json` records the SHA-256
of every vendored file; `scripts/verify-upstream.mjs` recomputes them and
fails on any missing, modified, or unexpected file. Never edit a file under
`ui/kilo-v756-webview/` or `ui/kilo-ui/` in place — change the pin and
re-vendor instead.

## Licenses

The upstream repository and both vendored paths are MIT licensed
(`Copyright (c) 2026 Kilo Code`, `Copyright (c) 2025 opencode`):

- `kilocode-LICENSE.txt` — upstream repository `LICENSE` (covers `kilo-ui`).
- `kilo-vscode-LICENSE.txt` — `packages/kilo-vscode/LICENSE` (covers
  `webview-ui`).
- `kilo-vscode-THIRD_PARTY_LICENSES/` — `packages/kilo-vscode/THIRD_PARTY_LICENSES`.

The MIT permission notice and copyright lines must remain with every copy or
substantial portion of this UI.

## Fetch protocol

`scripts/vendor-upstream.sh` (network required) performs exactly:

```
git clone --filter=blob:none --no-checkout --depth 1 --branch v7.5.6 \
    https://github.com/Kilo-Org/kilocode <tmp>/repo
git -C <tmp>/repo sparse-checkout set packages/kilo-vscode/webview-ui packages/kilo-ui
git -C <tmp>/repo checkout fa02955bfa17b60e57e0d7406d200a73337472ee
rsync -a --delete <tmp>/repo/packages/kilo-vscode/webview-ui/ ui/kilo-v756-webview/
rsync -a --delete <tmp>/repo/packages/kilo-ui/ ui/kilo-ui/
node scripts/verify-upstream.mjs --write
node scripts/verify-upstream.mjs --verify
```

When the network is blocked the script writes nothing, prints this protocol,
and exits nonzero. Vendored content is never fabricated: a blocked run leaves
`ui/` exactly as it was.

## Build output

The upstream webview ships source only; `dist/webview.js` and
`dist/webview.css` are esbuild output and are deliberately not committed.
`dist/` directories are excluded from the manifest and from unexpected-file
checks, so a locally produced bundle can live at
`ui/kilo-v756-webview/dist/` without invalidating verification. The VS Code
extension loads that bundle when present and falls back to its built-in chat
panel when absent (see `docs/upstream-ui.md`).
