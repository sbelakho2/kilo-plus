# Vendored upstream UI (frozen Kilo v7.5.6)

The frozen Kilo Code v7.5.6 UI is vendored verbatim under `ui/` so the VS
Code extension can serve the real webview instead of approximating it.

## Pin and provenance

| Field | Value |
|---|---|
| Repository | `https://github.com/Kilo-Org/kilocode` |
| Tag | `v7.5.6` |
| Commit | `fa02955bfa17b60e57e0d7406d200a73337472ee` |
| `packages/kilo-vscode/webview-ui` | `ui/kilo-v756-webview/` |
| `packages/kilo-ui` | `ui/kilo-ui/` |
| Hash algorithm | SHA-256 over every vendored file (`ui/upstream.json`) |

`ui/LICENSES/` carries the upstream MIT license, the `packages/kilo-vscode`
license and its third-party notices; see `ui/LICENSES/NOTICE.md` for the
provenance and the exact fetch protocol.

## Verification (offline)

```bash
node scripts/verify-upstream.mjs --verify     # 774 files, sha256, extra-file strict
node scripts/verify-upstream.mjs --self-test  # tampered copy must fail
scripts/vendor-upstream.sh --check            # same verify through the fetch script
```

The verifier fails on a missing, modified, or unexpected file inside either
vendored tree. Generated `dist/` build output is the one exception, because it
is not vendored source.

## Re-vendoring (network)

```bash
scripts/vendor-upstream.sh
```

The script clones the pinned tag with `--filter=blob:none --no-checkout`,
sparse-checks out exactly the two upstream paths, refuses to proceed if the
checked-out HEAD differs from the pin, copies them verbatim, regenerates the
manifest, and verifies it. If the network is blocked it writes nothing,
prints the fetch protocol, and exits nonzero — vendored content is never
fabricated.

## Extension wiring

- `apps/vscode/src/kilo-bridge.ts` — the message-ABI bridge. Inbound messages
  from the frozen UI are validated against the command set Faktor honors;
  unknown kinds, malformed envelopes, oversized payloads and unsupported
  structured fields (attachments, review comments, agent-manager context)
  are dropped with an explicit reason and logged. Outbound, native snapshots
  are translated into the upstream messages the webview consumes (`ready`,
  `connectionState`, `sessionsLoaded`, `sessionStatus`, `messagesLoaded`,
  `todoUpdated`, `error`), bounded by entry count and serialized bytes.
- `apps/vscode/src/webview.ts` — serves the vendored shell with a
  default-deny, nonce-only CSP when a bundle is present under
  `ui/kilo-v756-webview/dist/` (a built `dist/index.html` entry when present,
  else the upstream esbuild pair `dist/webview.js` + `dist/webview.css`; or
  `FAKTOR_UI_BUNDLE` points at a bundle directory). Without a bundle it
  records the fallback notice from `vendoredFallbackNotice` and serves the
  built-in Faktor chat panel exactly as before.
- Bundle discovery lives in `apps/vscode/src/kilo-bridge.ts`
  (`locateVendoredBundle`): entry paths from a built `index.html` are accepted
  only when strictly local and inside `dist/`; remote origins, traversal
  segments and missing referenced assets refuse the whole bundle.
- Build output and the visual-parity gate are documented in
  `docs/webview-dist-and-visual-gate.md`.
- `scripts/webview-visual-check.mjs` — offline static/visual check of the
  shell and tree contract (the screenshot suite in `tests/visual` is a Rust
  crate and is not modified).

## Building the bundle

The upstream repository ships source only; `dist/webview.{js,css}` are esbuild
output. The vendored source is byte-for-byte upstream and must never be
edited. To produce a bundle for the extension, build the upstream workspace
at the pinned commit (upstream's `packages/kilo-vscode/esbuild.js`) and copy
its `dist/webview.js`, `dist/webview.css`, `dist/shiki-worker.js`,
`dist/markdown-shiki-worker.js` and `assets/icons` into
`ui/kilo-v756-webview/dist/` and `ui/kilo-v756-webview/assets/` — `dist/` is
excluded from manifest verification. Until such a bundle exists the
extension uses the built-in panel; that is the current commit state.

## Known residuals

- No prebuilt bundle is committed (the upstream bundle cannot be built here
  without the full upstream workspace/registry), so the extension runs the
  fallback panel until `dist/` is produced.
- The native session listing has no timestamps, so `SessionInfo.createdAt` /
  `updatedAt` are the deterministic epoch, never a fabricated "now".
- Only the message subset above is bridged; upstream features with no native
  equivalent (marketplace, profile, cloud sessions, agent-manager worktrees)
  are not wired.
- Attachments, review comments and agent-manager context are refused loudly
  rather than silently dropped.
