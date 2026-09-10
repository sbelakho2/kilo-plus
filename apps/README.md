# IDE clients (frozen)

- `vscode/` — Faktor extension shell over the frozen Kilo Code v7.5.6
  webview. The upstream trees are vendored under `ui/` (pinned commit,
  SHA-256 manifest); `src/kilo-bridge.ts` translates native state onto the
  frozen message ABI, and `src/webview.ts` serves the vendored bundle when
  built, else the built-in chat panel. The extension only launches the
  Faktor daemon (`faktor-cli serve --port 0`) and speaks the native
  protocol.
- `jetbrains/` — frozen JetBrains 7.1.2 Kotlin shell (split-mode shared/
  frontend/backend); the process manager is modified only to launch the
  Faktor binary.
