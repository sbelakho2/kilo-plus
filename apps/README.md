# IDE clients (frozen)

- `vscode/` — frozen Kilo Code v7.5.6-derived webview shell (TypeScript/
  SolidJS). Never ported to Rust; the extension only launches the Faktor
  daemon (`faktor-plus serve --port 0`) and speaks the frozen protocol.
- `jetbrains/` — frozen JetBrains 7.1.2 Kotlin shell (split-mode shared/
  frontend/backend); the process manager is modified only to launch the
  Faktor binary.
