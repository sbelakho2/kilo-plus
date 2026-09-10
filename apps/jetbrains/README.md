# JetBrains split-mode bridge (Faktor)

The JetBrains side of the Faktor split-mode design. The daemon is the
`faktor-cli` binary launched as `serve --port 0`; this tree owns the
process lifecycle, the auth channel, the protocol clients, and a real
frontend panel. There is no placeholder code left in this tree.

The upstream JetBrains 7.1.2 Kotlin UI sources are not vendored (that
remains an external dependency), so the frontend here is a native Swing
panel that speaks Faktor Native Protocol v1 directly. The real
IntelliJ-platform tool-window adapter (`FaktorToolWindowFactory` +
`META-INF/plugin.xml`) embeds the same `FaktorChatPanel` without touching
the bridge.

## Modules

| Module | Contents |
| --- | --- |
| `:shared` | `dev.faktor.shared` — plain Kotlin data classes with zero dependencies. `Protocol.kt` is the frozen v7.5.6 wire contract (legacy migration glue); `NativeProtocol.kt` is the native surface: a JSON value model, a recursive-descent reader/writer, typed DTO parsers, and the strict request bodies. |
| `:backend` | `dev.faktor.backend` — `BackendProcessManager` (launch, startup line, bounded stdout drainer, SIGTERM-then-forcible stop), `NativeClient` (bearer-authenticated HTTP client of the native endpoints), `NativeEventStream` (SSE journal stream with cursor resume and bounded backoff). |
| `:frontend` | `dev.faktor.frontend` — `FaktorFrontendService` (the UI-free bridge: start/stop, session, task-run/agent/usage/verification/evidence routing, stream lifecycle), `FaktorChatPanel` (native Swing tool-window panel: chat input, streaming transcript, task/verification/budget status, agent controls, evidence retrieval) and `FaktorToolWindowFactory` (the IntelliJ tool-window host). `FaktorFrontendApp` launches the panel standalone. `src/main/resources/META-INF/plugin.xml` is the real plugin descriptor (`dev.faktor.jetbrains`, name/vendor `Faktor`, version `0.1.0`, `since-build 241`). |

## Authentication and lifecycle

- `BackendProcessManager` generates a 64-hex password with `SecureRandom`
  and passes it to the child only through `FAKTOR_SERVER_PASSWORD`
  (environment = protected channel; never argv, never disk, never logs).
- The frozen v7.5.6 startup line is the only stdout contract:
  `faktor server listening on http://127.0.0.1:<port>`.
- The native client authenticates every request with
  `Authorization: Bearer <password>`; the frozen v7.5.6 client keeps
  `Authorization: Basic base64("kilo:" + password)` for the compat
  surface (that literal is part of the frozen wire, not product
  branding).
- `stop()` is SIGTERM first, `destroyForcibly()` only after a 3s grace;
  the stdout drainer and the SSE thread stop with the process.

## Native protocol routing

| UI action | Endpoint |
| --- | --- |
| health / readiness | `GET /native/health`, `GET /native/ready` |
| new session | `POST /session/create`, `GET /session/list` |
| model catalog | `GET /models` |
| send / abort a turn | `POST /session/prompt`, `POST /native/session/{id}/abort` |
| status | `GET /session/{id}/projection`, `GET /native/session/{id}/tasks` |
| transcript | `GET /native/messages`, SSE `GET /api/session/{id}/events?events_after=` |
| journal paging | `GET /native/events` (cursor twin of the SSE stream) |
| task runs | `POST/GET /native/session/{id}/task-runs`, `POST .../{run_id}/cancel` |
| agents | `GET /native/agents`, `POST /native/agents/{id}/{pause,resume,cancel,retry,steer,model,budget}` |
| usage | `GET /native/usage`, `GET /native/session/{id}/usage` |
| verification | `GET /native/session/{id}/verification`, `GET /native/session/{id}/tasks/{task_id}/verification` |
| evidence | `GET /native/evidence/{id}`, `POST /native/evidence/{id}/retrieve` |

SSE frames carry `event:`, `id:` (journal sequence = resume cursor) and one
JSON `data:` line. Heartbeats are ignored but advance the cursor; oversized
frames are dropped loudly and skipped; reconnects resume from the last
delivered id with bounded exponential backoff.

## Verification

### IntelliJ plugin build (real, Gradle wrapper)

```bash
cd apps/jetbrains
./gradlew buildPlugin     # -> frontend/build/distributions/faktor-0.1.0.zip
./gradlew build           # all modules + test sources
```

The build resolves the Gradle 9.7.1 wrapper distribution from
`services.gradle.org`, the IntelliJ Platform Gradle plugin 2.18.1 and
Kotlin 2.4.20 from the Gradle Plugin Portal / Maven Central, and the
IntelliJ IDEA Community 2024.1.7 installer from the JetBrains CDN
(JDK 17 toolchain; `verifyPluginProjectConfiguration` currently reports
one warning: the plugin bundles the Kotlin 2.4 stdlib while the 2024.1
platform bundles 1.9.x). The project cache is redirected under `build/`
via `org.jetbrains.intellij.platform.intellijPlatformCache`, so no
generated state lands outside gitignored directories.

### kotlinc + daemon smoke (fallback, no Gradle)

```bash
bash apps/jetbrains/compile-and-smoke.sh
```

This builds `faktor-cli` if missing and then:

1. compiles `shared + backend + test + frontend` (Swing included) with
   `kotlinc` (kotlin-stdlib.jar from the compiler distribution, no
   network, no Gradle);
2. runs `BackendSmoke <binary>` — the frozen v7.5.6 wire flow (start →
   health → create session → send message → settle → messages → stop);
   a provider-less daemon answers the message send with HTTP 502 and the
   session lands `failed_recoverable`, both accepted as honest outcomes;
3. runs `NativeBridgeSmoke <binary>` — first the fake-server unit suite
   (`NativeClientTest`: JSON codec, every DTO parser incl. hostile
   payloads, exact method/path/bearer/body routing, typed error mapping,
   body bounds, SSE frames/heartbeat/oversized-frame/reconnect-resume),
   then the real native flow (start → bearer health → ready → create
   session → prompt → SSE frames + cursor → messages → journal page →
   task-runs → agents → usage → verification → task views → typed
   evidence error → stop).

Every step prints PASS/FAIL; the script exits nonzero on any failure. The
script compiles only the non-IntelliJ sources, so it stays runnable
without an SDK download.
