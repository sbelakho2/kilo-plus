// Where the frozen JetBrains 7.1.2 Kotlin frontend drops in.
//
// This placeholder documents the split-mode contract: the frontend renders
// session state exclusively from BackendProcessManager responses. The real
// 7.1.2 plugin sources replace this file's main() and keep using
// dev.kilop.backend.BackendProcessManager + dev.kilop.shared as-is.
package dev.kilop.frontend

import dev.kilop.backend.BackendProcessManager
import java.nio.file.Paths

fun main(args: Array<String>) {
    println("dev.kilop.frontend.PlaceholderFrontend")
    println()
    println("The frozen JetBrains 7.1.2 Kotlin frontend plugs in here.")
    println("Split-mode contract this scaffold provides:")
    println()
    println("  1. Launch:  BackendProcessManager(binary, dataDir).start()")
    println("     spawns `kilop-cli serve --port 0`, generates the 64-hex")
    println("     KILO_SERVER_PASSWORD, parses the frozen stdout line")
    println("     `kilo server listening on http://127.0.0.1:<port>` (5s cap)")
    println("     and returns BackendConnection(port, password, process).")
    println("  2. Auth:    every request carries")
    println("     Authorization: Basic base64(\"kilo:\" + password).")
    println("  3. Render:  session state comes ONLY from manager responses:")
    println("     - health()        GET /global/health")
    println("     - createSession() POST /session")
    println("     - sendMessage()   POST /session/{id}/message (turn, detached)")
    println("     - sessionState()  GET  /session/{id}")
    println("     - listMessages()  GET  /session/{id}/message?limit=5")
    println("     The state string (e.g. ready_for_next_turn, failed_recoverable,")
    println("     waiting for permission) drives the 7.1.2 chat rendering.")
    println("  4. Shutdown: stop(connection) — SIGTERM, destroyForcibly after")
    println("     a 3s grace; the stdout drainer stops with the process.")
    println()
    if (args.isNotEmpty()) {
        val manager = BackendProcessManager(
            Paths.get(args[0]),
            Paths.get(System.getProperty("java.io.tmpdir"), "kilop-frontend-demo")
        )
        val connection = manager.start()
        try {
            println("daemon up: ${connection.baseUrl} (pid ${connection.pid()})")
            println("health:   ${manager.health(connection)}")
        } finally {
            manager.stop(connection)
        }
    } else {
        println("(run with a kilop-cli binary path to demo the daemon lifecycle)")
    }
}
