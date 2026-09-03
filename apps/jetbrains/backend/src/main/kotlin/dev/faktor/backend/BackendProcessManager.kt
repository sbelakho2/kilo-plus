// The Kilo+ backend process manager: the JetBrains-side mirror of the Kilo
// backend-management pattern. It launches the Kilo+ binary, parses the frozen
// startup line from stdout, authenticates with the frontend-generated
// password, and speaks the v7.5.6 wire surface over java.net.http.
//
// Rules observed:
//  - stdout is drained on a dedicated thread so a full pipe never deadlocks
//    the manager; the ring buffer is bounded (1024 lines).
//  - stop() is SIGTERM first, destroyForcibly only after a grace period;
//    the drain thread stops with the process (no orphans, no leak).
//  - a crashed/missing daemon fails loudly: start() throws with the exit
//    code and a bounded stdout tail.
package dev.faktor.backend

import dev.faktor.shared.BasicAuth
import dev.faktor.shared.HealthResult
import dev.faktor.shared.MessageModel
import dev.faktor.shared.MessageSendRequest
import dev.faktor.shared.Part
import dev.faktor.shared.SessionCreateRequest
import dev.faktor.shared.StartupLine
import dev.faktor.shared.parseHealth
import dev.faktor.shared.parseMessageCount
import dev.faktor.shared.parseMessageId
import dev.faktor.shared.parseSessionId
import dev.faktor.shared.parseSessionState
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.security.SecureRandom
import java.time.Duration
import java.util.ArrayDeque
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** A daemon-side failure. [status] carries the HTTP status when one applies. */
class BackendException(
    message: String,
    val status: Int? = null,
    cause: Throwable? = null
) : Exception(message, cause)

private const val MAX_BUFFERED_LINES = 1024

/** A running daemon plus everything needed to talk to and stop it. */
class BackendConnection(
    val port: Int,
    val password: String,
    val process: Process,
    internal val sink: StdoutSink
) {
    internal fun recentLines(max: Int): List<String> = sink.recentLines(max)
    val baseUrl: String = "http://127.0.0.1:$port"

    fun isAlive(): Boolean = process.isAlive

    fun pid(): Long = process.pid()
}

/**
 * Manages the Kilo+ daemon lifecycle. The binary must be the faktor-cli
 * executable; it is launched as `serve --port 0 --data-dir <dir>` with the
 * generated password in `FAKTOR_SERVER_PASSWORD`.
 */
class BackendProcessManager(private val binaryPath: Path, private val dataDir: Path) {

    companion object {
        private const val STARTUP_TIMEOUT_MS = 5_000L
        private const val STOP_GRACE_MS = 3_000L
        private const val HTTP_TIMEOUT_SECONDS = 5L
        private const val PASSWORD_HEX_BYTES = 32
        private const val TERMINAL_STATE_POLL_MS = 200L
        private const val TERMINAL_STATE_DEADLINE_MS = 20_000L
    }

    private val rng = SecureRandom()

    private val http: HttpClient = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(HTTP_TIMEOUT_SECONDS))
        .build()

    /** Starts the daemon and waits (bounded) for the frozen startup line. */
    fun start(): BackendConnection {
        if (!Files.isRegularFile(binaryPath) || !Files.isExecutable(binaryPath)) {
            throw BackendException("kilo+ binary not found or not executable: $binaryPath")
        }
        val password = generatePassword()
        val pb = ProcessBuilder(
            binaryPath.toAbsolutePath().toString(),
            "serve", "--port", "0",
            "--data-dir", dataDir.toAbsolutePath().toString()
        )
        pb.environment()["FAKTOR_SERVER_PASSWORD"] = password
        // stdout is the startup-line contract and is drained by StdoutSink;
        // daemon logs ride stderr to the IDE console.
        pb.redirectError(ProcessBuilder.Redirect.INHERIT)
        val process = try {
            pb.start()
        } catch (e: IOException) {
            throw BackendException("failed to launch kilo+ binary: ${e.message}", cause = e)
        }
        val sink = StdoutSink(process)
        val port = sink.awaitStartupLine(STARTUP_TIMEOUT_MS)
        if (port == null) {
            val exit = try {
                process.exitValue()
            } catch (e: IllegalThreadStateException) {
                null
            }
            val tail = sink.recentLines(20)
            process.destroyForcibly()
            sink.stop()
            val reason = if (exit != null) {
                "daemon exited with code $exit before the startup line"
            } else {
                "startup line not seen within ${STARTUP_TIMEOUT_MS}ms"
            }
            throw BackendException(
                "$reason; stdout tail: ${tail.joinToString(" | ")}"
            )
        }
        return BackendConnection(port, password, process, sink)
    }

    /** SIGTERM, then destroyForcibly after [STOP_GRACE_MS]; stops the drainer. */
    fun stop(connection: BackendConnection) {
        val process = connection.process
        process.destroy()
        if (!process.waitFor(STOP_GRACE_MS, TimeUnit.MILLISECONDS)) {
            process.destroyForcibly()
            process.waitFor(1, TimeUnit.SECONDS)
        }
        connection.sink.stop()
    }

    /** `GET /global/health` (auth required). Throws on 401 / IO / bad body. */
    fun health(connection: BackendConnection): HealthResult {
        val resp = request(connection, "GET", "/global/health", null)
        return parseHealth(resp.body)
    }

    /** `POST /session` with the wire shape; returns the `sessionID`. */
    fun createSession(connection: BackendConnection, title: String): String {
        val body = SessionCreateRequest(title = title).toJson()
        val resp = request(connection, "POST", "/session", body)
        return parseSessionId(resp.body)
    }

    /**
     * `POST /session/{sessionID}/message` with parts=[{type:text,...}];
     * returns the `messageID`. The daemon runs the turn detached; the state
     * settles asynchronously (poll via [sessionState]).
     */
    fun sendMessage(connection: BackendConnection, sessionId: String, text: String): String {
        val body = MessageSendRequest(
            model = MessageModel(providerID = "default", modelID = "default"),
            parts = listOf(Part.TextPart(text))
        ).toJson()
        val resp = request(connection, "POST", "/session/$sessionId/message", body)
        return parseMessageId(resp.body)
    }

    /** `GET /session/{sessionID}/message?limit=5` → number of messages in the page. */
    fun listMessages(connection: BackendConnection, sessionId: String): Int {
        val resp = request(connection, "GET", "/session/$sessionId/message?limit=5", null)
        return parseMessageCount(resp.body)
    }

    /** `GET /session/{sessionID}` → the wire `state` string (e.g. `ready_for_next_turn`). */
    fun sessionState(connection: BackendConnection, sessionId: String): String {
        val resp = request(connection, "GET", "/session/$sessionId", null)
        return parseSessionState(resp.body)
    }

    /**
     * Polls the session state until it lands on `ready_for_next_turn` or
     * `failed_recoverable` (the two outcomes the smoke accepts: a provider
     * gap leaves the session failed_recoverable, never stuck mid-turn).
     * Throws [BackendException] when the deadline expires.
     */
    fun awaitSettledState(connection: BackendConnection, sessionId: String): String {
        val deadline = System.currentTimeMillis() + TERMINAL_STATE_DEADLINE_MS
        var last: String? = null
        while (true) {
            last = try {
                sessionState(connection, sessionId)
            } catch (e: BackendException) {
                last
            }
            if (last == "ready_for_next_turn" || last == "failed_recoverable") {
                return last!!
            }
            if (System.currentTimeMillis() >= deadline) {
                throw BackendException(
                    "session $sessionId did not settle within ${TERMINAL_STATE_DEADLINE_MS}ms; " +
                        "last state=${last ?: "unknown"}"
                )
            }
            Thread.sleep(TERMINAL_STATE_POLL_MS)
        }
    }

    private data class HttpResponseLite(val status: Int, val body: String)

    private fun request(
        connection: BackendConnection,
        method: String,
        path: String,
        body: String?
    ): HttpResponseLite {
        val builder = HttpRequest.newBuilder(URI.create(connection.baseUrl + path))
            .timeout(Duration.ofSeconds(HTTP_TIMEOUT_SECONDS))
            .header(BasicAuth.HEADER_NAME, BasicAuth(connection.password).headerValue)
            .header("Accept", "application/json")
        if (body != null) {
            builder.header("Content-Type", "application/json")
            builder.method(method, HttpRequest.BodyPublishers.ofString(body))
        } else {
            builder.method(method, HttpRequest.BodyPublishers.noBody())
        }
        val resp = try {
            http.send(builder.build(), HttpResponse.BodyHandlers.ofString())
        } catch (e: IOException) {
            throw BackendException(
                "request $method $path failed (daemon crashed?): ${e.message}",
                cause = e
            )
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
            throw BackendException("request $method $path interrupted", cause = e)
        }
        if (resp.statusCode() == 401) {
            throw BackendException("$method $path: unauthorized (401)", status = 401)
        }
        if (resp.statusCode() !in 200..299) {
            val bodyText = resp.body().take(500)
            throw BackendException(
                "$method $path: HTTP ${resp.statusCode()}: $bodyText",
                status = resp.statusCode()
            )
        }
        return HttpResponseLite(resp.statusCode(), resp.body())
    }

    private fun generatePassword(): String {
        val bytes = ByteArray(PASSWORD_HEX_BYTES)
        rng.nextBytes(bytes)
        val sb = StringBuilder(bytes.size * 2)
        for (b in bytes) sb.append(String.format("%02x", b))
        return sb.toString()
    }
}

/**
 * Drains the daemon stdout on a dedicated daemon thread into a bounded ring
 * buffer, and latches the startup line when it appears. EOF or process exit
 * without the startup line releases the latch so start() fails fast.
 */
class StdoutSink(process: Process) {
    private val lock = Object()
    private val ring = ArrayDeque<String>()
    private val startup = CountDownLatch(1)

    @Volatile
    private var startupPort: Int? = null

    @Volatile
    private var stopped = false

    private val thread: Thread = Thread({ drain(process) }, "faktor-stdout-drain")

    init {
        thread.isDaemon = true
        // Wake the latch if the process exits without printing the line.
        process.onExit().whenComplete { _, _ ->
            if (startupPort == null) startup.countDown()
        }
        thread.start()
    }

    /** Waits (bounded) for the startup line; null on timeout or early exit. */
    fun awaitStartupLine(timeoutMs: Long): Int? {
        val deadline = System.currentTimeMillis() + timeoutMs
        if (!startup.await(timeoutMs, TimeUnit.MILLISECONDS)) return null
        // The latch may have been tripped by onExit a micro-instant before
        // the drainer parsed the line; give it a brief chance.
        while (startupPort == null && System.currentTimeMillis() < deadline) {
            Thread.sleep(10)
        }
        return startupPort
    }

    fun recentLines(max: Int): List<String> = synchronized(lock) {
        ring.toList().takeLast(max)
    }

    fun stop() {
        stopped = true
        thread.interrupt()
        try {
            thread.join(1000)
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }

    private fun drain(process: Process) {
        val reader = BufferedReader(InputStreamReader(process.inputStream, Charsets.UTF_8))
        try {
            while (!stopped) {
                val line = reader.readLine() ?: break
                val parsed = StartupLine.parse(line)
                if (parsed != null && startupPort == null) {
                    startupPort = parsed.port
                    startup.countDown()
                }
                synchronized(lock) {
                    ring.addLast(line)
                    while (ring.size > MAX_BUFFERED_LINES) {
                        ring.removeFirst()
                    }
                }
            }
        } catch (e: IOException) {
            // Stream closed under us (e.g. destroyForcibly during stop()).
        }
        if (startupPort == null) startup.countDown()
    }
}
