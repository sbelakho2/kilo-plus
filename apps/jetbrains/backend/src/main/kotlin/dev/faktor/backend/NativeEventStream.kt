// SSE client for the daemon's journal event stream
// (`GET /api/session/{id}/events?events_after=<cursor>`), the stream the
// native server projects from the durable journal. Mirrors the VS Code
// eventStream.ts semantics:
//
//  - every frame carries `event:` (projection type), `id:` (the journal
//    sequence — the resume cursor) and one JSON `data:` line;
//  - heartbeats (`event: heartbeat` / comment keep-alives) are ignored but
//    still advance the cursor when they carry an id;
//  - on any disconnect the loop reconnects with bounded exponential backoff
//    and resumes from the last delivered frame id, so a reconnect can
//    neither duplicate nor skip events;
//  - frames larger than the bound are dropped LOUDLY (onError) and the
//    cursor advances past them when the frame carried an id, so a hostile
//    frame can never livelock the stream;
//  - the stream runs on one daemon thread; stop() closes the body, which
//    unblocks the reader (no orphan thread, no orphan socket).
package dev.faktor.backend

import dev.faktor.shared.JsonCodec
import dev.faktor.shared.JsonValue
import dev.faktor.shared.NativeProtocolException
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.net.URI
import java.net.URLEncoder
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.util.Random

/** One journal frame: [id] is the resume cursor, [event] the projection type. */
data class NativeSseEvent(val id: Long, val event: String, val data: JsonValue)

/**
 * One reconnecting SSE client for a single session. Listeners are invoked on
 * the stream thread; UI callers must marshal to their own event loop.
 */
class NativeEventStream(
    private val baseUrl: String,
    private val bearerToken: String,
    private val sessionId: String,
    cursor: Long = 0,
    private val maxFrameBytes: Int = DEFAULT_MAX_FRAME_BYTES,
    private val minBackoffMs: Long = 250L,
    private val maxBackoffMs: Long = 15_000L,
    private val timeoutMs: Long = DEFAULT_TIMEOUT_MS,
    private val onEvent: (NativeSseEvent) -> Unit,
    private val onStatus: (String, String?) -> Unit = { _, _ -> },
    private val onError: (Exception) -> Unit = {}
) {
    companion object {
        const val DEFAULT_MAX_FRAME_BYTES = 1 shl 20
        const val DEFAULT_TIMEOUT_MS = 30_000L
        private const val MAX_BACKOFF_ATTEMPT = 20
        private const val MAX_ERROR_BODY_CHARS = 200

        fun forConnection(
            connection: BackendConnection,
            sessionId: String,
            cursor: Long = 0,
            onEvent: (NativeSseEvent) -> Unit,
            onStatus: (String, String?) -> Unit = { _, _ -> },
            onError: (Exception) -> Unit = {}
        ): NativeEventStream = NativeEventStream(
            connection.baseUrl, connection.password, sessionId, cursor,
            onEvent = onEvent, onStatus = onStatus, onError = onError
        )
    }

    @Volatile
    private var cursorValue: Long = if (cursor < 0) 0 else cursor

    @Volatile
    private var stopped = true

    @Volatile
    private var stage: String = "stopped"

    @Volatile
    private var activeBody: InputStream? = null

    private var thread: Thread? = null

    private val random = Random()

    val cursor: Long
        get() = cursorValue

    val status: String
        get() = stage

    /** Seed the resume cursor (e.g. from a paged `/native/events` read). */
    fun setCursor(value: Long) {
        if (value >= 0) cursorValue = value
    }

    /** Starts streaming; idempotent while running. */
    fun start() {
        if (!stopped) return
        stopped = false
        val t = Thread({ loop() }, "faktor-sse-$sessionId")
        t.isDaemon = true
        thread = t
        t.start()
    }

    /** Stops streaming and unblocks the reader; idempotent. */
    fun stop() {
        if (stopped) return
        stopped = true
        closeQuietly()
        thread?.interrupt()
        try {
            thread?.join(1000)
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        thread = null
        setStage("stopped", null)
    }

    private fun loop() {
        val http = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(timeoutMs))
            .build()
        var attempt = 0
        while (!stopped) {
            if (attempt > 0) {
                val shift = if (attempt - 1 > 16) 16 else attempt - 1
                val backoff = minOf(maxBackoffMs, minBackoffMs shl shift) + random.nextInt(100)
                setStage("retrying", "reconnect in ${backoff}ms from cursor $cursorValue")
                try {
                    Thread.sleep(backoff)
                } catch (e: InterruptedException) {
                    Thread.currentThread().interrupt()
                    break
                }
                if (stopped) break
            }
            setStage(if (attempt == 0) "connecting" else "retrying", "from cursor $cursorValue")
            try {
                connectOnce(http)
                attempt = 1
                setStage("retrying", "stream ended at cursor $cursorValue")
            } catch (e: Exception) {
                if (stopped) break
                attempt = minOf(attempt + 1, MAX_BACKOFF_ATTEMPT)
                onError(wrap(e))
            }
        }
        setStage("stopped", null)
    }

    /** Connects and pumps frames until the stream ends. */
    private fun connectOnce(http: HttpClient) {
        val url = baseUrl.trimEnd('/') +
            "/api/session/" + URLEncoder.encode(sessionId, "UTF-8") +
            "/events?events_after=" + cursorValue
        val request = HttpRequest.newBuilder(URI.create(url))
            .timeout(Duration.ofMillis(timeoutMs))
            .header("Authorization", "Bearer $bearerToken")
            .header("Accept", "text/event-stream")
            .header("Last-Event-ID", cursorValue.toString())
            .header("Cache-Control", "no-cache")
            .GET()
            .build()
        val response = http.send(request, HttpResponse.BodyHandlers.ofInputStream())
        if (response.statusCode() !in 200..299) {
            val detail = response.body().use { readErrorBody(it) }
            throw NativeProtocolException(
                "GET /api/session/{id}/events",
                "stream rejected with HTTP ${response.statusCode()}$detail"
            )
        }
        val body = response.body()
        activeBody = body
        setStage("open", "cursor $cursorValue")
        try {
            pump(body)
        } finally {
            closeQuietly()
        }
    }

    /**
     * Line-framed SSE pump with a hard per-line bound: a hostile endless
     * line can never grow RAM, it only marks the frame dropped (the frame is
     * skipped loudly, cursor still advances when an id was seen).
     */
    private fun pump(body: InputStream) {
        val reader = InputStreamReader(body, Charsets.UTF_8)
        val line = StringBuilder()
        var lineOverlong = false
        var eventName: String? = null
        var frameId: Long? = null
        val data = StringBuilder()
        var dropped = false
        while (!stopped) {
            val c = reader.read()
            if (c < 0) return
            if (c == '\r'.code) continue
            if (c != '\n'.code) {
                if (line.length < maxFrameBytes) {
                    line.append(c.toChar())
                } else {
                    lineOverlong = true
                    dropped = true
                }
                continue
            }
            val text = if (lineOverlong) "" else line.toString()
            line.setLength(0)
            val overlong = lineOverlong
            lineOverlong = false
            if (text.isEmpty() && !overlong) {
                dispatch(eventName, frameId, data.toString(), dropped)
                eventName = null
                frameId = null
                data.setLength(0)
                dropped = false
                continue
            }
            if (overlong) {
                // Unknown overlong line: ignore its content, keep the frame
                // flagged dropped; the blank line still dispatches.
                continue
            }
            if (text.startsWith(":")) continue
            when {
                text.startsWith("event:") -> eventName = text.substring(6).trim()
                text.startsWith("id:") -> {
                    val parsed = text.substring(3).trim().toLongOrNull()
                    if (parsed != null && parsed >= 0) frameId = parsed
                }
                text.startsWith("data:") -> {
                    val chunk = text.substring(5)
                        .let { if (it.startsWith(" ")) it.substring(1) else it }
                    if (data.isNotEmpty()) data.append('\n')
                    data.append(chunk)
                    if (data.length > maxFrameBytes) dropped = true
                }
            }
        }
    }

    private fun dispatch(eventName: String?, frameId: Long?, raw: String, dropped: Boolean) {
        if (dropped) {
            onError(
                NativeProtocolException(
                    "sse frame ${frameId ?: "?"}",
                    "frame exceeded the $maxFrameBytes byte bound; skipped"
                )
            )
            if (frameId != null) cursorValue = maxOf(cursorValue, frameId)
            return
        }
        if (raw.isEmpty()) return
        val parsed = try {
            JsonCodec.parse(raw)
        } catch (e: NativeProtocolException) {
            onError(NativeProtocolException("sse frame ${frameId ?: "?"}", "data is not valid JSON"))
            return
        }
        val obj = parsed as? JsonValue.Obj
        val declared = (obj?.fields?.get("event") as? JsonValue.Str)?.value
        if (declared != null && eventName != null && eventName != declared) {
            onError(
                NativeProtocolException(
                    "sse frame ${frameId ?: "?"}",
                    "event field $eventName disagrees with data discriminator $declared"
                )
            )
            return
        }
        val tagged = declared ?: eventName
        if (tagged == null) {
            onError(
                NativeProtocolException("sse frame ${frameId ?: "?"}", "no event discriminator")
            )
            return
        }
        if (tagged == "heartbeat") {
            if (frameId != null) cursorValue = maxOf(cursorValue, frameId)
            return
        }
        if (frameId == null) {
            onError(
                NativeProtocolException("sse frame ($tagged)", "carries no id cursor; skipped")
            )
            return
        }
        if (frameId <= cursorValue) return
        cursorValue = frameId
        onEvent(NativeSseEvent(frameId, tagged, parsed))
    }

    private fun readErrorBody(stream: InputStream): String {
        val out = StringBuilder()
        val reader = InputStreamReader(stream, Charsets.UTF_8)
        try {
            while (out.length < MAX_ERROR_BODY_CHARS) {
                val c = reader.read()
                if (c < 0) break
                out.append(c.toChar())
            }
        } catch (e: IOException) {
            // Best effort only.
        }
        val snippet = out.toString().trim()
        return if (snippet.isEmpty()) "" else ": ${snippet.take(MAX_ERROR_BODY_CHARS)}"
    }

    private fun closeQuietly() {
        val body = activeBody
        activeBody = null
        if (body != null) {
            try {
                body.close()
            } catch (e: IOException) {
                // Already closed.
            }
        }
    }

    private fun setStage(value: String, detail: String?) {
        stage = value
        onStatus(value, detail)
    }

    private fun wrap(e: Exception): Exception =
        if (e is NativeProtocolException) e
        else NativeProtocolException("GET /api/session/{id}/events", e.message ?: e.javaClass.simpleName)
}
