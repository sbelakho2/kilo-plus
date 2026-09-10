// Adversarial tests for the native-protocol bridge:
//
//   * JSON codec: round trips, hostile input, bounded behavior.
//   * DTO parsers: valid fixtures parse; missing fields / wrong types /
//     trailing junk are loud NativeProtocolExceptions.
//   * NativeClient routing: a fake raw-socket HTTP daemon asserts the exact
//     method+path+Bearer header+body for every endpoint the panel calls,
//     maps 401/500 bodies to typed errors, and enforces the body bound.
//   * NativeEventStream: fake SSE daemon tests frames, heartbeat handling,
//     oversized-frame dropping, reconnect with cursor resume (the second
//     request must carry events_after=<last delivered id>).
//   * NativeBridgeSmoke: a real `faktor-cli` daemon end-to-end (start,
//     authenticate, create session, prompt, SSE, task-run/agent/usage/
//     verification/evidence routes), mirroring BackendSmoke.
//
// No kotlin.test/JUnit and no network beyond 127.0.0.1: the raw-socket fake
// daemon keeps the whole suite dependency-free and offline.
package dev.faktor.backend

import dev.faktor.shared.JsonCodec
import dev.faktor.shared.JsonValue
import dev.faktor.shared.NativeApiException
import dev.faktor.shared.NativeProtocolException
import dev.faktor.shared.NativeRequests
import dev.faktor.shared.parseNativeAgentControlAck
import dev.faktor.shared.parseNativeAgents
import dev.faktor.shared.parseNativeEvidence
import dev.faktor.shared.parseNativeEvidenceRetrieval
import dev.faktor.shared.parseNativeHealth
import dev.faktor.shared.parseNativeModelCatalog
import dev.faktor.shared.parseNativeProjection
import dev.faktor.shared.parseNativePromptReceipt
import dev.faktor.shared.parseNativeReady
import dev.faktor.shared.parseNativeSessionCreated
import dev.faktor.shared.parseNativeSessionList
import dev.faktor.shared.parseNativeSessionUsage
import dev.faktor.shared.parseNativeTaskRunCancelled
import dev.faktor.shared.parseNativeTaskRunStarted
import dev.faktor.shared.parseNativeTaskRuns
import dev.faktor.shared.parseNativeTaskVerification
import dev.faktor.shared.parseNativeTaskViews
import dev.faktor.shared.parseNativeUsage
import dev.faktor.shared.parseNativeVerificationView
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.nio.file.Files
import java.nio.file.Paths
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

object NativeClientTest {

    @JvmStatic
    fun runAll() {
        assertJsonCodec()
        assertRequestBodies()
        assertResponseParsers()
        assertHostileParsers()
        assertClientRoutes()
        assertErrorMapping()
        assertBodyBound()
        assertSseStreaming()
        assertSseOversizedFrame()
        println("PASS all native client unit assertions")
    }
}

// ----------------------------------------------------------------- fixtures

private const val HEALTH_JSON = "{\"ok\":true,\"version\":\"1.2.3\"}"

private const val READY_JSON = "{\"ready\":true}"

private const val CREATED_JSON =
    "{\"id\":\"7\",\"title\":\"T\",\"created_ms\":1750000000000}"

private const val SESSIONS_JSON =
    "{\"sessions\":[{\"id\":\"7\",\"title\":\"T\",\"provider\":\"fake\"," +
        "\"model\":\"m\",\"state\":\"ready\"}]}"

private const val MODELS_JSON =
    "[{\"provider\":\"fake\",\"model\":\"m\",\"context\":1000,\"maxOutput\":100," +
        "\"tools\":true,\"parallelTools\":false,\"reasoning\":false,\"thinking\":false," +
        "\"vision\":false,\"structuredOutput\":false,\"embeddings\":false," +
        "\"streaming\":true,\"source\":\"conservativeDefault\"}]"

private const val PROMPT_JSON =
    "{\"op_id\":\"op-1\",\"accepted\":true,\"queued\":false}"

private const val ABORT_JSON = "{\"aborted\":[\"op-1\"]}"

private const val PROJECTION_JSON = "{" +
    "\"session\":{\"id\":\"7\",\"title\":\"T\",\"provider\":\"fake\",\"model\":\"m\"," +
    "\"lifecycle\":\"open\"}," +
    "\"state\":{\"machine\":\"streaming\",\"label\":\"streaming\",\"active\":true," +
    "\"terminal\":false}," +
    "\"activeModel\":{\"provider\":\"fake\",\"model\":\"m\",\"variant\":null}," +
    "\"activeTool\":{\"tool\":\"read_file\",\"opId\":\"op-2\",\"startedMs\":5," +
    "\"status\":\"running\"}," +
    "\"progress\":null," +
    "\"filesChanged\":[\"a.rs\",\"b.rs\"]," +
    "\"lastCheckpoint\":null," +
    "\"verification\":[{\"opId\":\"op-3\",\"tool\":\"shell\",\"startedMs\":6," +
    "\"effectStatus\":\"unknown\"}]," +
    "\"contextUsage\":null,\"queued\":2" +
    "}"

private const val TASKS_JSON = "[" +
    "{\"goal\":\"ship it\",\"constraints\":[\"no network\"],\"state\":\"in_progress\"," +
    "\"milestones\":{\"completed\":[\"m1\"],\"open\":[\"m2\"]}," +
    "\"decisions\":[],\"failures\":[],\"changedFiles\":[\"a.rs\"]," +
    "\"tests\":{\"run\":[\"cargo test\"],\"failed\":[]}," +
    "\"preferences\":[]," +
    "\"verification\":[{\"id\":\"v1\",\"detail\":\"failed:check\",\"status\":\"failed\"}]," +
    "\"progress\":null," +
    "\"budget\":{\"maxTokens\":100,\"maxTurns\":5,\"spentTokens\":7,\"spentTurns\":1," +
    "\"maxCostMicro\":50,\"spentCostMicro\":2,\"openReservedMicro\":0}}" +
    "]"

private const val TASK_RUNS_JSON = "[" +
    "{\"task_id\":3,\"run_id\":\"run-1\",\"mode\":\"in_session\",\"state\":\"Running\"," +
    "\"goal\":\"ship it\",\"item_ids\":[\"main\"],\"model\":\"m\"}" +
    "]"

private const val TASK_RUN_STARTED_JSON =
    "{\"task_id\":3,\"run_id\":\"run-1\",\"state\":\"Running\"}"

private const val TASK_RUN_CANCELLED_JSON =
    "{\"run_id\":\"run-1\",\"cancelled\":true}"

private const val AGENTS_JSON = "[" +
    "{\"agent_id\":\"child-1\",\"kind\":\"child\",\"run_id\":\"run-1\"," +
    "\"session_id\":9,\"worktree_id\":1,\"goal\":\"work\",\"state\":\"Running\"," +
    "\"model\":\"m\",\"budget\":1000,\"ownership\":\"Mutating\"," +
    "\"capabilities\":[],\"progress\":null,\"result\":null,\"item_ids\":[\"main\"]}" +
    "]"

private const val AGENT_ACK_JSON = "{\"queuedSeq\":4,\"applied\":false}"

private const val USAGE_JSON = "{" +
    "\"sessions\":2,\"totals\":{\"budget\":100,\"spent\":40},\"perSession\":[]," +
    "\"durable\":{\"sessionsWithCalls\":1," +
    "\"providerCalls\":{\"tokens\":1234,\"prefixObservations\":1," +
    "\"prefixTokens\":100,\"prefixStabilityObservations\":0}," +
    "\"taskSpend\":{\"settledCostMicro\":77}," +
    "\"reservations\":{\"open\":{\"count\":0,\"predictedMicro\":0}," +
    "\"settled\":{\"count\":0,\"predictedMicro\":0,\"spentMicro\":0," +
    "\"providerReportedMicro\":0}," +
    "\"refunded\":{\"count\":0,\"predictedMicro\":0}," +
    "\"uncertain\":{\"count\":0,\"predictedMicro\":0},\"truncated\":false}}," +
    "\"truncated\":false" +
    "}"

private const val SESSION_USAGE_JSON = "{" +
    "\"sessionId\":\"7\",\"providerCalls\":{\"tokens\":10,\"prefixObservations\":[]}," +
    "\"prefixStability\":null,\"tasks\":[{\"taskId\":\"3\"," +
    "\"budget\":{\"maxTokens\":100,\"maxTurns\":5,\"spentTokens\":7,\"spentTurns\":1," +
    "\"maxCostMicro\":50,\"spentCostMicro\":2,\"openReservedMicro\":0}," +
    "\"reservations\":{\"open\":{\"count\":0,\"predictedMicro\":0},\"settled\":{}," +
    "\"refunded\":{},\"uncertain\":{},\"truncated\":false}}]" +
    "}"

private const val VERIFICATION_JSON = "{" +
    "\"owed\":[{\"opId\":\"op-3\",\"tool\":\"shell\",\"startedMs\":6," +
    "\"status\":\"running\",\"effectStatus\":\"unknown\"}]," +
    "\"failedChecks\":[{\"id\":\"v1\",\"detail\":\"failed:check\",\"status\":\"failed\"}]" +
    "}"

private const val TASK_VERIFICATION_JSON = "{" +
    "\"sessionId\":\"7\",\"taskId\":\"3\",\"records\":[{" +
    "\"recordId\":\"r1\",\"revision\":\"rev\",\"workspaceId\":\"1\",\"worktreeId\":\"1\"," +
    "\"treeHash\":null," +
    "\"criteria\":[{\"criterionKey\":\"c1\",\"passed\":true,\"evidence\":null}]," +
    "\"checks\":[{\"check\":\"unit\",\"program\":\"cargo\",\"args\":[\"test\"]," +
    "\"category\":\"test\",\"required\":true,\"status\":\"passed\",\"startedMs\":1," +
    "\"finishedMs\":2,\"exit\":0,\"summary\":null}]," +
    "\"changedFiles\":[{\"path\":\"a.rs\",\"digestHex\":\"aa\",\"size\":3}]," +
    "\"unrelatedChanges\":[],\"reviewer\":null,\"status\":\"passed\"," +
    "\"startedMs\":1,\"completedMs\":2}]}"

private const val EVIDENCE_JSON = "{" +
    "\"id\":9,\"kind\":\"tool_output\",\"sessionId\":7,\"workspaceId\":1," +
    "\"taskId\":null,\"sourceRevision\":null,\"compressibility\":null," +
    "\"backingCompleteness\":null,\"backingRetained\":true,\"backingLen\":5," +
    "\"allowRanges\":true,\"allowSearch\":true,\"maxBytes\":1048576" +
    "}"

/** `{"bytesBase64":"aGVsbG8="}` decodes to `hello`. */
private const val EVIDENCE_RETRIEVAL_JSON = "{" +
    "\"id\":9,\"selector\":{\"selector\":\"all\"},\"bytesBase64\":\"aGVsbG8=\"," +
    "\"byteLen\":5,\"truncatedByPolicy\":false" +
    "}"

private const val MESSAGES_JSON = "{" +
    "\"sessionId\":\"7\",\"messages\":[" +
    "{\"seq\":1,\"id\":1,\"role\":\"user\",\"createdMs\":1,\"data\":{\"text\":\"hi\"}," +
    "\"parts\":[]}," +
    "{\"seq\":2,\"id\":2,\"role\":\"assistant\",\"createdMs\":2,\"data\":{}," +
    "\"parts\":[{\"kind\":\"text\",\"createdMs\":2,\"data\":{\"text\":\"hello\"}}]}" +
    "],\"hasMore\":false,\"nextBefore\":null}"

private const val EVENTS_JSON = "{" +
    "\"sessionId\":\"7\",\"events\":[" +
    "{\"seq\":1,\"kind\":\"session_created\",\"state\":\"idle\",\"opId\":null," +
    "\"tsMs\":1,\"payload\":{}}" +
    "],\"hasMore\":false,\"nextCursor\":null}"

// ------------------------------------------------------------- JSON codec

private fun assertJsonCodec() {
    val value = JsonCodec.parse("{\"a\":1,\"b\":[true,null,\"x\\n\"],\"c\":{\"d\":2.5}}")
    val root = value as JsonValue.Obj
    assertEquals(1L, (root.fields["a"] as JsonValue.Int64).value)
    val array = root.fields["b"] as JsonValue.Arr
    assertEquals(3, array.items.size)
    assertEquals("x\n", (array.items[2] as JsonValue.Str).value)
    val nested = root.fields["c"] as JsonValue.Obj
    assertEquals(2.5, (nested.fields["d"] as JsonValue.Dbl).value)
    assertEquals(
        "{\"a\":1,\"b\":[true,null,\"x\\n\"],\"c\":{\"d\":2.5}}",
        JsonCodec.write(value)
    )
    for (hostile in listOf(
        "{\"a\":1} trailing",
        "\"unterminated",
        "{\"a\":}",
        "[1,2",
        "{\"a\":01x}"
    )) {
        try {
            JsonCodec.parse(hostile)
            fail("hostile JSON must be rejected: $hostile")
        } catch (e: NativeProtocolException) {
            // expected
        }
    }
}

private fun assertRequestBodies() {
    assertEquals(
        "{\"provider\":\"p\",\"model\":\"m\",\"workspace\":\"/ws\",\"title\":\"T\"}",
        NativeRequests.createSession("p", "m", "/ws", "T")
    )
    assertEquals("{\"provider\":\"p\",\"model\":\"m\"}", NativeRequests.createSession("p", "m"))
    assertEquals(
        "{\"session_id\":\"7\",\"prompt\":\"hi\"}",
        NativeRequests.prompt("7", "hi")
    )
    assertEquals("{\"session_id\":\"7\"}", NativeRequests.abort("7"))
    assertEquals(
        "{\"session_id\":\"7\",\"op_id\":\"op\"}",
        NativeRequests.abort("7", "op")
    )
    assertEquals(
        "{\"goal\":\"g\",\"criteria\":[\"c1\"],\"model\":\"m1\",\"max_tokens\":100," +
            "\"max_cost_micro\":200,\"mutation_mode\":\"shadow\"}",
        NativeRequests.startTaskRun("g", listOf("c1"), "m1", 100, 200, "shadow")
    )
    assertEquals("{\"max_cost_micro\":5}", NativeRequests.changeBudget(maxCostMicro = 5))
    assertEquals("{\"selector\":\"all\"}", NativeRequests.evidenceSelectorAll())
    assertEquals("{\"text\":\"a\\\"b\\n\"}", NativeRequests.steer("a\"b\n"))
}

// ---------------------------------------------------------- response parsers

private fun assertResponseParsers() {
    assertEquals(true, parseNativeHealth(HEALTH_JSON).ok)
    assertEquals("1.2.3", parseNativeHealth(HEALTH_JSON).version)
    assertEquals(true, parseNativeReady(READY_JSON).ready)
    val created = parseNativeSessionCreated(CREATED_JSON)
    assertEquals("7", created.id)
    assertEquals("T", created.title)
    assertEquals(1750000000000L, created.createdMs)
    assertEquals("fake", parseNativeSessionList(SESSIONS_JSON)[0].provider)
    assertEquals("m", parseNativeModelCatalog(MODELS_JSON)[0].model)
    assertEquals("op-1", parseNativePromptReceipt(PROMPT_JSON).opId)

    val projection = parseNativeProjection(PROJECTION_JSON)
    assertEquals("7", projection.sessionId)
    assertEquals("streaming", projection.machine)
    assertEquals(true, projection.active)
    assertEquals("fake", projection.activeModel!!.provider)
    assertEquals("read_file", projection.activeTool!!.tool)
    assertEquals(listOf("a.rs", "b.rs"), projection.filesChanged)
    assertEquals(1, projection.verification.size)
    assertEquals(2, projection.queued)

    val tasks = parseNativeTaskViews(TASKS_JSON)
    assertEquals("ship it", tasks[0].goal)
    assertEquals(listOf("m1"), tasks[0].milestones.completed)
    assertEquals(7L, tasks[0].budget!!.spentTokens)
    assertEquals(1, tasks[0].testsRun.size)

    val runs = parseNativeTaskRuns(TASK_RUNS_JSON)
    assertEquals("run-1", runs[0].runId)
    assertEquals(3L, runs[0].taskId)
    assertEquals("Running", parseNativeTaskRunStarted(TASK_RUN_STARTED_JSON).state)
    assertEquals(true, parseNativeTaskRunCancelled(TASK_RUN_CANCELLED_JSON).cancelled)

    val agents = parseNativeAgents(AGENTS_JSON)
    assertEquals("child-1", agents[0].agentId)
    assertEquals(1000L, agents[0].budget)
    val ack = parseNativeAgentControlAck(AGENT_ACK_JSON)
    assertEquals(4L, ack.queuedSeq)
    assertEquals(false, ack.applied)

    val usage = parseNativeUsage(USAGE_JSON)
    assertEquals(2L, usage.sessions)
    assertEquals(1234L, usage.durableTokens)
    assertEquals(77L, usage.settledCostMicro)
    assertEquals("7", parseNativeSessionUsage(SESSION_USAGE_JSON).sessionId)
    assertEquals(10L, parseNativeSessionUsage(SESSION_USAGE_JSON).tokens)

    val verification = parseNativeVerificationView(VERIFICATION_JSON)
    assertEquals(1, verification.owed.size)
    assertEquals("failed:check", verification.failedChecks[0].detail)
    val taskVerification = parseNativeTaskVerification(TASK_VERIFICATION_JSON)
    assertEquals(1, taskVerification.records[0].criteriaPassed)
    assertEquals(listOf("unit=passed"), taskVerification.records[0].checks)

    val evidence = parseNativeEvidence(EVIDENCE_JSON)
    assertEquals(9L, evidence.id)
    assertEquals(true, evidence.allowSearch)
    val retrieval = parseNativeEvidenceRetrieval(EVIDENCE_RETRIEVAL_JSON)
    assertEquals("hello", String(retrieval.bytes, Charsets.UTF_8))
    assertEquals(5L, retrieval.byteLen)

    val messages = dev.faktor.shared.parseNativeMessagePage(MESSAGES_JSON)
    assertEquals("hi", messages.messages[0].text)
    assertEquals("hello", messages.messages[1].text)
    val events = dev.faktor.shared.parseNativeEventPage(EVENTS_JSON)
    assertEquals("session_created", events.events[0].kind)
}

private fun assertHostileParsers() {
    val hostile = listOf(
        "{}",
        "{\"ok\":\"true\",\"version\":\"1\"}",
        "{\"ok\":true}",
        "{\"ok\":true,\"version\":\"1\"} junk"
    )
    for (text in hostile) {
        try {
            parseNativeHealth(text)
            fail("hostile health must be rejected: $text")
        } catch (e: NativeProtocolException) {
            // expected
        }
    }
    try {
        parseNativeEvidenceRetrieval(
            "{\"id\":1,\"selector\":{},\"bytesBase64\":\"!!!\",\"byteLen\":1," +
                "\"truncatedByPolicy\":false}"
        )
        fail("invalid base64 must be rejected")
    } catch (e: NativeProtocolException) {
        // expected
    }
}

// --------------------------------------------------------------- fake daemon

/** One captured request from the fake daemon. */
private class FakeRequest(
    val method: String,
    val path: String,
    val query: Map<String, String>,
    val headers: Map<String, String>,
    val body: String
)

private class FakeSseWriter(private val out: BufferedOutputStream) {
    fun write(text: String) {
        out.write(text.toByteArray(Charsets.UTF_8))
        out.flush()
    }

    fun comment(text: String) {
        write(": $text\n")
    }

    fun frame(id: Long?, event: String, data: String) {
        if (id != null) write("id: $id\n")
        write("event: $event\n")
        write("data: $data\n\n")
    }
}

private class FakeResponse(private val out: BufferedOutputStream) {
    fun json(status: Int, body: String) {
        send(status, "application/json", body.toByteArray(Charsets.UTF_8))
    }

    fun text(status: Int, body: String) {
        send(status, "text/plain", body.toByteArray(Charsets.UTF_8))
    }

    private fun send(status: Int, contentType: String, bytes: ByteArray) {
        val head = "HTTP/1.1 $status X\r\nContent-Type: $contentType\r\n" +
            "Content-Length: ${bytes.size}\r\nConnection: close\r\n\r\n"
        out.write(head.toByteArray(Charsets.UTF_8))
        out.write(bytes)
        out.flush()
    }

    fun stream(status: Int, contentType: String, block: (FakeSseWriter) -> Unit) {
        val head = "HTTP/1.1 $status X\r\nContent-Type: $contentType\r\n" +
            "Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        out.write(head.toByteArray(Charsets.UTF_8))
        out.flush()
        block(FakeSseWriter(out))
    }
}

/**
 * Raw-socket HTTP/1.1 fake: exact method+path routing, request capture,
 * chunk-free streaming responses. One thread per connection; all threads
 * are daemons, so a test can abandon a held SSE stream.
 */
private class FakeDaemon {
    private val server = ServerSocket(0, 50, InetAddress.getByName("127.0.0.1"))
    val requests: MutableList<FakeRequest> = Collections.synchronizedList(ArrayList<FakeRequest>())
    private val handlers =
        Collections.synchronizedMap(HashMap<String, (FakeRequest, FakeResponse) -> Unit>())
    @Volatile private var running = true
    private var acceptThread: Thread? = null

    val baseUrl: String
        get() = "http://127.0.0.1:" + server.localPort

    fun on(method: String, path: String, handler: (FakeRequest, FakeResponse) -> Unit) {
        handlers["$method $path"] = handler
    }

    fun start() {
        val thread = Thread({ acceptLoop() }, "fake-daemon-accept")
        thread.isDaemon = true
        acceptThread = thread
        thread.start()
    }

    fun stop() {
        running = false
        try {
            server.close()
        } catch (e: Exception) {
            // Already closed.
        }
        try {
            acceptThread?.join(500)
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        }
    }

    fun requestCount(method: String, path: String): Int = synchronized(requests) {
        requests.count { it.method == method && it.path == path }
    }

    private fun acceptLoop() {
        while (running) {
            val socket = try {
                server.accept()
            } catch (e: Exception) {
                break
            }
            val thread = Thread({ handle(socket) }, "fake-daemon-conn")
            thread.isDaemon = true
            thread.start()
        }
    }

    private fun handle(socket: java.net.Socket) {
        try {
            socket.use {
                val input = BufferedInputStream(socket.getInputStream())
                val requestLine = readLine(input) ?: return
                val parts = requestLine.split(' ')
                if (parts.size < 2) return
                val method = parts[0]
                val target = parts[1]
                val qIndex = target.indexOf('?')
                val path = if (qIndex < 0) target else target.substring(0, qIndex)
                val query = LinkedHashMap<String, String>()
                if (qIndex >= 0) {
                    for (pair in target.substring(qIndex + 1).split('&')) {
                        if (pair.isEmpty()) continue
                        val eq = pair.indexOf('=')
                        if (eq < 0) {
                            query[pair] = ""
                        } else {
                            query[pair.substring(0, eq)] = pair.substring(eq + 1)
                        }
                    }
                }
                val headers = LinkedHashMap<String, String>()
                while (true) {
                    val header = readLine(input) ?: return
                    if (header.isEmpty()) break
                    val colon = header.indexOf(':')
                    if (colon > 0) {
                        headers[lowerAscii(header.substring(0, colon).trim())] =
                            header.substring(colon + 1).trim()
                    }
                }
                val length = headers["content-length"]?.toIntOrNull() ?: 0
                val bodyBytes = ByteArray(length)
                var read = 0
                while (read < length) {
                    val n = input.read(bodyBytes, read, length - read)
                    if (n < 0) break
                    read += n
                }
                val request = FakeRequest(
                    method, path, query, headers,
                    String(bodyBytes, 0, read, Charsets.UTF_8)
                )
                requests.add(request)
                val response = FakeResponse(BufferedOutputStream(socket.getOutputStream()))
                val handler = handlers["$method $path"]
                if (handler == null) {
                    response.json(
                        404,
                        "{\"error\":{\"code\":\"not_found\",\"message\":\"no route\"," +
                            "\"retryable\":false}}"
                    )
                } else {
                    handler(request, response)
                }
            }
        } catch (e: Exception) {
            // Client disconnect; the test already has what it needs.
        }
    }

    /** ASCII-only lowercase (works on kotlinc 1.3, unlike String.lowercase()). */
    private fun lowerAscii(text: String): String {
        val sb = StringBuilder(text.length)
        for (c in text) {
            sb.append(if (c in 'A'..'Z') (c + 32).toChar() else c)
        }
        return sb.toString()
    }

    private fun readLine(input: InputStream): String? {
        val out = StringBuilder()
        while (true) {
            val c = input.read()
            if (c < 0) return if (out.isEmpty()) null else out.toString()
            if (c == '\n'.code) return out.toString()
            if (c != '\r'.code) out.append(c.toChar())
            if (out.length > 65536) return out.toString()
        }
    }
}

// ------------------------------------------------------------- client routes

private fun assertClientRoutes() {
    val daemon = FakeDaemon()
    daemon.on("GET", "/native/health") { _, response -> response.json(200, HEALTH_JSON) }
    daemon.on("GET", "/native/ready") { _, response -> response.json(200, READY_JSON) }
    daemon.on("POST", "/session/create") { _, response -> response.json(200, CREATED_JSON) }
    daemon.on("GET", "/session/list") { _, response -> response.json(200, SESSIONS_JSON) }
    daemon.on("GET", "/models") { _, response -> response.json(200, MODELS_JSON) }
    daemon.on("POST", "/session/prompt") { _, response -> response.json(200, PROMPT_JSON) }
    daemon.on("POST", "/native/session/7/abort") { _, response -> response.json(200, ABORT_JSON) }
    daemon.on("GET", "/session/7/projection") { _, response ->
        response.json(200, PROJECTION_JSON)
    }
    daemon.on("GET", "/native/messages") { _, response -> response.json(200, MESSAGES_JSON) }
    daemon.on("GET", "/native/events") { _, response -> response.json(200, EVENTS_JSON) }
    daemon.on("GET", "/native/session/7/tasks") { _, response -> response.json(200, TASKS_JSON) }
    daemon.on("GET", "/native/session/7/task-runs") { _, response ->
        response.json(200, TASK_RUNS_JSON)
    }
    daemon.on("POST", "/native/session/7/task-runs") { _, response ->
        response.json(200, TASK_RUN_STARTED_JSON)
    }
    daemon.on("POST", "/native/session/7/task-runs/run-1/cancel") { _, response ->
        response.json(200, TASK_RUN_CANCELLED_JSON)
    }
    daemon.on("GET", "/native/agents") { _, response -> response.json(200, AGENTS_JSON) }
    for (action in listOf("pause", "resume", "cancel", "retry", "steer", "model", "budget")) {
        daemon.on("POST", "/native/agents/child-1/$action") { _, response ->
            response.json(200, AGENT_ACK_JSON)
        }
    }
    daemon.on("GET", "/native/usage") { _, response -> response.json(200, USAGE_JSON) }
    daemon.on("GET", "/native/session/7/usage") { _, response ->
        response.json(200, SESSION_USAGE_JSON)
    }
    daemon.on("GET", "/native/session/7/verification") { _, response ->
        response.json(200, VERIFICATION_JSON)
    }
    daemon.on("GET", "/native/session/7/tasks/3/verification") { _, response ->
        response.json(200, TASK_VERIFICATION_JSON)
    }
    daemon.on("GET", "/native/evidence/9") { _, response -> response.json(200, EVIDENCE_JSON) }
    daemon.on("POST", "/native/evidence/9/retrieve") { _, response ->
        response.json(200, EVIDENCE_RETRIEVAL_JSON)
    }
    daemon.start()
    try {
        val client = NativeClient(daemon.baseUrl, "tok")
        assertEquals(true, client.health().ok)
        client.createSession("p", "m", "/ws", "T")
        client.prompt("7", "hi")
        client.abortSession("7", "op-1")
        assertEquals("streaming", client.projection("7").machine)
        assertEquals(2, client.messages("7", limit = 20).messages.size)
        assertEquals(1, client.events("7", after = 0).events.size)
        assertEquals("run-1", client.taskRuns("7")[0].runId)
        assertEquals("run-1", client.taskRunState("7", "run-1").runId)
        client.startTaskRun("7", "g")
        client.cancelTaskRun("7", "run-1")
        assertEquals("child-1", client.agents("7")[0].agentId)
        client.pauseAgent("child-1")
        client.resumeAgent("child-1")
        client.cancelAgent("child-1")
        client.retryAgent("child-1")
        client.steerAgent("child-1", "note")
        client.setAgentModel("child-1", "m")
        client.setAgentBudget("child-1", maxTokens = 1000)
        assertEquals(2L, client.usage().sessions)
        assertEquals("7", client.sessionUsage("7").sessionId)
        assertEquals(1, client.verification("7").owed.size)
        assertEquals("3", client.taskVerification("7", "3").taskId)
        assertEquals(9L, client.evidence("7", 9).id)
        assertEquals("hello", String(client.retrieveEvidence("7", 9, "{\"selector\":\"all\"}").bytes))

        val first = daemon.requests[0]
        assertEquals("Bearer tok", first.headers["authorization"])
        val create = daemon.requests.first { it.method == "POST" && it.path == "/session/create" }
        assertEquals(
            "{\"provider\":\"p\",\"model\":\"m\",\"workspace\":\"/ws\",\"title\":\"T\"}",
            create.body
        )
        val prompt = daemon.requests.first { it.path == "/session/prompt" }
        assertEquals("{\"session_id\":\"7\",\"prompt\":\"hi\"}", prompt.body)
        for (action in listOf("pause", "resume", "cancel", "retry", "steer", "model", "budget")) {
            assertEquals(
                1,
                daemon.requestCount("POST", "/native/agents/child-1/$action")
            )
        }
    } finally {
        daemon.stop()
    }
}

private fun assertErrorMapping() {
    val daemon = FakeDaemon()
    daemon.on("GET", "/native/health") { _, response ->
        response.json(
            401,
            "{\"error\":{\"code\":\"unauthorized\",\"message\":\"bad password\"," +
                "\"retryable\":false}}"
        )
    }
    daemon.on("GET", "/native/ready") { _, response ->
        response.text(500, "boom")
    }
    daemon.start()
    try {
        val client = NativeClient(daemon.baseUrl, "wrong")
        try {
            client.health()
            fail("401 must throw")
        } catch (e: NativeApiException) {
            assertEquals(401, e.status)
            assertEquals("unauthorized", e.code)
            assertEquals(false, e.retryable)
        }
        try {
            client.ready()
            fail("500 must throw")
        } catch (e: NativeApiException) {
            assertEquals(500, e.status)
            assertEquals("http_error", e.code)
        }
    } finally {
        daemon.stop()
    }
}

private fun assertBodyBound() {
    val daemon = FakeDaemon()
    val big = ByteArray(4096) { 'x'.code.toByte() }
    daemon.on("GET", "/native/health") { _, response ->
        response.json(200, "{\"ok\":true,\"version\":\"" + String(big) + "\"}")
    }
    daemon.start()
    try {
        val client = NativeClient(daemon.baseUrl, "tok", maxBodyBytes = 64)
        try {
            client.health()
            fail("oversized body must be rejected")
        } catch (e: NativeProtocolException) {
            assertTrue(
                e.detail.contains("exceeded bound"),
                "detail must name the bound: ${e.detail}"
            )
        }
    } finally {
        daemon.stop()
    }
}

// ------------------------------------------------------------------- SSE

private fun awaitLatch(latch: CountDownLatch, timeoutMs: Long, what: String) {
    if (!latch.await(timeoutMs, TimeUnit.MILLISECONDS)) {
        fail("timed out waiting for $what")
    }
}

private fun assertSseStreaming() {
    val daemon = FakeDaemon()
    val held = CountDownLatch(1)
    daemon.on("GET", "/api/session/42/events") { request, response ->
        val after = request.query["events_after"]?.toLongOrNull() ?: 0L
        if (after < 1L) {
            response.stream(200, "text/event-stream") { writer ->
                writer.comment("keep-alive")
                writer.write("event: heartbeat\ndata: {}\n\n")
                writer.frame(1, "agent_state_changed", "{\"event\":\"agent_state_changed\",\"state\":\"streaming\"}")
            }
        } else {
            response.stream(200, "text/event-stream") { writer ->
                writer.frame(2, "agent_state_changed", "{\"event\":\"agent_state_changed\",\"state\":\"ready\"}")
                held.await(10, TimeUnit.SECONDS)
            }
        }
    }
    daemon.start()
    val events = Collections.synchronizedList(ArrayList<NativeSseEvent>())
    val delivered = CountDownLatch(2)
    val errors = Collections.synchronizedList(ArrayList<String>())
    val stream = NativeEventStream(
        daemon.baseUrl, "tok", "42", 0,
        minBackoffMs = 10, maxBackoffMs = 50,
        onEvent = { event ->
            events.add(event)
            delivered.countDown()
        },
        onError = { error -> errors.add(error.message ?: "error") }
    )
    try {
        stream.start()
        awaitLatch(delivered, 10_000, "two SSE frames")
        assertEquals(2, events.size)
        assertEquals(1L, events[0].id)
        assertEquals(2L, events[1].id)
        assertEquals(2L, stream.cursor)
        assertTrue(errors.isEmpty(), "no frame errors expected: $errors")
        val second = daemon.requests.first {
            it.path == "/api/session/42/events" && it.query["events_after"] == "1"
        }
        assertEquals("1", second.query["events_after"])
        assertEquals("Bearer tok", second.headers["authorization"])
        assertEquals("text/event-stream", second.headers["accept"])
    } finally {
        stream.stop()
        held.countDown()
        daemon.stop()
    }
}

private fun assertSseOversizedFrame() {
    val daemon = FakeDaemon()
    val held = CountDownLatch(1)
    val bigData = StringBuilder("{\"x\":\"")
    for (i in 0 until 200) bigData.append('a')
    bigData.append("\"}")
    daemon.on("GET", "/api/session/9/events") { _, response ->
        response.stream(200, "text/event-stream") { writer ->
            writer.frame(9, "agent_state_changed", bigData.toString())
            writer.frame(10, "agent_state_changed", "{\"event\":\"agent_state_changed\"}")
            held.await(10, TimeUnit.SECONDS)
        }
    }
    daemon.start()
    val events = Collections.synchronizedList(ArrayList<NativeSseEvent>())
    val delivered = CountDownLatch(1)
    val errors = Collections.synchronizedList(ArrayList<String>())
    val stream = NativeEventStream(
        daemon.baseUrl, "tok", "9", 0,
        maxFrameBytes = 64, minBackoffMs = 10, maxBackoffMs = 50,
        onEvent = { event ->
            events.add(event)
            delivered.countDown()
        },
        onError = { error -> errors.add(error.message ?: "error") }
    )
    try {
        stream.start()
        awaitLatch(delivered, 10_000, "the valid frame after the oversized one")
        assertEquals(1, events.size)
        assertEquals(10L, events[0].id)
        assertEquals(10L, stream.cursor)
        assertTrue(errors.isNotEmpty(), "oversized frame must be reported loudly")
        assertTrue(
            errors.any { it.contains("exceeded") },
            "error must name the bound: $errors"
        )
    } finally {
        stream.stop()
        held.countDown()
        daemon.stop()
    }
}

// ------------------------------------------------------- real-daemon smoke

/**
 * End-to-end native bridge smoke against a REAL daemon binary (args[0]):
 * lifecycle -> bearer auth -> session -> prompt -> SSE -> projections ->
 * task-run/agent/usage/verification/evidence routes -> stop.
 */
object NativeBridgeSmoke {

    private var failures = 0

    @JvmStatic
    fun main(args: Array<String>) {
        if (args.isEmpty()) {
            println("FAIL usage: NativeBridgeSmoke <faktor-cli binary path>")
            kotlin.system.exitProcess(1)
        }
        val binary = Paths.get(args[0])
        val dataDir = Files.createTempDirectory("faktor-native-smoke-")

        step("native protocol unit assertions") { NativeClientTest.runAll() }

        val manager = BackendProcessManager(binary, dataDir)
        var connection: BackendConnection? = null
        var stream: NativeEventStream? = null
        try {
            step("start daemon (startup line + port)") {
                connection = manager.start()
                println("  port=${connection!!.port} pid=${connection!!.pid()}")
            }
            val conn = connection
            if (conn != null) {
                val client = NativeClient.forConnection(conn)
                step("native health via bearer auth") {
                    val health = client.health()
                    if (!health.ok) fail("native health ok=false")
                    println("  version=${health.version}")
                }
                step("native readiness") {
                    val ready = client.awaitReady(10_000L)
                    if (!ready.ready) fail("daemon never became ready")
                }
                var sessionId: String? = null
                step("create native session (POST /session/create)") {
                    val created = client.createSession(
                        "default", "default", null, "native bridge smoke"
                    )
                    sessionId = created.id
                    if (created.id.isEmpty()) fail("empty session id")
                    println("  session=${created.id}")
                }
                val sid = sessionId
                if (sid != null) {
                    val events = Collections.synchronizedList(ArrayList<NativeSseEvent>())
                    step("open SSE journal stream (cursor 0)") {
                        val s = NativeEventStream.forConnection(
                            conn, sid, 0,
                            onEvent = { event -> events.add(event) },
                            onError = { error -> println("  sse error: ${error.message}") }
                        )
                        stream = s
                        s.start()
                    }
                    step("prompt (POST /session/prompt)") {
                        val receipt = client.prompt(sid, "ping from native bridge smoke")
                        if (!receipt.accepted) fail("prompt not accepted")
                        println("  op=${receipt.opId} queued=${receipt.queued}")
                    }
                    step("projection settles (ready | failed_*)") {
                        val deadline = System.currentTimeMillis() + 20_000L
                        var machine = "unknown"
                        while (System.currentTimeMillis() < deadline) {
                            machine = client.projection(sid).machine
                            if (machine == "ready_for_next_turn" ||
                                machine == "failed_recoverable" ||
                                machine == "failed_permanent" ||
                                machine == "ready"
                            ) {
                                break
                            }
                            Thread.sleep(200L)
                        }
                        println("  machine=$machine")
                        if (machine != "ready_for_next_turn" &&
                            machine != "failed_recoverable" &&
                            machine != "failed_permanent" &&
                            machine != "ready"
                        ) {
                            fail("session did not settle: $machine")
                        }
                    }
                    step("messages page carries the prompt") {
                        val page = client.messages(sid, limit = 20)
                        if (page.messages.isEmpty()) fail("no messages")
                        if (page.messages.none { it.text.contains("ping from native bridge smoke") }) {
                            fail("prompt not visible in messages page")
                        }
                    }
                    step("SSE delivered frames and advanced the cursor") {
                        if (events.isEmpty()) fail("no SSE frames")
                        println("  frames=${events.size} cursor=${stream!!.cursor}")
                        if (stream!!.cursor <= 0L) fail("cursor did not advance")
                    }
                    step("journal event page (cursor twin)") {
                        val page = client.events(sid, after = 0, limit = 64)
                        if (page.events.isEmpty()) fail("no journal events")
                    }
                    step("task-run listing (GET task-runs)") {
                        println("  runs=${client.taskRuns(sid).size}")
                    }
                    step("agent listing (GET native agents)") {
                        println("  agents=${client.agents(sid).size}")
                    }
                    step("global usage (GET /native/usage)") {
                        val usage = client.usage()
                        println("  sessions=${usage.sessions} durableTokens=${usage.durableTokens}")
                    }
                    step("session usage (GET session usage)") {
                        val usage = client.sessionUsage(sid)
                        if (usage.sessionId != sid) fail("usage session mismatch")
                    }
                    step("verification view") {
                        val verification = client.verification(sid)
                        println(
                            "  owed=${verification.owed.size} failed=${verification.failedChecks.size}"
                        )
                    }
                    step("task views (GET session tasks)") {
                        println("  tasks=${client.taskViews(sid).size}")
                    }
                    step("evidence access is typed (404 unknown / 503 unwired)") {
                        try {
                            client.evidence(sid, 1L)
                            fail("unknown evidence id must not answer 200")
                        } catch (e: NativeApiException) {
                            if (e.status != 404 && e.status != 503) {
                                fail("unexpected evidence error ${e.status} ${e.code}")
                            }
                        }
                    }
                }
            }
        } finally {
            stream?.stop()
            if (connection != null) {
                val c = connection!!
                try {
                    manager.stop(c)
                    if (c.process.isAlive) fail("daemon still alive after stop")
                    println("PASS stop daemon")
                } catch (e: Throwable) {
                    failures++
                    println("FAIL stop daemon: ${e.message}")
                    c.process.destroyForcibly()
                }
            }
            dataDir.toFile().deleteRecursively()
        }
        println(if (failures == 0) "NATIVE SMOKE PASS" else "NATIVE SMOKE FAIL ($failures)")
        kotlin.system.exitProcess(if (failures == 0) 0 else 1)
    }

    private fun step(name: String, body: () -> Unit) {
        try {
            body()
            println("PASS $name")
        } catch (e: Throwable) {
            failures++
            println("FAIL $name: ${e.message}")
        }
    }
}
