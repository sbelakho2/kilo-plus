// Dependency-free tests for the split-mode backend manager, plus a
// plain-main smoke runner that exercises the FULL flow against a real
// daemon binary.
//
// No kotlin.test / JUnit: the assertion helpers below use plain Kotlin
// `check`/`require`, so the whole file compiles with plain kotlinc (no
// network, no Gradle) and `compile-and-smoke.sh` runs `BackendSmoke
// <binary>` as a self-contained main that exits 0/1. When the frozen
// frontend lands and the plugin gains a real Gradle build, the same checks
// can be re-annotated with kotlin.test.
package dev.kilop.backend

import dev.kilop.shared.BasicAuth
import dev.kilop.shared.HealthResult
import dev.kilop.shared.MessageModel
import dev.kilop.shared.MessageSendRequest
import dev.kilop.shared.Model
import dev.kilop.shared.Part
import dev.kilop.shared.SessionCreateRequest
import dev.kilop.shared.StartupLine
import dev.kilop.shared.parseHealth
import dev.kilop.shared.parseMessageCount
import dev.kilop.shared.parseMessageId
import dev.kilop.shared.parseSessionId
import dev.kilop.shared.parseSessionState
import java.nio.file.Files
import java.nio.file.Paths
object BackendProcessManagerTest {

    @JvmStatic
    fun runAll() {
        assertFixtureStartupLine()
        assertFixtureAuthHeader()
        assertRequestJsonShapes()
        assertResponseParsers()
        assertMissingBinaryFailsLoudly()
        println("PASS all unit assertions")
    }
}

fun assertEquals(expected: Any?, actual: Any?, message: String? = null) {
    check(expected == actual) { "${message ?: "assertion"}: expected $expected, actual $actual" }
}

fun assertTrue(cond: Boolean, message: String? = null) {
    check(cond) { message ?: "assertion failed" }
}

fun fail(message: String): Nothing = throw IllegalStateException(message)

/**
 * The self-contained smoke: runs every step against a REAL daemon binary
 * (args[0]) and exits 0/1. Prints PASS/FAIL per step.
 */
object BackendSmoke {

    private var failures = 0

    @JvmStatic
    fun main(args: Array<String>) {
        if (args.isEmpty()) {
            println("FAIL usage: BackendSmoke <kilo+ binary path>")
            kotlin.system.exitProcess(1)
        }
        val binary = Paths.get(args[0])
        val dataDir = Files.createTempDirectory("kilop-smoke-")

        step("unit assertions") { BackendProcessManagerTest.runAll() }

        val manager = BackendProcessManager(binary, dataDir)
        var connection: BackendConnection? = null
        var sessionId: String? = null
        try {
            step("start daemon (startup line + port)") {
                connection = manager.start()
                println("  port=${connection!!.port} pid=${connection!!.pid()}")
            }
            if (connection != null) {
                step("health (GET /global/health)") {
                    val h: HealthResult = manager.health(connection!!)
                    if (!h.ok) fail("health ok=false")
                    if (h.version.isEmpty()) fail("health version is empty")
                    if (h.protocol.isEmpty()) fail("health protocol is empty")
                }
                step("create session (POST /session)") {
                    sessionId = manager.createSession(connection!!, "kotlin split-mode smoke")
                    if (sessionId!!.isEmpty()) fail("empty sessionID")
                }
                step("send message (POST /session/{id}/message)") {
                    val messageId = manager.sendMessage(
                        connection!!, sessionId!!, "ping from kotlin smoke"
                    )
                    if (messageId.isEmpty()) fail("empty messageID")
                }
                step("session state settles (ready_for_next_turn | failed_recoverable)") {
                    val state = manager.awaitSettledState(connection!!, sessionId!!)
                    println("  state=$state")
                    if (state != "ready_for_next_turn" && state != "failed_recoverable") {
                        fail("unexpected settled state $state")
                    }
                }
                step("list messages (GET /session/{id}/message?limit=5)") {
                    val count = manager.listMessages(connection!!, sessionId!!)
                    println("  messages=$count")
                    if (count < 1) fail("expected >= 1 message, got $count")
                }
            }
        } finally {
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
        println(if (failures == 0) "SMOKE PASS" else "SMOKE FAIL ($failures)")
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

// ------------------------------------------------------------------ fixtures

private fun assertFixtureStartupLine() {
    val line = "kilo server listening on http://127.0.0.1:45678"
    val parsed = StartupLine.parse(line)
    assertTrue(parsed != null, "fixture startup line must parse")
    assertEquals(45678, parsed!!.port, "fixture port")
    assertEquals(
        "kilo server listening on http://127.0.0.1:45678",
        parsed.toString(),
        "roundtrip"
    )
    assertEquals(null, StartupLine.parse(""), "empty line must not parse")
    assertEquals(null, StartupLine.parse("kilo server listening on http://127.0.0.1"), "no port")
    assertEquals(
        null,
        StartupLine.parse("kilo server listening on http://127.0.0.1:0x10"),
        "hex port must not parse"
    )
    assertEquals(
        null,
        StartupLine.parse("noise kilo server listening on http://127.0.0.1:45678"),
        "leading junk must not parse"
    )
}

private fun assertFixtureAuthHeader() {
    val password = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    val auth = BasicAuth(password)
    assertEquals(
        "Basic a2lsbzowMTIzNDU2Nzg5YWJjZGVmMDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWYwMTIzNDU2Nzg5YWJjZGVm",
        auth.headerValue,
        "header must equal the frozen fixture (Basic <base64('kilo:'+password)>)"
    )
    assertEquals("Authorization", BasicAuth.HEADER_NAME)
}

private fun assertRequestJsonShapes() {
    val create = SessionCreateRequest(
        title = "T",
        model = Model(id = "m1", providerID = "ollama", variant = "fast")
    )
    assertEquals(
        "{\"title\":\"T\",\"model\":{\"id\":\"m1\",\"providerID\":\"ollama\",\"variant\":\"fast\"}}",
        create.toJson()
    )
    assertEquals("{\"title\":\"T\"}", SessionCreateRequest(title = "T").toJson())
    assertEquals("{}", SessionCreateRequest().toJson())

    val send = MessageSendRequest(
        model = MessageModel(providerID = "ollama", modelID = "qwen3.8"),
        parts = listOf(Part.TextPart("hi"))
    )
    assertEquals(
        "{\"model\":{\"providerID\":\"ollama\",\"modelID\":\"qwen3.8\"}," +
            "\"parts\":[{\"type\":\"text\",\"text\":\"hi\"}]}",
        send.toJson()
    )

    val tool = MessageSendRequest(
        model = MessageModel(providerID = "p", modelID = "m"),
        parts = listOf(Part.ToolPart(callId = "c1", name = "read_file", input = "{}"))
    )
    assertEquals(
        "{\"model\":{\"providerID\":\"p\",\"modelID\":\"m\"}," +
            "\"parts\":[{\"type\":\"tool\",\"callID\":\"c1\",\"name\":\"read_file\",\"input\":{}}]}",
        tool.toJson()
    )

    val escaped = MessageSendRequest(
        model = MessageModel(providerID = "p", modelID = "m"),
        parts = listOf(Part.TextPart("say \"hi\"\nnext"))
    )
    assertEquals(
        "{\"model\":{\"providerID\":\"p\",\"modelID\":\"m\"}," +
            "\"parts\":[{\"type\":\"text\",\"text\":\"say \\\"hi\\\"\\nnext\"}]}",
        escaped.toJson()
    )
}

private fun assertResponseParsers() {
    assertEquals(
        "42",
        parseSessionId("{\"sessionID\":\"42\",\"title\":\"T\",\"createdMs\":1750000000000}")
    )
    assertEquals(
        "7",
        parseMessageId("{\"messageID\":\"7\",\"accepted\":true,\"queued\":false}")
    )
    val h = parseHealth("{\"ok\":true,\"version\":\"0.5.0\",\"protocol\":\"v756\"}")
    assertTrue(h.ok, "health ok")
    assertEquals("0.5.0", h.version)
    assertEquals("v756", h.protocol)
    assertEquals(
        2,
        parseMessageCount(
            "{\"sessionID\":\"1\",\"hasMore\":false," +
                "\"messages\":[" +
                "{\"messageID\":\"1\",\"role\":\"user\",\"parts\":[],\"createdMs\":1}," +
                "{\"messageID\":\"2\",\"role\":\"assistant\",\"parts\":[],\"createdMs\":2}" +
                "]}"
        )
    )
    assertEquals(
        "failed_recoverable",
        parseSessionState(
            "{\"sessionID\":\"1\",\"title\":\"t\",\"state\":\"failed_recoverable\"," +
                "\"createdMs\":1,\"updatedMs\":2}"
        )
    )
}

private fun assertMissingBinaryFailsLoudly() {
    try {
        BackendProcessManager(
            Paths.get("/nonexistent/kilo-plus-bin"),
            Files.createTempDirectory("kilop-missing-")
        ).start()
        fail("start() must fail loudly for a missing binary")
    } catch (e: BackendException) {
        assertTrue(e.message!!.contains("not found"), "message: ${e.message}")
    }
}
