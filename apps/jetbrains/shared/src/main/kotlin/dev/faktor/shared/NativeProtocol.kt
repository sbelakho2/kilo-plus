// Faktor Native Protocol v1 (docs/native-protocol.md) as plain Kotlin data
// classes. This is the daemon's OWN HTTP/SSE surface, not the frozen
// v7.5.6 wire contract in Protocol.kt: the IDE shells post typed native
// requests and render native projections, never fabricated v7.5.6 frames
// (docs/native-protocol.md, "UI-adaptation principle").
//
// Zero external dependencies on purpose: a small JSON value model with a
// recursive-descent reader and a writer, plus typed accessors that fail
// loudly on shape drift (`NativeProtocolException`) and typed API errors
// (`NativeApiException`). The authoritative shapes live in
// crates/server/src/native/*.rs and are mirrored by the VS Code
// `apps/vscode/src/nativeClient.ts` validators.
package dev.faktor.shared

import java.util.Base64

/** A native-protocol response or request-body shape violation. */
class NativeProtocolException(val path: String, val detail: String) :
    Exception("native protocol violation at $path: $detail")

/** A structured error body `{error: {code, message, retryable}}` from the daemon. */
class NativeApiException(
    val status: Int,
    val code: String,
    val detail: String,
    val retryable: Boolean
) : Exception("native API error $status $code: $detail")

// ---------------------------------------------------------------- JSON model

/** A parsed JSON value; integers stay exact (`Int64`), floats are `Dbl`. */
sealed class JsonValue {
    object Null : JsonValue()
    data class Bool(val value: Boolean) : JsonValue()
    data class Int64(val value: Long) : JsonValue()
    data class Dbl(val value: Double) : JsonValue()
    data class Str(val value: String) : JsonValue()
    data class Arr(val items: List<JsonValue>) : JsonValue()
    data class Obj(val fields: Map<String, JsonValue>) : JsonValue()
}

/** Serializes a JSON value tree; rejects non-finite numbers loudly. */
object JsonCodec {
    fun parse(text: String): JsonValue {
        val parser = Parser(text)
        val value = parser.parseValue()
        parser.skipWs()
        if (!parser.atEnd()) {
            throw NativeProtocolException("json", "trailing characters after JSON value")
        }
        return value
    }

    fun write(value: JsonValue): String {
        val sb = StringBuilder()
        writeValue(sb, value)
        return sb.toString()
    }

    private fun writeValue(sb: StringBuilder, value: JsonValue) {
        when (value) {
            is JsonValue.Null -> sb.append("null")
            is JsonValue.Bool -> sb.append(if (value.value) "true" else "false")
            is JsonValue.Int64 -> sb.append(value.value)
            is JsonValue.Dbl -> {
                if (value.value.isNaN() || value.value.isInfinite()) {
                    throw NativeProtocolException("json", "non-finite number is not valid JSON")
                }
                sb.append(value.value)
            }
            is JsonValue.Str -> writeString(sb, value.value)
            is JsonValue.Arr -> {
                sb.append('[')
                for ((i, item) in value.items.withIndex()) {
                    if (i > 0) sb.append(',')
                    writeValue(sb, item)
                }
                sb.append(']')
            }
            is JsonValue.Obj -> {
                sb.append('{')
                var first = true
                for ((key, field) in value.fields) {
                    if (!first) sb.append(',')
                    first = false
                    writeString(sb, key)
                    sb.append(':')
                    writeValue(sb, field)
                }
                sb.append('}')
            }
        }
    }

    /** RFC 8259 string escaping. */
    fun writeString(sb: StringBuilder, s: String) {
        sb.append('"')
        for (c in s) {
            when (c) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                '\b' -> sb.append("\\b")
                '\u000C' -> sb.append("\\f")
                else -> if (c < ' ') sb.append(String.format("\\u%04x", c.toInt())) else sb.append(c)
            }
        }
        sb.append('"')
    }

    private class Parser(private val s: String) {
        private var i = 0

        fun atEnd(): Boolean = i >= s.length

        fun skipWs() {
            while (i < s.length && isWs(s[i])) i++
        }

        private fun isWs(c: Char): Boolean = c == ' ' || c == '\t' || c == '\n' || c == '\r'

        fun parseValue(): JsonValue {
            skipWs()
            if (i >= s.length) throw NativeProtocolException("json", "unexpected end of JSON")
            return when (s[i]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> JsonValue.Str(parseString())
                't' -> {
                    expect("true")
                    JsonValue.Bool(true)
                }
                'f' -> {
                    expect("false")
                    JsonValue.Bool(false)
                }
                'n' -> {
                    expect("null")
                    JsonValue.Null
                }
                else -> parseNumber()
            }
        }

        private fun expect(word: String) {
            if (!s.startsWith(word, i)) {
                throw NativeProtocolException("json", "malformed literal at offset $i")
            }
            i += word.length
        }

        private fun parseObject(): JsonValue {
            i++
            val fields = LinkedHashMap<String, JsonValue>()
            skipWs()
            if (i < s.length && s[i] == '}') {
                i++
                return JsonValue.Obj(fields)
            }
            while (true) {
                skipWs()
                if (i >= s.length || s[i] != '"') {
                    throw NativeProtocolException("json", "expected string key in object")
                }
                val key = parseString()
                skipWs()
                if (i >= s.length || s[i] != ':') {
                    throw NativeProtocolException("json", "expected ':' after object key")
                }
                i++
                fields[key] = parseValue()
                skipWs()
                if (i >= s.length) throw NativeProtocolException("json", "unterminated object")
                when (s[i]) {
                    ',' -> i++
                    '}' -> {
                        i++
                        return JsonValue.Obj(fields)
                    }
                    else -> throw NativeProtocolException("json", "expected ',' or '}' in object")
                }
            }
        }

        private fun parseArray(): JsonValue {
            i++
            val items = ArrayList<JsonValue>()
            skipWs()
            if (i < s.length && s[i] == ']') {
                i++
                return JsonValue.Arr(items)
            }
            while (true) {
                items.add(parseValue())
                skipWs()
                if (i >= s.length) throw NativeProtocolException("json", "unterminated array")
                when (s[i]) {
                    ',' -> i++
                    ']' -> {
                        i++
                        return JsonValue.Arr(items)
                    }
                    else -> throw NativeProtocolException("json", "expected ',' or ']' in array")
                }
            }
        }

        private fun parseString(): String {
            i++
            val sb = StringBuilder()
            while (true) {
                if (i >= s.length) throw NativeProtocolException("json", "unterminated string")
                val c = s[i]
                i++
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        if (i >= s.length) throw NativeProtocolException("json", "unterminated escape")
                        val e = s[i]
                        i++
                        when (e) {
                            '"' -> sb.append('"')
                            '\\' -> sb.append('\\')
                            '/' -> sb.append('/')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> {
                                if (i + 4 > s.length) {
                                    throw NativeProtocolException("json", "short \\u escape")
                                }
                                val hex = s.substring(i, i + 4)
                                i += 4
                                val code = hex.toIntOrNull(16)
                                    ?: throw NativeProtocolException("json", "bad \\u escape $hex")
                                sb.append(code.toChar())
                            }
                            else -> throw NativeProtocolException("json", "unknown escape \\$e")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun parseNumber(): JsonValue {
            val start = i
            if (i < s.length && s[i] == '-') i++
            while (i < s.length && isNumberChar(s[i])) i++
            val text = s.substring(start, i)
            if (text.isEmpty()) {
                throw NativeProtocolException("json", "expected a value at offset $start")
            }
            return try {
                if (text.contains('.') || text.contains('e') || text.contains('E')) {
                    JsonValue.Dbl(text.toDouble())
                } else {
                    JsonValue.Int64(text.toLong())
                }
            } catch (e: NumberFormatException) {
                throw NativeProtocolException("json", "bad number \"$text\"")
            }
        }

        private fun isNumberChar(c: Char): Boolean =
            c.isDigit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-'
    }
}

/** Path-aware view over a parsed value; every accessor fails loudly on drift. */
class JsonView internal constructor(val path: String, val value: JsonValue) {

    fun objectValue(): JsonView {
        if (value !is JsonValue.Obj) fail("expected an object, got ${typeName(value)}")
        return this
    }

    fun field(key: String): JsonView {
        val obj = value as? JsonValue.Obj ?: fail("expected an object, got ${typeName(value)}")
        val v = obj.fields[key] ?: fail("missing required field \"$key\"")
        return JsonView("$path.$key", v)
    }

    /** Absent or JSON null both read as absent (the native surfaces use null). */
    fun optionalField(key: String): JsonView? {
        val obj = value as? JsonValue.Obj ?: fail("expected an object, got ${typeName(value)}")
        val v = obj.fields[key] ?: return null
        if (v is JsonValue.Null) return null
        return JsonView("$path.$key", v)
    }

    fun string(): String {
        val v = value as? JsonValue.Str ?: fail("expected a string, got ${typeName(value)}")
        return v.value
    }

    fun long(): Long {
        val v = value as? JsonValue.Int64 ?: fail("expected an integer, got ${typeName(value)}")
        return v.value
    }

    fun int(): Int {
        val v = long()
        if (v < Int.MIN_VALUE || v > Int.MAX_VALUE) fail("integer $v exceeds Int range")
        return v.toInt()
    }

    fun bool(): Boolean {
        val v = value as? JsonValue.Bool ?: fail("expected a boolean, got ${typeName(value)}")
        return v.value
    }

    fun array(): List<JsonView> {
        val v = value as? JsonValue.Arr ?: fail("expected an array, got ${typeName(value)}")
        return v.items.mapIndexed { index, item -> JsonView("$path[$index]", item) }
    }

    fun stringArray(): List<String> = array().map { it.string() }

    fun rawJson(): String = JsonCodec.write(value)

    fun fail(detail: String): Nothing = throw NativeProtocolException(path, detail)

    private fun typeName(v: JsonValue): String = when (v) {
        is JsonValue.Null -> "null"
        is JsonValue.Bool -> "a boolean"
        is JsonValue.Int64 -> "an integer"
        is JsonValue.Dbl -> "a number"
        is JsonValue.Str -> "a string"
        is JsonValue.Arr -> "an array"
        is JsonValue.Obj -> "an object"
    }
}

fun JsonValue.view(path: String): JsonView = JsonView(path, this)

/** Ergonomic object builder for native request bodies. */
class JsonObjectBuilder {
    private val fields = LinkedHashMap<String, JsonValue>()

    fun put(name: String, value: JsonValue?): JsonObjectBuilder {
        if (value != null) fields[name] = value
        return this
    }

    fun put(name: String, value: String?): JsonObjectBuilder =
        put(name, if (value == null) null else JsonValue.Str(value))

    fun put(name: String, value: Long?): JsonObjectBuilder =
        put(name, if (value == null) null else JsonValue.Int64(value))

    fun put(name: String, value: Boolean?): JsonObjectBuilder =
        put(name, if (value == null) null else JsonValue.Bool(value))

    fun putStrings(name: String, values: List<String>?): JsonObjectBuilder =
        put(name, if (values == null) null else JsonValue.Arr(values.map { JsonValue.Str(it) }))

    fun build(): JsonValue.Obj = JsonValue.Obj(fields)

    fun toJson(): String = JsonCodec.write(build())
}

// --------------------------------------------------------------- native DTOs

data class NativeHealth(val ok: Boolean, val version: String)

data class NativeReady(val ready: Boolean)

data class NativeSessionCreated(val id: String, val title: String, val createdMs: Long)

data class NativeSessionSummary(
    val id: String,
    val title: String,
    val provider: String,
    val model: String,
    val state: String
)

data class NativeModelInfo(
    val provider: String,
    val model: String,
    val context: Long,
    val maxOutput: Long,
    val tools: Boolean,
    val reasoning: Boolean,
    val source: String
)

data class NativePromptReceipt(val opId: String, val accepted: Boolean, val queued: Boolean)

data class NativeAbortAck(val aborted: List<String>)

data class NativeActiveModel(val provider: String, val model: String, val variant: String?)

data class NativeActiveTool(val tool: String, val opId: String, val startedMs: Long, val status: String)

data class NativeVerificationOwed(
    val opId: String,
    val tool: String,
    val startedMs: Long,
    val status: String,
    val effectStatus: String?
)

data class NativeVerificationFact(val id: String, val detail: String, val status: String)

data class NativeVerificationView(
    val owed: List<NativeVerificationOwed>,
    val failedChecks: List<NativeVerificationFact>
)

data class NativeProjection(
    val sessionId: String,
    val title: String,
    val provider: String,
    val model: String,
    val lifecycle: String,
    val machine: String,
    val label: String,
    val active: Boolean,
    val terminal: Boolean,
    val activeModel: NativeActiveModel?,
    val activeTool: NativeActiveTool?,
    val filesChanged: List<String>,
    val verification: List<NativeVerificationOwed>,
    val queued: Int
)

data class NativeTaskBudget(
    val maxTokens: Long?,
    val spentTokens: Long?,
    val maxCostMicro: Long?,
    val spentCostMicro: Long,
    val openReservedMicro: Long
)

data class NativeMilestones(val completed: List<String>, val open: List<String>)

data class NativeTaskView(
    val goal: String,
    val state: String,
    val milestones: NativeMilestones,
    val changedFiles: List<String>,
    val testsRun: List<String>,
    val testsFailed: List<String>,
    val budget: NativeTaskBudget?
)

data class NativeTaskRun(
    val taskId: Long,
    val runId: String,
    val mode: String,
    val state: String,
    val goal: String?,
    val itemIds: List<String>,
    val model: String?
)

data class NativeTaskRunStarted(val taskId: Long, val runId: String, val state: String)

data class NativeTaskRunCancelled(val runId: String, val cancelled: Boolean)

data class NativeAgent(
    val agentId: String,
    val kind: String,
    val runId: String,
    val sessionId: Long,
    val worktreeId: Long,
    val goal: String,
    val state: String,
    val model: String?,
    val budget: Long?,
    val ownership: String,
    val itemIds: List<String>
)

data class NativeAgentControlAck(val queuedSeq: Long?, val applied: Boolean?)

data class NativeUsageTotals(
    val sessions: Long,
    val budget: Long,
    val spent: Long,
    val sessionsWithCalls: Long,
    val durableTokens: Long,
    val settledCostMicro: Long
)

data class NativeSessionTaskUsage(val taskId: String, val budget: NativeTaskBudget)

data class NativeSessionUsage(
    val sessionId: String,
    val tokens: Long,
    val tasks: List<NativeSessionTaskUsage>
)

data class NativeVerificationRecord(
    val recordId: String,
    val status: String,
    val criteriaPassed: Int,
    val criteriaTotal: Int,
    val checks: List<String>,
    val changedFiles: List<String>
)

data class NativeTaskVerification(
    val sessionId: String,
    val taskId: String,
    val records: List<NativeVerificationRecord>
)

data class NativeEvidence(
    val id: Long,
    val sessionId: Long,
    val workspaceId: Long,
    val backingRetained: Boolean,
    val backingLen: Long?,
    val allowRanges: Boolean,
    val allowSearch: Boolean,
    val maxBytes: Long
)

/** One retrieved evidence slice; [bytes] are the decoded payload (bounded). */
class NativeEvidenceRetrieval(
    val id: Long,
    val selectorJson: String,
    val bytes: ByteArray,
    val byteLen: Long,
    val truncatedByPolicy: Boolean
)

data class NativeMessage(
    val seq: Long,
    val id: Long,
    val role: String,
    val createdMs: Long,
    val text: String
)

data class NativeMessagePage(
    val messages: List<NativeMessage>,
    val hasMore: Boolean,
    val nextBefore: Long?
)

data class NativeEventRow(
    val seq: Long,
    val kind: String,
    val state: String,
    val opId: String?,
    val tsMs: Long
)

data class NativeEventPage(
    val events: List<NativeEventRow>,
    val hasMore: Boolean,
    val nextCursor: Long?
)

// ----------------------------------------------------------------- parsers

private fun missing(path: String, detail: String): Nothing =
    throw NativeProtocolException(path, detail)

fun parseNativeHealth(json: String): NativeHealth {
    val v = JsonCodec.parse(json).view("GET /native/health")
    return NativeHealth(ok = v.field("ok").bool(), version = v.field("version").string())
}

fun parseNativeReady(json: String): NativeReady {
    val v = JsonCodec.parse(json).view("GET /native/ready")
    return NativeReady(ready = v.field("ready").bool())
}

fun parseNativeSessionCreated(json: String): NativeSessionCreated {
    val v = JsonCodec.parse(json).view("POST /session/create")
    return NativeSessionCreated(
        id = v.field("id").string(),
        title = v.field("title").string(),
        createdMs = v.field("created_ms").long()
    )
}

fun parseNativeSessionList(json: String): List<NativeSessionSummary> {
    val v = JsonCodec.parse(json).view("GET /session/list")
    val sessions = v.field("sessions").array()
    return sessions.map {
        NativeSessionSummary(
            id = it.field("id").string(),
            title = it.field("title").string(),
            provider = it.field("provider").string(),
            model = it.field("model").string(),
            state = it.field("state").string()
        )
    }
}

fun parseNativeModelCatalog(json: String): List<NativeModelInfo> {
    val v = JsonCodec.parse(json).view("GET /models")
    return v.array().map {
        NativeModelInfo(
            provider = it.field("provider").string(),
            model = it.field("model").string(),
            context = it.field("context").long(),
            maxOutput = it.field("maxOutput").long(),
            tools = it.field("tools").bool(),
            reasoning = it.field("reasoning").bool(),
            source = it.field("source").string()
        )
    }
}

fun parseNativePromptReceipt(json: String): NativePromptReceipt {
    val v = JsonCodec.parse(json).view("POST /session/prompt")
    return NativePromptReceipt(
        opId = v.field("op_id").string(),
        accepted = v.field("accepted").bool(),
        queued = v.field("queued").bool()
    )
}

fun parseNativeAbortAck(json: String): NativeAbortAck {
    val v = JsonCodec.parse(json).view("POST /native/session/{id}/abort")
    return NativeAbortAck(aborted = v.field("aborted").stringArray())
}

private fun parseActiveModel(v: JsonView): NativeActiveModel = NativeActiveModel(
    provider = v.field("provider").string(),
    model = v.field("model").string(),
    variant = v.optionalField("variant")?.string()
)

private fun parseActiveTool(v: JsonView): NativeActiveTool = NativeActiveTool(
    tool = v.field("tool").string(),
    opId = v.field("opId").string(),
    startedMs = v.field("startedMs").long(),
    status = v.field("status").string()
)

// The projection's inline verification entries carry no `status` (the
// dedicated /verification route does); both are represented by this DTO.
private fun parseOwed(v: JsonView): NativeVerificationOwed = NativeVerificationOwed(
    opId = v.field("opId").string(),
    tool = v.field("tool").string(),
    startedMs = v.field("startedMs").long(),
    status = v.optionalField("status")?.string() ?: "",
    effectStatus = v.optionalField("effectStatus")?.string()
)

private fun parseVerificationFact(v: JsonView): NativeVerificationFact = NativeVerificationFact(
    id = v.field("id").string(),
    detail = v.field("detail").string(),
    status = v.field("status").string()
)

fun parseNativeProjection(json: String): NativeProjection {
    val v = JsonCodec.parse(json).view("GET /session/{id}/projection")
    val session = v.field("session")
    val state = v.field("state")
    val activeModel = v.optionalField("activeModel")
    val activeTool = v.optionalField("activeTool")
    return NativeProjection(
        sessionId = session.field("id").string(),
        title = session.field("title").string(),
        provider = session.field("provider").string(),
        model = session.field("model").string(),
        lifecycle = session.field("lifecycle").string(),
        machine = state.field("machine").string(),
        label = state.field("label").string(),
        active = state.field("active").bool(),
        terminal = state.field("terminal").bool(),
        activeModel = if (activeModel == null) null else parseActiveModel(activeModel),
        activeTool = if (activeTool == null) null else parseActiveTool(activeTool),
        filesChanged = v.field("filesChanged").stringArray(),
        verification = v.field("verification").array().map { parseOwed(it) },
        queued = v.field("queued").int()
    )
}

private fun parseBudget(v: JsonView): NativeTaskBudget = NativeTaskBudget(
    maxTokens = v.optionalField("maxTokens")?.long(),
    spentTokens = v.optionalField("spentTokens")?.long(),
    maxCostMicro = v.optionalField("maxCostMicro")?.long(),
    spentCostMicro = v.field("spentCostMicro").long(),
    openReservedMicro = v.field("openReservedMicro").long()
)

fun parseNativeTaskViews(json: String): List<NativeTaskView> {
    val v = JsonCodec.parse(json).view("GET /native/session/{id}/tasks")
    return v.array().map {
        val milestones = it.field("milestones")
        val tests = it.field("tests")
        val budget = it.optionalField("budget")
        NativeTaskView(
            goal = it.field("goal").string(),
            state = it.field("state").string(),
            milestones = NativeMilestones(
                completed = milestones.field("completed").stringArray(),
                open = milestones.field("open").stringArray()
            ),
            changedFiles = it.field("changedFiles").stringArray(),
            testsRun = tests.field("run").stringArray(),
            testsFailed = tests.field("failed").stringArray(),
            budget = if (budget == null) null else parseBudget(budget)
        )
    }
}

fun parseNativeTaskRuns(json: String): List<NativeTaskRun> {
    val v = JsonCodec.parse(json).view("GET /native/session/{id}/task-runs")
    return v.array().map {
        NativeTaskRun(
            taskId = it.field("task_id").long(),
            runId = it.field("run_id").string(),
            mode = it.field("mode").string(),
            state = it.field("state").string(),
            goal = it.optionalField("goal")?.string(),
            itemIds = it.optionalField("item_ids")?.stringArray() ?: emptyList(),
            model = it.optionalField("model")?.string()
        )
    }
}

fun parseNativeTaskRunStarted(json: String): NativeTaskRunStarted {
    val v = JsonCodec.parse(json).view("POST /native/session/{id}/task-runs")
    return NativeTaskRunStarted(
        taskId = v.field("task_id").long(),
        runId = v.field("run_id").string(),
        state = v.field("state").string()
    )
}

fun parseNativeTaskRunCancelled(json: String): NativeTaskRunCancelled {
    val v = JsonCodec.parse(json).view("POST /native/session/{id}/task-runs/{run_id}/cancel")
    return NativeTaskRunCancelled(
        runId = v.field("run_id").string(),
        cancelled = v.field("cancelled").bool()
    )
}

fun parseNativeAgents(json: String): List<NativeAgent> {
    val v = JsonCodec.parse(json).view("GET /native/agents")
    return v.array().map {
        NativeAgent(
            agentId = it.field("agent_id").string(),
            kind = it.field("kind").string(),
            runId = it.field("run_id").string(),
            sessionId = it.field("session_id").long(),
            worktreeId = it.field("worktree_id").long(),
            goal = it.field("goal").string(),
            state = it.field("state").string(),
            model = it.optionalField("model")?.string(),
            budget = it.optionalField("budget")?.long(),
            ownership = it.field("ownership").string(),
            itemIds = it.optionalField("item_ids")?.stringArray() ?: emptyList()
        )
    }
}

fun parseNativeAgentControlAck(json: String): NativeAgentControlAck {
    val v = JsonCodec.parse(json).view("POST /native/agents/{child_id}/control")
    return NativeAgentControlAck(
        queuedSeq = v.optionalField("queuedSeq")?.long(),
        applied = v.optionalField("applied")?.bool()
    )
}

fun parseNativeUsage(json: String): NativeUsageTotals {
    val v = JsonCodec.parse(json).view("GET /native/usage")
    val totals = v.field("totals")
    val durable = v.field("durable")
    val calls = durable.field("providerCalls")
    val taskSpend = durable.field("taskSpend")
    return NativeUsageTotals(
        sessions = v.field("sessions").long(),
        budget = totals.field("budget").long(),
        spent = totals.field("spent").long(),
        sessionsWithCalls = durable.field("sessionsWithCalls").long(),
        durableTokens = calls.field("tokens").long(),
        settledCostMicro = taskSpend.field("settledCostMicro").long()
    )
}

fun parseNativeSessionUsage(json: String): NativeSessionUsage {
    val v = JsonCodec.parse(json).view("GET /native/session/{id}/usage")
    val calls = v.field("providerCalls")
    return NativeSessionUsage(
        sessionId = v.field("sessionId").string(),
        tokens = calls.field("tokens").long(),
        tasks = v.field("tasks").array().map {
            NativeSessionTaskUsage(
                taskId = it.field("taskId").string(),
                budget = parseBudget(it.field("budget"))
            )
        }
    )
}

fun parseNativeVerificationView(json: String): NativeVerificationView {
    val v = JsonCodec.parse(json).view("GET /native/session/{id}/verification")
    return NativeVerificationView(
        owed = v.field("owed").array().map { parseOwed(it) },
        failedChecks = v.field("failedChecks").array().map { parseVerificationFact(it) }
    )
}

fun parseNativeTaskVerification(json: String): NativeTaskVerification {
    val v = JsonCodec.parse(json).view("GET /native/session/{id}/tasks/{task_id}/verification")
    return NativeTaskVerification(
        sessionId = v.field("sessionId").string(),
        taskId = v.field("taskId").string(),
        records = v.field("records").array().map { record ->
            val criteria = record.field("criteria").array()
            NativeVerificationRecord(
                recordId = record.field("recordId").string(),
                status = record.field("status").string(),
                criteriaPassed = criteria.count { it.field("passed").bool() },
                criteriaTotal = criteria.size,
                checks = record.field("checks").array().map {
                    it.field("check").string() + "=" + it.field("status").string()
                },
                changedFiles = record.field("changedFiles").array().map {
                    it.field("path").string()
                }
            )
        }
    )
}

fun parseNativeEvidence(json: String): NativeEvidence {
    val v = JsonCodec.parse(json).view("GET /native/evidence/{id}")
    return NativeEvidence(
        id = v.field("id").long(),
        sessionId = v.field("sessionId").long(),
        workspaceId = v.field("workspaceId").long(),
        backingRetained = v.field("backingRetained").bool(),
        backingLen = v.optionalField("backingLen")?.long(),
        allowRanges = v.field("allowRanges").bool(),
        allowSearch = v.field("allowSearch").bool(),
        maxBytes = v.field("maxBytes").long()
    )
}

fun parseNativeEvidenceRetrieval(json: String): NativeEvidenceRetrieval {
    val v = JsonCodec.parse(json).view("POST /native/evidence/{id}/retrieve")
    val base64 = v.field("bytesBase64").string()
    val bytes = try {
        Base64.getDecoder().decode(base64)
    } catch (e: IllegalArgumentException) {
        missing("POST /native/evidence/{id}/retrieve.bytesBase64", "not valid base64")
    }
    return NativeEvidenceRetrieval(
        id = v.field("id").long(),
        selectorJson = v.field("selector").rawJson(),
        bytes = bytes,
        byteLen = v.field("byteLen").long(),
        truncatedByPolicy = v.field("truncatedByPolicy").bool()
    )
}

private fun messageText(message: JsonView): String {
    val rendered = StringBuilder()
    for (part in message.field("parts").array()) {
        val partData = part.optionalField("data") ?: continue
        val text = partData.optionalField("text") ?: continue
        if (text.value is JsonValue.Str) {
            if (rendered.isNotEmpty()) rendered.append('\n')
            val kind = part.optionalField("kind")?.string()
            if (kind != null && kind != "text") rendered.append('[').append(kind).append("] ")
            rendered.append(text.string())
        }
    }
    if (rendered.isNotEmpty()) return rendered.toString()
    val data = message.optionalField("data") ?: return ""
    val text = data.optionalField("text")
    return if (text != null && text.value is JsonValue.Str) text.string() else data.rawJson()
}

fun parseNativeMessagePage(json: String): NativeMessagePage {
    val v = JsonCodec.parse(json).view("GET /native/messages")
    return NativeMessagePage(
        messages = v.field("messages").array().map {
            NativeMessage(
                seq = it.field("seq").long(),
                id = it.field("id").long(),
                role = it.field("role").string(),
                createdMs = it.field("createdMs").long(),
                text = messageText(it)
            )
        },
        hasMore = v.field("hasMore").bool(),
        nextBefore = v.optionalField("nextBefore")?.long()
    )
}

fun parseNativeEventPage(json: String): NativeEventPage {
    val v = JsonCodec.parse(json).view("GET /native/events")
    return NativeEventPage(
        events = v.field("events").array().map {
            NativeEventRow(
                seq = it.field("seq").long(),
                kind = it.field("kind").string(),
                state = it.field("state").string(),
                opId = it.optionalField("opId")?.string(),
                tsMs = it.field("tsMs").long()
            )
        },
        hasMore = v.field("hasMore").bool(),
        nextCursor = v.optionalField("nextCursor")?.long()
    )
}

// --------------------------------------------------------------- requests

/** Request bodies for the native surface, shaped field-for-field on the daemon strict DTOs. */
object NativeRequests {

    fun createSession(
        provider: String,
        model: String,
        workspace: String? = null,
        title: String? = null
    ): String = JsonObjectBuilder()
        .put("provider", provider)
        .put("model", model)
        .put("workspace", workspace)
        .put("title", title)
        .toJson()

    fun prompt(sessionId: String, prompt: String, files: List<String>? = null): String {
        val builder = JsonObjectBuilder()
            .put("session_id", sessionId)
            .put("prompt", prompt)
        if (files != null && files.isNotEmpty()) builder.putStrings("files", files)
        return builder.toJson()
    }

    fun abort(sessionId: String, opId: String? = null): String = JsonObjectBuilder()
        .put("session_id", sessionId)
        .put("op_id", opId)
        .toJson()

    fun startTaskRun(
        goal: String,
        criteria: List<String>? = null,
        model: String? = null,
        maxTokens: Long? = null,
        maxCostMicro: Long? = null,
        mutationMode: String? = null
    ): String = JsonObjectBuilder()
        .put("goal", goal)
        .putStrings("criteria", criteria)
        .put("model", model)
        .put("max_tokens", maxTokens)
        .put("max_cost_micro", maxCostMicro)
        .put("mutation_mode", mutationMode)
        .toJson()

    fun steer(text: String): String = JsonObjectBuilder().put("text", text).toJson()

    fun changeModel(model: String): String = JsonObjectBuilder().put("model", model).toJson()

    fun changeBudget(maxTokens: Long? = null, maxCostMicro: Long? = null): String =
        JsonObjectBuilder()
            .put("max_tokens", maxTokens)
            .put("max_cost_micro", maxCostMicro)
            .toJson()

    fun evidenceSelectorAll(): String =
        JsonObjectBuilder().put("selector", "all").toJson()

    fun evidenceSelectorRange(start: Long, end: Long): String =
        JsonObjectBuilder()
            .put("selector", "byte_range")
            .put("start", start)
            .put("end", end)
            .toJson()

    fun evidenceSelectorSearch(query: String, maxHits: Long): String =
        JsonObjectBuilder()
            .put("selector", "search")
            .put("query", query)
            .put("max_hits", maxHits)
            .toJson()
}
