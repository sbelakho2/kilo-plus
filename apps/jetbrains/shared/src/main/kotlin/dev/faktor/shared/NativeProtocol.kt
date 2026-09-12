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

    /** One nested object (strict DTO members only). */
    fun putObject(name: String, block: JsonObjectBuilder.() -> Unit): JsonObjectBuilder =
        put(name, JsonObjectBuilder().apply(block).build())

    /** An array of nested objects (insertion order preserved). */
    fun putObjects(name: String, values: List<JsonObjectBuilder>): JsonObjectBuilder =
        put(name, if (values.isEmpty()) null else JsonValue.Arr(values.map { it.build() }))

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

/** One explicit plan step (additive task-view field; `depends_on` is the DAG edge). */
data class NativePlanStep(
    val id: String,
    val summary: String,
    val state: String,
    val dependsOn: List<String>
)

/** The durable child blocker of the native agent listing (`blocker` field). */
data class NativeBlocker(
    val kind: String,
    val reason: String,
    val dependency: String?,
    val resolution: String?,
    val lastProgressMs: Long?
)

/** The bounded live progress record of one agent (`progress` field). */
data class NativeAgentProgress(
    val lastOutputAt: Long?,
    val lastProgressAt: Long?,
    val lastOpCompletedAt: Long?,
    val inFlightOp: String?,
    val silenceMs: Long?,
    val stallThresholdMs: Long?,
    val stalled: Boolean
)

/** The latest durable merge envelope of one child (`result.merge`). */
data class NativeChildMerge(
    val changeSetId: String?,
    val merged: Long?,
    val rejected: Long?,
    val conflicts: Long?
)

/** The durable latest-result summary of one child (`result` field). */
data class NativeChildResult(val summary: String, val merge: NativeChildMerge?)

data class NativeTaskView(
    val goal: String,
    val state: String,
    val milestones: NativeMilestones,
    val changedFiles: List<String>,
    val testsRun: List<String>,
    val testsFailed: List<String>,
    val budget: NativeTaskBudget?,
    val acceptanceCriteria: List<String> = emptyList(),
    val plan: List<NativePlanStep> = emptyList(),
    val blockers: List<String> = emptyList(),
    val evidenceRefs: List<String> = emptyList(),
    val phase: String? = null,
    val progress: NativeAgentProgress? = null,
    /** Additive durable completion contract + step rows; null when the
     * serving daemon exposes no completion read (never fabricated). */
    val completion: NativeTaskCompletion? = null
)

/** The Task-mode completion contract (wire vocabulary `include_*`). */
data class NativeCompletionContract(
    val includeCommit: Boolean,
    val includePush: Boolean,
    val includePr: Boolean
) {
    /** All-false means today's default path: no contract is sent. */
    val isDefault: Boolean
        get() = !includeCommit && !includePush && !includePr

    /** The requested conditional steps, in gate order. */
    fun requestedSteps(): List<String> {
        val steps = mutableListOf<String>()
        if (includeCommit) steps.add("commit")
        if (includePush) steps.add("push")
        if (includePr) steps.add("pr")
        return steps
    }

    companion object {
        fun of(includeCommit: Boolean, includePush: Boolean, includePr: Boolean) =
            NativeCompletionContract(includeCommit, includePush, includePr)
    }
}

/** One durable completion-step row as served on a task view. */
data class NativeCompletionStepStatus(
    val step: String,
    val status: String,
    val detail: String
)

/** The durable completion surface of one task (contract + step rows). */
data class NativeTaskCompletion(
    val contract: NativeCompletionContract,
    val steps: List<NativeCompletionStepStatus>
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

/**
 * One durable run-family board post (`GET/POST /native/session/{id}/board`).
 * `authorChild` null means the run root posted; `revision` is the per-board
 * monotonic cursor (id == revision).
 */
data class NativeBoardPost(
    val id: Long,
    val boardId: Long,
    val authorChild: Long?,
    val authorSession: Long,
    val subject: String,
    val body: String,
    val refs: List<String>,
    val revision: Long,
    val createdMs: Long
)

/** One bounded newest-first page of the path session's board. */
data class NativeBoardPage(
    val boardId: Long,
    val revision: Long,
    val posts: List<NativeBoardPost>,
    /** Exclusive cursor for the next OLDER page (passed back as `since`). */
    val nextBeforeRevision: Long?,
    val hasMore: Boolean
)

data class NativeAgent(
    val agentId: String,
    val kind: String,
    val runId: String,
    val sessionId: Long,
    val worktreeId: Long,
    val goal: String,
    val state: String,
    val model: String?,
    /** The child session's durable provider (children only; null when a
     * pre-provider daemon serves the entry). The (provider, model) pair is
     * the only safe catalog join key. */
    val provider: String? = null,
    val budget: Long?,
    val ownership: String,
    val itemIds: List<String>,
    val itemId: String? = null,
    val itemKind: String? = null,
    val blocker: NativeBlocker? = null,
    val capabilities: List<String> = emptyList(),
    val progress: NativeAgentProgress? = null,
    val result: NativeChildResult? = null,
    /** Durable presentation/attention state: "foreground" or "background". */
    val presentation: String = "foreground"
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

/** One durable criterion verdict of a verification record. */
data class NativeCriterionVerdict(
    val criterionKey: String,
    val passed: Boolean,
    val evidence: String?
)

data class NativeVerificationRecord(
    val recordId: String,
    val status: String,
    val criteriaPassed: Int,
    val criteriaTotal: Int,
    val checks: List<String>,
    val changedFiles: List<String>,
    val criteria: List<NativeCriterionVerdict> = emptyList(),
    val checkSummaries: List<String> = emptyList()
)

data class NativeTaskVerification(
    val sessionId: String,
    val taskId: String,
    val records: List<NativeVerificationRecord>
)

// ------------------------------------------------ orchestrator graph (native)

data class NativeGraphWorkItem(val itemId: String, val kind: String, val state: String)

data class NativeGraphChild(
    val childId: String,
    val sessionId: Long,
    val operationId: Long,
    val worktreeId: Long,
    val ownership: String,
    val state: String,
    val blocker: NativeBlocker?,
    val budget: Long?,
    val planStepIndex: Int?,
    val merge: NativeChildMerge?
)

data class NativeOrchestratorGraph(
    val planId: String,
    val goal: String,
    val state: String,
    val workItems: List<NativeGraphWorkItem>,
    val children: List<NativeGraphChild>
)

// ------------------------------------------------- tournament (native, v1)

data class NativeTournamentCriterion(val id: String, val spec: String)

/** One tournament candidate: identity, location, verification and review verdicts, measured axes. */
data class NativeTournamentCandidate(
    val childId: String,
    val worktree: String,
    val baseRevision: String,
    val state: String,
    val verification: Long?,
    val verificationPass: Boolean?,
    val reviewRank: String?,
    val reviewer: String?,
    val costMicro: Long,
    val wallMs: Long
)

data class NativeTournament(
    val id: String,
    val runFamily: String,
    val goal: String,
    val criteria: List<NativeTournamentCriterion>,
    val candidates: List<NativeTournamentCandidate>,
    val winner: String?,
    val state: String
)

/** The start receipt of a tournament (`POST /native/session/{id}/tournament`). */
data class NativeTournamentStarted(
    val tournamentId: String,
    val runId: String,
    val candidates: List<String>,
    val state: String,
    val winner: String?
)

/** One listing summary of a durable tournament (`GET .../tournaments`). */
data class NativeTournamentSummary(
    val id: String,
    val state: String,
    val candidateCount: Long,
    val winner: String?,
    val decidedMs: Long?
)

/** One discarded candidate of a decision (`discarded[]`). */
data class NativeTournamentDiscarded(val childId: String, val reason: String)

/** The deterministic decision ack (`POST .../tournaments/{id}/decide`). */
data class NativeTournamentDecision(
    val tournamentId: String,
    val winner: String,
    val rationale: String,
    val discarded: List<NativeTournamentDiscarded>
)

/** One durable presentation transition ack (`changed=false` = idempotent). */
data class NativePresentationAck(
    val childId: String,
    val presentation: String,
    val changed: Boolean
)

// ------------------------------------------------- permissions (SDK reply path)

data class NativePermissionEntry(
    val id: String,
    val sessionId: String,
    val capability: String,
    val detail: String
)

data class NativePermissionAck(val ok: Boolean)

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

private fun parseBlocker(v: JsonView): NativeBlocker = NativeBlocker(
    kind = v.field("kind").string(),
    reason = v.field("reason").string(),
    dependency = v.optionalField("dependency")?.string(),
    resolution = v.optionalField("resolution")?.string(),
    lastProgressMs = v.optionalField("last_progress_ms")?.long()
)

private fun parseAgentProgress(v: JsonView): NativeAgentProgress = NativeAgentProgress(
    lastOutputAt = v.optionalField("lastOutputAt")?.long(),
    lastProgressAt = v.optionalField("lastProgressAt")?.long(),
    lastOpCompletedAt = v.optionalField("lastOpCompletedAt")?.long(),
    inFlightOp = v.optionalField("inFlightOp")?.string(),
    silenceMs = v.optionalField("silenceMs")?.long(),
    stallThresholdMs = v.optionalField("stallThresholdMs")?.long(),
    stalled = v.optionalField("stalled")?.bool() ?: false
)

/** The bounded text of a JSON value: strings verbatim, other shapes as JSON. */
private fun jsonText(v: JsonView): String? {
    val value = v.value
    return if (value is JsonValue.Str) value.value else v.rawJson()
}

private fun parseChildMerge(v: JsonView): NativeChildMerge = NativeChildMerge(
    changeSetId = v.optionalField("change_set_id")?.string(),
    merged = v.optionalField("merged")?.long(),
    rejected = v.optionalField("rejected")?.long(),
    conflicts = v.optionalField("conflicts")?.long()
)

private fun parseChildResult(v: JsonView): NativeChildResult = NativeChildResult(
    summary = v.optionalField("summary")?.string() ?: "",
    merge = v.optionalField("merge")?.let { parseChildMerge(it) }
)

/** First present key wins (additive-field convention: camelCase or snake_case). */
private fun optionalAny(v: JsonView, vararg keys: String): JsonView? {
    for (key in keys) {
        val found = v.optionalField(key)
        if (found != null) return found
    }
    return null
}

private fun optionalStrings(v: JsonView, vararg keys: String): List<String> =
    optionalAny(v, *keys)?.stringArray() ?: emptyList()

private fun parsePlanSteps(v: JsonView): List<NativePlanStep> {
    val plan = optionalAny(v, "plan", "plan_steps", "planSteps") ?: return emptyList()
    return plan.array().map { step ->
        val rawId = step.optionalField("id") ?: step.fail("missing required field \"id\"")
        val id = when (val value = rawId.value) {
            is JsonValue.Str -> value.value
            is JsonValue.Int64 -> value.value.toString()
            else -> rawId.fail("expected a non-empty string or integer id")
        }
        NativePlanStep(
            id = id,
            summary = optionalAny(step, "summary", "title")?.string() ?: "",
            state = optionalAny(step, "state", "status")?.string() ?: "pending",
            dependsOn = optionalStrings(step, "depends_on", "dependsOn")
        )
    }
}

private fun parseTaskBlockers(v: JsonView): List<String> =
    optionalAny(v, "blockers", "blocked_on")?.array()?.map { entry ->
        if (entry.value is JsonValue.Str) {
            entry.string()
        } else {
            val detail = optionalAny(entry, "detail", "message", "summary", "reason")
                ?.let { jsonText(it) }
            detail ?: entry.fail("expected a non-empty blocker detail")
        }
    } ?: emptyList()

/**
 * Strict additive completion parse: when the daemon serves a `completion`
 * block both members are required and every step row carries
 * step/status/detail. A malformed block is a loud protocol failure, never
 * a silently empty (or fabricated) success.
 */
private fun parseTaskCompletion(v: JsonView): NativeTaskCompletion {
    val contract = v.field("contract")
    val steps = v.field("steps").array().map { step ->
        NativeCompletionStepStatus(
            step = step.field("step").string(),
            status = step.field("status").string(),
            detail = step.field("detail").string()
        )
    }
    return NativeTaskCompletion(
        contract = NativeCompletionContract(
            includeCommit = contract.field("include_commit").bool(),
            includePush = contract.field("include_push").bool(),
            includePr = contract.field("include_pr").bool()
        ),
        steps = steps
    )
}

fun parseNativeTaskViews(json: String): List<NativeTaskView> {
    val v = JsonCodec.parse(json).view("GET /native/session/{id}/tasks")
    return v.array().map {
        val milestones = it.field("milestones")
        val tests = it.field("tests")
        val budget = it.optionalField("budget")
        val progress = it.optionalField("progress")
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
            budget = if (budget == null) null else parseBudget(budget),
            acceptanceCriteria = optionalStrings(it, "acceptanceCriteria", "acceptance_criteria"),
            plan = parsePlanSteps(it),
            blockers = parseTaskBlockers(it),
            evidenceRefs = optionalStrings(it, "evidenceRefs", "evidence_refs"),
            phase = it.optionalField("phase")?.string(),
            progress = progress?.let { p -> parseAgentProgress(p) },
            completion = it.optionalField("completion")?.let { c -> parseTaskCompletion(c) }
        )
    }
}

private fun parseBoardPost(v: JsonView): NativeBoardPost = NativeBoardPost(
    id = v.field("id").long(),
    boardId = v.field("board_id").long(),
    authorChild = v.optionalField("author_child")?.long(),
    authorSession = v.field("author_session").long(),
    subject = v.field("subject").string(),
    body = v.field("body").string(),
    refs = v.field("refs").stringArray(),
    revision = v.field("revision").long(),
    createdMs = v.field("created_ms").long()
)

/** Strict parse of one board page (`GET /native/session/{id}/board`). */
fun parseNativeBoardPage(json: String): NativeBoardPage {
    val v = JsonCodec.parse(json).view("GET /native/session/{id}/board")
    return NativeBoardPage(
        boardId = v.field("board_id").long(),
        revision = v.field("revision").long(),
        posts = v.field("posts").array().map { parseBoardPost(it) },
        nextBeforeRevision = v.optionalField("next_before_revision")?.long(),
        hasMore = v.field("has_more").bool()
    )
}

/** Strict parse of one board post response (`POST .../board`, 201). */
fun parseNativeBoardPost(json: String): NativeBoardPost =
    parseBoardPost(JsonCodec.parse(json).view("POST /native/session/{id}/board"))

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
        val blocker = it.optionalField("blocker")
        val capabilities = it.optionalField("capabilities")?.array()
            ?.map { entry -> entry.rawJson() } ?: emptyList()
        val progress = it.optionalField("progress")
        val result = it.optionalField("result")
        NativeAgent(
            agentId = it.field("agent_id").string(),
            kind = it.field("kind").string(),
            runId = it.field("run_id").string(),
            sessionId = it.field("session_id").long(),
            worktreeId = it.field("worktree_id").long(),
            goal = it.field("goal").string(),
            state = it.field("state").string(),
            model = it.optionalField("model")?.string(),
            provider = it.optionalField("provider")?.string(),
            budget = it.optionalField("budget")?.long(),
            ownership = it.field("ownership").string(),
            itemIds = it.optionalField("item_ids")?.stringArray() ?: emptyList(),
            itemId = it.optionalField("item_id")?.string(),
            itemKind = it.optionalField("item_kind")?.string(),
            blocker = blocker?.let { b -> parseBlocker(b) },
            capabilities = capabilities,
            progress = progress?.let { p -> parseAgentProgress(p) },
            result = result?.let { r -> parseChildResult(r) },
            presentation = it.optionalField("presentation")?.string() ?: "foreground"
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
                },
                criteria = criteria.map { criterion ->
                    NativeCriterionVerdict(
                        criterionKey = criterion.field("criterionKey").string(),
                        passed = criterion.field("passed").bool(),
                        evidence = criterion.optionalField("evidence")?.let { e -> jsonText(e) }
                    )
                },
                checkSummaries = record.field("checks").array()
                    .mapNotNull { check -> check.optionalField("summary")?.let { s -> jsonText(s) } }
            )
        }
    )
}

// ---------------------------------------------------- orchestrator graph parse

fun parseNativeOrchestratorGraph(json: String): NativeOrchestratorGraph {
    val v = JsonCodec.parse(json).view("GET /native/orchestrator/graph")
    return NativeOrchestratorGraph(
        planId = v.field("plan_id").string(),
        goal = v.field("goal").string(),
        state = v.field("state").string(),
        workItems = v.field("work_items").array().map {
            NativeGraphWorkItem(
                itemId = it.field("item_id").string(),
                kind = it.field("kind").string(),
                state = it.field("state").string()
            )
        },
        children = v.field("children").array().map {
            val blocker = it.optionalField("blocker")
            val merge = it.optionalField("merge")
            NativeGraphChild(
                childId = it.field("child_id").string(),
                sessionId = it.field("session_id").long(),
                operationId = it.field("operation_id").long(),
                worktreeId = it.field("worktree_id").long(),
                ownership = it.field("ownership").string(),
                state = it.field("state").string(),
                blocker = blocker?.let { b -> parseBlocker(b) },
                budget = it.optionalField("budget")?.long(),
                planStepIndex = it.optionalField("plan_step_index")?.int(),
                merge = if (merge == null) null else parseChildMerge(merge)
            )
        }
    )
}

// ----------------------------------------------------------- tournament parse

fun parseNativeTournament(json: String): NativeTournament {
    val v = JsonCodec.parse(json).view("GET /native/session/{id}/tournament/{tournament_id}")
    return NativeTournament(
        id = v.field("id").string(),
        runFamily = v.field("run_family").string(),
        goal = v.field("goal").string(),
        criteria = v.field("criteria").array().map {
            NativeTournamentCriterion(
                id = it.field("id").string(),
                spec = it.field("spec").string()
            )
        },
        candidates = v.field("candidates").array().map { candidate ->
            val review = candidate.optionalField("review")
            NativeTournamentCandidate(
                childId = candidate.field("child_id").string(),
                worktree = candidate.field("worktree").string(),
                baseRevision = candidate.field("base_revision").string(),
                state = candidate.field("state").string(),
                verification = candidate.optionalField("verification")?.long(),
                verificationPass = candidate.optionalField("verification_pass")?.bool(),
                reviewRank = review?.field("rank")?.string(),
                reviewer = review?.field("reviewer")?.string(),
                costMicro = candidate.field("cost_micro").long(),
                wallMs = candidate.field("wall_ms").long()
            )
        },
        winner = v.optionalField("winner")?.string(),
        state = v.field("state").string()
    )
}

fun parseNativeTournamentStarted(json: String): NativeTournamentStarted {
    val v = JsonCodec.parse(json).view("POST /native/session/{id}/tournament")
    return NativeTournamentStarted(
        tournamentId = v.field("tournament_id").string(),
        runId = v.field("run_id").string(),
        candidates = v.field("candidates").stringArray(),
        state = v.field("state").string(),
        winner = v.optionalField("winner")?.string()
    )
}

fun parseNativeTournamentSummaries(json: String): List<NativeTournamentSummary> {
    val v = JsonCodec.parse(json).view("GET /native/session/{id}/tournaments")
    return v.array().map {
        NativeTournamentSummary(
            id = it.field("id").string(),
            state = it.field("state").string(),
            candidateCount = it.field("candidate_count").long(),
            winner = it.optionalField("winner")?.string(),
            decidedMs = it.optionalField("decided_ms")?.long()
        )
    }
}

fun parseNativeTournamentDecision(json: String): NativeTournamentDecision {
    val v = JsonCodec.parse(json).view("POST /native/session/{id}/tournaments/{id}/decide")
    return NativeTournamentDecision(
        tournamentId = v.field("tournament_id").string(),
        winner = v.field("winner").string(),
        rationale = v.field("rationale").string(),
        discarded = v.field("discarded").array().map {
            NativeTournamentDiscarded(
                childId = it.field("child_id").string(),
                reason = it.field("reason").string()
            )
        }
    )
}

fun parseNativePresentationAck(json: String): NativePresentationAck {
    val v = JsonCodec.parse(json).view("POST /native/session/{id}/agents/{child}/presentation")
    return NativePresentationAck(
        childId = v.field("child_id").string(),
        presentation = v.field("presentation").string(),
        changed = v.field("changed").bool()
    )
}

// ----------------------------------------------------------- permission parse

fun parseNativePermissionList(json: String): List<NativePermissionEntry> {
    val v = JsonCodec.parse(json).view("GET /permission/list")
    return v.field("permissions").array().map {
        NativePermissionEntry(
            id = it.field("id").string(),
            sessionId = it.field("session_id").string(),
            capability = it.field("capability").string(),
            detail = it.optionalField("detail")?.let { d -> jsonText(d) } ?: ""
        )
    }
}

fun parseNativePermissionAck(json: String): NativePermissionAck =
    NativePermissionAck(JsonCodec.parse(json).view("POST /permission/reply").field("ok").bool())

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
        mutationMode: String? = null,
        files: List<String>? = null,
        completionContract: NativeCompletionContract? = null
    ): String {
        val builder = JsonObjectBuilder()
            .put("goal", goal)
            .putStrings("criteria", criteria)
            .put("model", model)
            .put("max_tokens", maxTokens)
            .put("max_cost_micro", maxCostMicro)
            .put("mutation_mode", mutationMode)
            .putStrings("files", if (files.isNullOrEmpty()) null else files)
        // A non-default completion contract requires explicit work items (the
        // daemon refuses it on the plain-prompt path). ONE mutating `main`
        // item keeps the same in-session drive with the durable contract seam;
        // the default path stays byte-identical.
        val contract = completionContract?.takeIf { !it.isDefault }
        if (contract != null) {
            builder.putObjects(
                "work_items",
                listOf(
                    JsonObjectBuilder()
                        .put("id", "main")
                        .put("kind", "Implementation")
                        .put("summary", goal)
                        .put("ownership", JsonValue.Str("isolated_worktree"))
                )
            )
            builder.putObject("completion_contract") {
                put("include_commit", contract.includeCommit)
                put("include_push", contract.includePush)
                put("include_pr", contract.includePr)
            }
        }
        return builder.toJson()
    }

    fun startTournament(
        goal: String,
        criteria: List<String>,
        n: Int,
        model: String? = null,
        maxTokens: Long? = null,
        maxCostMicro: Long? = null,
        mutationMode: String? = null,
        files: List<String>? = null
    ): String = JsonObjectBuilder()
        .put("goal", goal)
        .putStrings("criteria", criteria)
        .put("n", n.toLong())
        .put("model", model)
        .put("max_tokens", maxTokens)
        .put("max_cost_micro", maxCostMicro)
        .put("mutation_mode", mutationMode)
        .putStrings("files", if (files.isNullOrEmpty()) null else files)
        .toJson()

    fun steer(text: String): String = JsonObjectBuilder().put("text", text).toJson()

    /** The strict decide body: no operator input, exactly `{}`. */
    fun decideTournament(): String = JsonObjectBuilder().toJson()

    fun abortTournament(reason: String?): String =
        JsonObjectBuilder().put("reason", reason?.takeIf { it.isNotEmpty() }).toJson()

    fun changePresentation(presentation: String): String =
        JsonObjectBuilder().put("state", presentation).toJson()

    /** One bounded board post body; the daemon re-validates every bound. */
    fun boardPost(subject: String, body: String, refs: List<String>? = null): String {
        val builder = JsonObjectBuilder().put("subject", subject).put("body", body)
        if (!refs.isNullOrEmpty()) builder.putStrings("refs", refs)
        return builder.toJson()
    }

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

    fun evidenceSelectorLines(start: Long, end: Long): String =
        JsonObjectBuilder()
            .put("selector", "line_range")
            .put("start", start)
            .put("end", end)
            .toJson()

    fun evidenceSelectorSearch(query: String, maxHits: Long): String =
        JsonObjectBuilder()
            .put("selector", "search")
            .put("query", query)
            .put("max_hits", maxHits)
            .toJson()

    fun permissionReply(permissionId: String, decision: String): String =
        JsonObjectBuilder()
            .put("permission_id", permissionId)
            .put("decision", decision)
            .toJson()
}
