// The frozen v7.5.6 wire contract as plain Kotlin data classes.
//
// Zero external dependencies on purpose: no kotlinx.serialization, no okhttp,
// no org.json. Serialization is hand-built for exactly the two requests the
// JetBrains shell sends (session create, message send); parsing is a tiny
// recursive-descent JSON reader for the daemon responses. The authoritative
// shapes live in crates/protocol/src/v756/wire.rs and the frozen fixtures in
// compat/kilo-v756/ — if a field name changes there, this file must change
// too (wire compatibility is a frozen contract).
package dev.faktor.shared

import java.util.Base64
import java.util.regex.Pattern

/** Thrown when a daemon line or JSON payload violates the frozen wire contract. */
class ProtocolException(message: String) : Exception(message)

/**
 * The frozen daemon stdout line: `faktor server listening on http://127.0.0.1:<port>`.
 * Nothing else is ever printed on stdout; there is no JSON handshake (see
 * compat/kilo-v756/startup_line.json).
 */
data class StartupLine(val port: Int) {
    override fun toString(): String = "faktor server listening on http://127.0.0.1:$port"

    companion object {
        private val PATTERN: Pattern =
            Pattern.compile("^faktor server listening on http://127\\.0\\.0\\.1:(\\d+)$")

        /** Parses one exact startup line; returns null for any other content. */
        fun parse(line: String): StartupLine? {
            val m = PATTERN.matcher(line)
            if (!m.matches()) return null
            return StartupLine(m.group(1).toInt())
        }
    }
}

/**
 * Basic auth for every daemon request (compat/kilo-v756/basic_auth.json):
 * `Authorization: Basic base64("kilo:" + FAKTOR_SERVER_PASSWORD)`. The daemon
 * splits the decoded payload at the FIRST colon; username must be exactly
 * `kilo`; the password is compared constant-time.
 */
class BasicAuth(val password: String) {
    val headerValue: String
        get() = "Basic " + Base64.getEncoder()
            .encodeToString((BasicAuth.USERNAME + ":" + password).toByteArray(Charsets.UTF_8))

    companion object {
        const val HEADER_NAME: String = "Authorization"
        const val USERNAME: String = "kilo"
    }
}

/**
 * Model selection inside `SessionCreateRequest` — wire shape `{id,
 * providerID, variant?}` (crates/protocol/src/v756/wire.rs `SessionModel`).
 */
data class Model(val id: String, val providerID: String, val variant: String? = null) {
    fun toJson(): String {
        val sb = StringBuilder("{\"id\":")
            .append(jsonString(id))
            .append(",\"providerID\":")
            .append(jsonString(providerID))
        if (variant != null) {
            sb.append(",\"variant\":").append(jsonString(variant))
        }
        return sb.append("}").toString()
    }
}

/**
 * Model selection inside `MessageSendRequest` — wire shape `{providerID,
 * modelID}` (wire.rs `MessageModel`). Note this is a DIFFERENT shape from the
 * session-create model (`id` there, `modelID` here); the two are kept as
 * distinct types so the wire is never guessed.
 */
data class MessageModel(val providerID: String, val modelID: String) {
    fun toJson(): String =
        "{\"providerID\":" + jsonString(providerID) + ",\"modelID\":" + jsonString(modelID) + "}"
}

/** `POST /session` body. `model` is optional: the daemon falls back to the default. */
data class SessionCreateRequest(val title: String? = null, val model: Model? = null) {
    fun toJson(): String {
        val fields = StringBuilder()
        if (title != null) {
            fields.append("\"title\":").append(jsonString(title))
        }
        if (model != null) {
            if (fields.isNotEmpty()) fields.append(',')
            fields.append("\"model\":").append(model.toJson())
        }
        return "{" + fields.toString() + "}"
    }
}

/** One part of a message. The discriminator is the `type` field (exact v7.5.6 kinds). */
sealed class Part {
    abstract fun toJson(): String

    /** `{"type":"text","text":...}` */
    data class TextPart(val text: String) : Part() {
        override fun toJson(): String =
            "{\"type\":\"text\",\"text\":" + jsonString(text) + "}"
    }

    /**
     * `{"type":"tool","callID":...,"name":...,"input":...}`.
     * [input] must be valid raw JSON (e.g. `"{}"` or `"{\"path\":\"a.rs\"}"`);
     * it is embedded verbatim so arbitrary tool payloads need no schema.
     */
    data class ToolPart(val callId: String, val name: String, val input: String) : Part() {
        override fun toJson(): String =
            "{\"type\":\"tool\",\"callID\":" + jsonString(callId) +
                ",\"name\":" + jsonString(name) + ",\"input\":" + input + "}"
    }
}

/** `POST /session/{sessionID}/message` body (the full turn). */
data class MessageSendRequest(val model: MessageModel, val parts: List<Part>) {
    fun toJson(): String {
        val sb = StringBuilder("{\"model\":")
            .append(model.toJson())
            .append(",\"parts\":[")
        for ((i, p) in parts.withIndex()) {
            if (i > 0) sb.append(',')
            sb.append(p.toJson())
        }
        return sb.append("]}").toString()
    }
}

/** `GET /global/health` response: `{ok, version, protocol}`. */
data class HealthResult(val ok: Boolean, val version: String, val protocol: String)

/** Extracts `sessionID` from a `POST /session` response. */
fun parseSessionId(json: String): String = MiniJson.requiredString(json, "sessionID")

/** Extracts `messageID` from a `POST /session/{id}/message` response. */
fun parseMessageId(json: String): String = MiniJson.requiredString(json, "messageID")

/** Extracts `state` from a `GET /session/{sessionID}` summary. */
fun parseSessionState(json: String): String = MiniJson.requiredString(json, "state")

/** Counts the `messages` array of a `GET /session/{id}/message` page. */
fun parseMessageCount(json: String): Int {
    val messages = MiniJson.parseObject(json)["messages"]
        ?: throw ProtocolException("missing \"messages\" in message page")
    val list = messages as? List<*> ?: throw ProtocolException("\"messages\" is not an array")
    return list.size
}

/** Parses `GET /global/health` into [HealthResult]. */
fun parseHealth(json: String): HealthResult {
    val obj = MiniJson.parseObject(json)
    return HealthResult(
        ok = obj["ok"] as? Boolean ?: throw ProtocolException("missing \"ok\" in health"),
        version = obj["version"] as? String
            ?: throw ProtocolException("missing \"version\" in health"),
        protocol = obj["protocol"] as? String
            ?: throw ProtocolException("missing \"protocol\" in health")
    )
}

/** Minimal dependency-free JSON string escaping (RFC 8259). */
internal fun jsonString(s: String): String {
    val sb = StringBuilder(s.length + 2)
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
    return sb.append('"').toString()
}

/**
 * Tiny recursive-descent JSON reader (object/array/string/number/bool/null).
 * Enough for the daemon's v7.5.6 responses; rejects malformed input loudly.
 */
internal object MiniJson {
    fun parseObject(text: String): Map<String, Any?> {
        val v = parse(text)
        return v as? Map<String, Any?>
            ?: throw ProtocolException("expected a JSON object, got ${className(v)}")
    }

    fun requiredString(text: String, key: String): String {
        val v = parseObject(text)[key]
            ?: throw ProtocolException("missing \"$key\" in response")
        return v as? String ?: throw ProtocolException("\"$key\" is not a string")
    }

    private fun className(v: Any?): String = if (v == null) "null" else v.javaClass.simpleName

    fun parse(text: String): Any? {
        val p = Parser(text)
        val v = p.parseValue()
        p.skipWs()
        if (!p.atEnd()) throw ProtocolException("trailing characters after JSON value")
        return v
    }

    private class Parser(private val s: String) {
        private var i = 0

        fun atEnd(): Boolean = i >= s.length

        fun skipWs() {
            while (i < s.length && isWs(s[i])) i++
        }

        private fun isWs(c: Char): Boolean =
            c == ' ' || c == '\t' || c == '\n' || c == '\r'

        fun parseValue(): Any? {
            skipWs()
            if (i >= s.length) throw ProtocolException("unexpected end of JSON")
            return when (s[i]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> parseString()
                't' -> {
                    expect("true")
                    true
                }
                'f' -> {
                    expect("false")
                    false
                }
                'n' -> {
                    expect("null")
                    null
                }
                else -> parseNumber()
            }
        }

        private fun expect(word: String) {
            if (!s.startsWith(word, i)) {
                throw ProtocolException("malformed JSON literal at offset $i")
            }
            i += word.length
        }

        private fun parseObject(): Map<String, Any?> {
            i++
            val m = LinkedHashMap<String, Any?>()
            skipWs()
            if (i < s.length && s[i] == '}') {
                i++
                return m
            }
            while (true) {
                skipWs()
                if (i >= s.length || s[i] != '"') {
                    throw ProtocolException("expected string key in object")
                }
                val key = parseString()
                skipWs()
                if (i >= s.length || s[i] != ':') throw ProtocolException("expected ':'")
                i++
                m[key] = parseValue()
                skipWs()
                if (i >= s.length) throw ProtocolException("unterminated object")
                when (s[i]) {
                    ',' -> i++
                    '}' -> {
                        i++
                        return m
                    }
                    else -> throw ProtocolException("expected ',' or '}' in object")
                }
            }
        }

        private fun parseArray(): List<Any?> {
            i++
            val a = ArrayList<Any?>()
            skipWs()
            if (i < s.length && s[i] == ']') {
                i++
                return a
            }
            while (true) {
                a.add(parseValue())
                skipWs()
                if (i >= s.length) throw ProtocolException("unterminated array")
                when (s[i]) {
                    ',' -> i++
                    ']' -> {
                        i++
                        return a
                    }
                    else -> throw ProtocolException("expected ',' or ']' in array")
                }
            }
        }

        private fun parseString(): String {
            i++
            val sb = StringBuilder()
            while (true) {
                if (i >= s.length) throw ProtocolException("unterminated string")
                val c = s[i]
                i++
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        if (i >= s.length) throw ProtocolException("unterminated escape")
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
                                if (i + 4 > s.length) throw ProtocolException("short \\u escape")
                                val hex = s.substring(i, i + 4)
                                i += 4
                                val code = hex.toIntOrNull(16)
                                    ?: throw ProtocolException("bad \\u escape $hex")
                                sb.append(code.toChar())
                            }
                            else -> throw ProtocolException("unknown escape \\$e")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun parseNumber(): Any {
            val start = i
            if (i < s.length && s[i] == '-') i++
            while (i < s.length && isNumberChar(s[i])) i++
            val text = s.substring(start, i)
            return try {
                if (text.contains('.') || text.contains('e') || text.contains('E')) {
                    text.toDouble()
                } else {
                    text.toLong()
                }
            } catch (e: NumberFormatException) {
                throw ProtocolException("bad JSON number \"$text\"")
            }
        }

        private fun isNumberChar(c: Char): Boolean =
            c.isDigit() || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-'
    }
}
