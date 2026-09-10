//! Golden wire fixtures: the exact JSON shapes this crate produces and
//! accepts for the official ACP v1 surface it implements. Tests compare
//! wire frames against these semantically (key order irrelevant) and, for
//! canonical serialization, byte-for-byte (serde_json serializes maps with
//! sorted keys, so encoding a fixture reproduces the produced frame
//! exactly).
//!
//! Field names follow the official Agent Client Protocol v1 schema
//! (`agentclientprotocol/agent-client-protocol`): `initialize` carries
//! `protocolVersion: 1` plus optional `clientCapabilities`/`clientInfo`,
//! `session/new` answers `{sessionId}`, `session/prompt` takes
//! `{sessionId, prompt: [content blocks]}` and answers `{stopReason}`,
//! `session/update` notifications carry `{sessionId, update}` with update
//! kinds such as `agent_message_chunk` (`{sessionUpdate, content}`), and
//! errors are `{code, message}` with `data` omitted when absent.

/// Official-shape `initialize` request (protocolVersion 1, plus tolerated
/// `clientCapabilities` and `clientInfo`).
pub const INITIALIZE_REQUEST_V1: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{"fs":{"readTextFile":true,"writeTextFile":true}},"clientInfo":{"name":"test-client","version":"1.0.0"}}}"#;

/// Official-shape `initialize` request with an unsupported major version.
pub const INITIALIZE_REQUEST_V2: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":2,"clientCapabilities":{}}}"#;

/// Legacy string-version `initialize` (the crate's own pre-conformance
/// wire): must be rejected loudly, never silently accepted.
pub const INITIALIZE_REQUEST_LEGACY_STRING: &str =
    r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"0.1.0"}}"#;

/// Official `initialize` response: `{protocolVersion, agentCapabilities,
/// authMethods}` — no fields this agent cannot populate honestly
/// (`loadSession: false`, no image/audio/embeddedContext prompts).
pub const INITIALIZE_RESPONSE: &str = r#"{"jsonrpc":"2.0","id":1,"result":{"agentCapabilities":{"loadSession":false,"promptCapabilities":{"audio":false,"embeddedContext":false,"image":false}},"authMethods":[],"protocolVersion":1}}"#;

/// Typed version-mismatch error for `protocolVersion: 2`.
pub const INITIALIZE_ERROR_V2: &str = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"data":{"protocolVersion":2,"supportedProtocolVersion":1},"message":"unsupported protocol version; this agent supports protocol version 1"}}"#;

/// Typed version-mismatch error for the legacy string `"0.1.0"`.
pub const INITIALIZE_ERROR_LEGACY_STRING: &str = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"data":{"protocolVersion":"0.1.0","supportedProtocolVersion":1},"message":"unsupported protocol version; this agent supports protocol version 1"}}"#;

/// Official-shape `session/new` request.
pub const SESSION_NEW_REQUEST: &str =
    r#"{"jsonrpc":"2.0","id":2,"method":"session/new","params":{}}"#;

/// `session/new` response: `{sessionId}`.
pub const SESSION_NEW_RESPONSE: &str =
    r#"{"jsonrpc":"2.0","id":2,"result":{"sessionId":"sess-1"}}"#;

/// Official-shape `session/prompt` request with a text content block.
pub const PROMPT_REQUEST: &str = r#"{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":"sess-1","prompt":[{"type":"text","text":"Hello agent"}]}}"#;

/// `session/prompt` response after a successful turn: official
/// `stopReason`, with the sync seam's opaque backend report under the
/// official `_meta` extension member.
pub const PROMPT_RESPONSE_END_TURN: &str =
    r#"{"jsonrpc":"2.0","id":3,"result":{"_meta":{"echo":"Hello agent"},"stopReason":"end_turn"}}"#;

/// `session/prompt` response after a cancelled turn (the official
/// cancelled state; no `_meta`, the run did not complete).
pub const PROMPT_RESPONSE_CANCELLED: &str =
    r#"{"jsonrpc":"2.0","id":3,"result":{"stopReason":"cancelled"}}"#;

/// Official-shape `session/cancel` notification (no id — cancel is a
/// notification in ACP v1).
pub const CANCEL_NOTIFICATION: &str =
    r#"{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"sess-1"}}"#;

/// Id-bearing `session/cancel` request (tolerated extension form).
pub const CANCEL_REQUEST: &str =
    r#"{"jsonrpc":"2.0","id":5,"method":"session/cancel","params":{"sessionId":"sess-1"}}"#;

/// Cancel acknowledgement: an empty result (the ack carries no outcome —
/// the turn's `stopReason` response is the outcome signal).
pub const CANCEL_ACK_RESPONSE: &str = r#"{"jsonrpc":"2.0","id":5,"result":{}}"#;

/// `session/update` notification: busy status frame
/// (`agentStateChanged`/`agentState.status`). Notifications omit `id`
/// entirely (the official SDK classifies an id-bearing frame as a request).
pub const UPDATE_FRAME_STATE_BUSY: &str = r#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"sess-1","update":{"agentState":{"status":"busy"},"kind":"agentStateChanged"}}}"#;

/// `session/update` notification: busy status frame with an optional
/// `message`.
pub const UPDATE_FRAME_STATE_BUSY_MESSAGE: &str = r#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"sess-1","update":{"agentState":{"message":"thinking hard","status":"busy"},"kind":"agentStateChanged"}}}"#;

/// `session/update` notification: idle status frame.
pub const UPDATE_FRAME_STATE_IDLE: &str = r#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"sess-1","update":{"agentState":{"status":"idle"},"kind":"agentStateChanged"}}}"#;

/// `session/update` notification: error status frame with a message.
pub const UPDATE_FRAME_STATE_ERROR: &str = r#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"sess-1","update":{"agentState":{"message":"backend exploded","status":"error"},"kind":"agentStateChanged"}}}"#;

/// `session/update` notification: official `agent_message_chunk` text
/// content frame.
pub const UPDATE_FRAME_TEXT_CHUNK: &str = r#"{"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"sess-1","update":{"content":{"text":"partial","type":"text"},"sessionUpdate":"agent_message_chunk"}}}"#;

/// Unknown-method error (official message, no `data`).
pub const ERROR_METHOD_NOT_FOUND: &str =
    r#"{"jsonrpc":"2.0","id":9,"error":{"code":-32601,"message":"Method not found"}}"#;

/// Parse-error frame for unparseable JSON bodies (null id).
pub const ERROR_PARSE: &str =
    r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Parse error"}}"#;

/// Internal-error frame: `-32603` with the backend message in `data` (the
/// official `into_internal_error` convention).
pub const ERROR_INTERNAL_BACKEND: &str = r#"{"jsonrpc":"2.0","id":3,"error":{"code":-32603,"data":"backend exploded","message":"Internal error"}}"#;

/// Invalid-params error for a prompt whose sessionId is missing.
pub const ERROR_INVALID_PARAMS_SESSION: &str = r#"{"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"missing string field \"sessionId\""}}"#;

/// Busy error when a session already has a running and a queued prompt.
pub const ERROR_SESSION_BUSY: &str = r#"{"jsonrpc":"2.0","id":12,"error":{"code":-32001,"message":"A prompt turn is already in progress for this session"}}"#;

// ---------------------------------------------------------------------------
// Wave: advertised-surface mapping (session/load, permissions, tool calls,
// plans, MCP, client fs, authenticate)
// ---------------------------------------------------------------------------

/// Official `initialize` request carrying an extension declaration (one
/// accepted, one unknown name) and negotiated client filesystem
/// capabilities.
pub const INITIALIZE_REQUEST_EXTENSIONS: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"extensions":["faktor.agentStateChanged","unknown.extension"],"clientCapabilities":{"fs":{"readTextFile":true,"writeTextFile":false}}}}"#;

/// `initialize` response for a backend reporting load+MCP capabilities and
/// a client that declared the Faktor status extension: only the accepted
/// extension subset is echoed.
pub const INITIALIZE_RESPONSE_CAPABLE: &str = r#"{"jsonrpc":"2.0","id":1,"result":{"agentCapabilities":{"loadSession":true,"mcpCapabilities":{"http":true,"sse":false},"promptCapabilities":{"audio":false,"embeddedContext":false,"image":false}},"authMethods":[],"extensions":["faktor.agentStateChanged"],"protocolVersion":1}}"#;

/// Malformed extension declaration: refused loudly, never silently ignored.
pub const INITIALIZE_ERROR_BAD_EXTENSIONS: &str = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"\"extensions\" must be an array of strings"}}"#;

/// Malformed client fs capability: refused loudly.
pub const INITIALIZE_ERROR_BAD_FS: &str = r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"\"clientCapabilities.fs.readTextFile\" must be a boolean"}}"#;

/// Official-shape `session/load` request (history is replayed through
/// `session/update` notifications; the response itself is the empty
/// official object).
pub const LOAD_REQUEST: &str = r#"{"jsonrpc":"2.0","id":4,"method":"session/load","params":{"sessionId":"sess-1","cwd":"/work","mcpServers":[]}}"#;

/// `session/load` response after a bounded replay: official empty result.
pub const LOAD_RESPONSE: &str = r#"{"jsonrpc":"2.0","id":4,"result":{}}"#;

/// Foreign/unknown session refusal (official invalid params, never an
/// empty replay).
pub const ERROR_LOAD_FOREIGN_SESSION: &str =
    r#"{"jsonrpc":"2.0","id":4,"error":{"code":-32602,"message":"unknown session \"foreign\""}}"#;

/// Incomplete-history refusal: the bounded load window cannot represent
/// the whole conversation, so nothing is replayed.
pub const ERROR_LOAD_INCOMPLETE_HISTORY: &str = r#"{"jsonrpc":"2.0","id":4,"error":{"code":-32603,"data":"session history exceeds the bounded load window (older messages exist)","message":"Internal error"}}"#;

/// MCP servers refused while the backend reports no MCP capability.
pub const ERROR_MCP_UNSUPPORTED: &str = r#"{"jsonrpc":"2.0","id":2,"error":{"code":-32602,"message":"this agent does not support MCP servers"}}"#;

/// `authenticate` refusal while `authMethods` is empty.
pub const ERROR_AUTHENTICATE: &str = r#"{"jsonrpc":"2.0","id":7,"error":{"code":-32602,"data":{"methodId":"api-key"},"message":"no authentication methods are available"}}"#;

/// First server→client permission request on a fresh connection (server id
/// allocation starts at 1).
pub const PERMISSION_REQUEST_FRAME: &str = r#"{"jsonrpc":"2.0","id":1,"method":"session/request_permission","params":{"sessionId":"sess-1","toolCall":{"toolCallId":"call-1","title":"echo"},"options":[{"optionId":"allow_once","name":"Allow once","kind":"allow_once"},{"optionId":"reject_once","name":"Reject once","kind":"reject_once"}]}}"#;

/// Client answer selecting the allow option.
pub const PERMISSION_ALLOW_RESPONSE: &str = r#"{"jsonrpc":"2.0","id":1,"result":{"outcome":{"outcome":"selected","optionId":"allow_once"}}}"#;

/// Client answer selecting the reject option.
pub const PERMISSION_DENY_RESPONSE: &str = r#"{"jsonrpc":"2.0","id":1,"result":{"outcome":{"outcome":"selected","optionId":"reject_once"}}}"#;

/// Client answer cancelling the permission request.
pub const PERMISSION_CANCELLED_RESPONSE: &str =
    r#"{"jsonrpc":"2.0","id":1,"result":{"outcome":{"outcome":"cancelled"}}}"#;

/// First server→client `fs/read_text_file` request on a fresh connection.
pub const FS_READ_REQUEST_FRAME: &str = r#"{"jsonrpc":"2.0","id":1,"method":"fs/read_text_file","params":{"sessionId":"sess-1","path":"/work/notes.txt"}}"#;

/// Client `fs/read_text_file` answer.
pub const FS_READ_RESPONSE: &str = r#"{"jsonrpc":"2.0","id":1,"result":{"content":"alpha"}}"#;

/// Official `tool_call` update body from the native tool state `running`.
pub const UPDATE_FRAME_TOOL_CALL: &str = r#"{"rawInput":{"x":1},"sessionUpdate":"tool_call","status":"in_progress","title":"echo","toolCallId":"call-1"}"#;

/// Official `tool_call` body for an unrecognized native state: the
/// optional `status` is omitted (documented degradation) and `kind` is
/// omitted because the native surface has none.
pub const UPDATE_FRAME_TOOL_CALL_DEGRADED: &str =
    r#"{"rawInput":{},"sessionUpdate":"tool_call","title":"echo","toolCallId":"call-1"}"#;

/// Official `tool_call_update` body from a failed native tool result.
pub const UPDATE_FRAME_TOOL_RESULT_FAILED: &str = r#"{"content":[{"content":{"text":"boom","type":"text"},"type":"content"}],"sessionUpdate":"tool_call_update","status":"failed","toolCallId":"call-1"}"#;

/// Official `plan` body with the documented native degradation: ledger
/// steps carry no priority or status, so both are conservative defaults.
pub const UPDATE_FRAME_PLAN: &str = r#"{"entries":[{"content":"step one","priority":"medium","status":"pending"},{"content":"step two","priority":"medium","status":"pending"}],"sessionUpdate":"plan"}"#;
