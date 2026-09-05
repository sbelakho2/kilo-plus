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
/// (`agentStateChanged`/`agentState.status`).
pub const UPDATE_FRAME_STATE_BUSY: &str = r#"{"jsonrpc":"2.0","id":null,"method":"session/update","params":{"sessionId":"sess-1","update":{"agentState":{"status":"busy"},"kind":"agentStateChanged"}}}"#;

/// `session/update` notification: busy status frame with an optional
/// `message`.
pub const UPDATE_FRAME_STATE_BUSY_MESSAGE: &str = r#"{"jsonrpc":"2.0","id":null,"method":"session/update","params":{"sessionId":"sess-1","update":{"agentState":{"message":"thinking hard","status":"busy"},"kind":"agentStateChanged"}}}"#;

/// `session/update` notification: idle status frame.
pub const UPDATE_FRAME_STATE_IDLE: &str = r#"{"jsonrpc":"2.0","id":null,"method":"session/update","params":{"sessionId":"sess-1","update":{"agentState":{"status":"idle"},"kind":"agentStateChanged"}}}"#;

/// `session/update` notification: error status frame with a message.
pub const UPDATE_FRAME_STATE_ERROR: &str = r#"{"jsonrpc":"2.0","id":null,"method":"session/update","params":{"sessionId":"sess-1","update":{"agentState":{"message":"backend exploded","status":"error"},"kind":"agentStateChanged"}}}"#;

/// `session/update` notification: official `agent_message_chunk` text
/// content frame.
pub const UPDATE_FRAME_TEXT_CHUNK: &str = r#"{"jsonrpc":"2.0","id":null,"method":"session/update","params":{"sessionId":"sess-1","update":{"content":{"text":"partial","type":"text"},"sessionUpdate":"agent_message_chunk"}}}"#;

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
