//! ACP wire protocol: JSON-RPC 2.0 over newline-delimited JSON (NDJSON, the
//! official ACP v1 transport) with a dual-mode reader that also accepts the
//! legacy `Content-Length` framing this crate shipped before conformance.
//! Helpers written from scratch, not shared code.
//!
//! # Framing (official transport first, legacy accepted)
//!
//! The official ACP v1 transport is one JSON message per line: the official
//! SDK's `ByteStreams` reader splits on `\n` and its writer terminates every
//! message with `\n`. [`Framing::Ndjson`] is therefore the primary mode.
//! Legacy peers that open with a `Content-Length: N\r\n\r\n` header are
//! detected and served with the same framing they used
//! ([`Framing::ContentLength`]); responses always follow the framing detected
//! from the peer's first bytes, so neither client sees a mixed stream.
//!
//! Detection is [`detect_framing`]: the first non-whitespace byte `{` (or `[`)
//! selects NDJSON, anything else selects Content-Length. [`parse_ndjson`]
//! accepts `\r\n` line endings and skips blank lines, and bounds a single
//! line at [`MAX_FRAME_BYTES`] exactly like a declared frame body.
//!
//! # Method surface
//!
//! This crate exposes a SUBSET of the ACP method space. The exact
//! on-the-wire strings are frozen:
//!
//! | variant            | wire string     | role                    |
//! |--------------------|-----------------|-------------------------|
//! | [`AcpMethod::Initialize`]  | `initialize`   | handshake lifecycle     |
//! | [`AcpMethod::AgentInfo`]   | `agent_info`   | agent metadata (extension) |
//! | [`AcpMethod::SessionNew`]  | `session/new`  | session creation        |
//! | [`AcpMethod::SessionLoad`] | `session/load` | bounded native history replay |
//! | [`AcpMethod::SessionPrompt`]| `session/prompt` | run one prompt turn   |
//! | [`AcpMethod::SessionCancel`]| `session/cancel` | cancel the active turn |
//! | [`AcpMethod::SessionAbort`] | `session/abort` | DEPRECATED alias of cancel |
//! | (no variant) | `$/cancel_request` | request-level cancel of the matching running turn (handled by the server reader) |
//! | [`AcpMethod::SessionList`] | `session/list` | session inventory (extension) |
//! | [`AcpMethod::Authenticate`]| `authenticate` | refused (no auth flow exists) |
//! | [`AcpMethod::RequestPermission`] | `session/request_permission` | agent→client permission round trip |
//! | [`AcpMethod::FsReadTextFile`] | `fs/read_text_file` | agent→client bounded read |
//! | [`AcpMethod::FsWriteTextFile`] | `fs/write_text_file` | agent→client write |
//! | [`AcpMethod::Shutdown`]    | `shutdown`     | lifecycle end (extension) |
//!
//! # Bounds (bounded everything)
//!
//! - A declared `Content-Length` larger than [`MAX_FRAME_BYTES`] (16 MiB) is
//!   rejected up front: hostile headers fail without buffering the body.
//! - Headers must terminate within [`MAX_HEADER_BYTES`].
//! - Request bodies that parse are additionally capped by the server at
//!   [`crate::MAX_PARAMS_BYTES`]; backend results are capped at
//!   [`crate::MAX_RESPONSE_BYTES`].
//!
//! # Framing rules
//!
//! - Frame terminator is exactly `\r\n\r\n`; header lines may use either
//!   CRLF or LF internally. Unknown headers are ignored; a
//!   `Content-Length` header is mandatory and may not repeat.
//! - `parse_frame` is a pure function over an accumulated byte buffer: it
//!   returns `None` while the current frame is incomplete, so callers can
//!   feed arbitrarily fragmented reads.
//! - JSON-RPC notifications carry NO `id` member at all: the official SDK
//!   classifies any message with an `id` (even `null`) as a request and would
//!   never deliver a `session/update` to its notification handler. Absent
//!   `id` is the only emitted form; `id: null` is still accepted on parse.
//!   Error responses to unparseable input carry a null `id`, per JSON-RPC
//!   2.0 §5.1.

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};

/// Hard cap on a single declared frame body (16 MiB). Bigger declarations
/// are a hostile-header error before any body is buffered.
pub const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// A frame header must be fully received within this many bytes or the
/// connection is refused as unframed garbage.
pub const MAX_HEADER_BYTES: usize = 16 * 1024;

const FRAME_TERMINATOR: &[u8] = b"\r\n\r\n";
const MAX_HEADER_LINES: usize = 64;

/// Per-connection wire framing. The server detects this from the peer's
/// first non-whitespace byte ([`detect_framing`]) and answers in the same
/// mode, so an official (NDJSON) client and a legacy (`Content-Length`)
/// client each see a consistent stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// Official ACP v1 transport: one JSON message per `\n`-terminated line.
    Ndjson,
    /// Legacy LSP/MCP-style `Content-Length: N\r\n\r\n<body>` framing.
    ContentLength,
}

/// Decide the connection framing from the bytes received so far.
///
/// Returns `None` while every byte is ASCII whitespace. The first
/// non-whitespace byte decides: `{` or `[` (a JSON message or batch) selects
/// [`Framing::Ndjson`]; anything else is treated as the legacy
/// `Content-Length` header path, whose parser refuses non-framed garbage
/// with a typed error.
pub fn detect_framing(bytes: &[u8]) -> Option<Framing> {
    for byte in bytes {
        if byte.is_ascii_whitespace() {
            continue;
        }
        return Some(match byte {
            b'{' | b'[' => Framing::Ndjson,
            _ => Framing::ContentLength,
        });
    }
    None
}

/// The wire method strings, as enumerated above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcpMethod {
    Initialize,
    Shutdown,
    SessionNew,
    /// Official `session/load` (capability-gated; bounded replay).
    SessionLoad,
    SessionPrompt,
    /// Official ACP v1 cancel method.
    SessionCancel,
    /// Deprecated alias of [`AcpMethod::SessionCancel`]; accepted on the
    /// wire but never advertised.
    SessionAbort,
    SessionList,
    AgentInfo,
    /// Official `authenticate` (refused while `authMethods` is empty).
    Authenticate,
    /// Official agent→client `session/request_permission`.
    RequestPermission,
    /// Official agent→client `fs/read_text_file`.
    FsReadTextFile,
    /// Official agent→client `fs/write_text_file`.
    FsWriteTextFile,
}

impl AcpMethod {
    /// Exact on-the-wire method string for this variant.
    pub fn as_str(self) -> &'static str {
        match self {
            AcpMethod::Initialize => "initialize",
            AcpMethod::Shutdown => "shutdown",
            AcpMethod::SessionNew => "session/new",
            AcpMethod::SessionLoad => "session/load",
            AcpMethod::SessionPrompt => "session/prompt",
            AcpMethod::SessionCancel => "session/cancel",
            AcpMethod::SessionAbort => "session/abort",
            AcpMethod::SessionList => "session/list",
            AcpMethod::AgentInfo => "agent_info",
            AcpMethod::Authenticate => "authenticate",
            AcpMethod::RequestPermission => "session/request_permission",
            AcpMethod::FsReadTextFile => "fs/read_text_file",
            AcpMethod::FsWriteTextFile => "fs/write_text_file",
        }
    }

    /// Inverse of [`AcpMethod::as_str`].
    pub fn from_wire_str(s: &str) -> Option<AcpMethod> {
        Some(match s {
            "initialize" => AcpMethod::Initialize,
            "shutdown" => AcpMethod::Shutdown,
            "session/new" => AcpMethod::SessionNew,
            "session/load" => AcpMethod::SessionLoad,
            "session/prompt" => AcpMethod::SessionPrompt,
            "session/cancel" => AcpMethod::SessionCancel,
            // Deprecated pre-conformance alias of session/cancel.
            "session/abort" => AcpMethod::SessionAbort,
            "session/list" => AcpMethod::SessionList,
            "agent_info" => AcpMethod::AgentInfo,
            "authenticate" => AcpMethod::Authenticate,
            "session/request_permission" => AcpMethod::RequestPermission,
            "fs/read_text_file" => AcpMethod::FsReadTextFile,
            "fs/write_text_file" => AcpMethod::FsWriteTextFile,
            _ => return None,
        })
    }
}

impl std::str::FromStr for AcpMethod {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        AcpMethod::from_wire_str(s).ok_or(())
    }
}

/// A JSON-RPC request with a numeric id (never a notification).
#[derive(Debug, Clone, PartialEq)]
pub struct AcpRequest {
    pub id: u64,
    pub method: String,
    pub params: serde_json::Value,
}

impl AcpRequest {
    pub fn new(id: u64, method: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            id,
            method: method.into(),
            params,
        }
    }

    /// Full wire object: `{"jsonrpc":"2.0","id":..,"method":..,"params":..}`.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.id,
            "method": self.method,
            "params": self.params,
        })
    }
}

impl Serialize for AcpRequest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("AcpRequest", 4)?;
        s.serialize_field("jsonrpc", "2.0")?;
        s.serialize_field("id", &self.id)?;
        s.serialize_field("method", &self.method)?;
        s.serialize_field("params", &self.params)?;
        s.end()
    }
}

/// A JSON-RPC response: exactly one of `result`/`error` is `Some`.
#[derive(Debug, Clone, PartialEq)]
pub struct AcpResponse {
    pub id: u64,
    pub result: Option<serde_json::Value>,
    pub error: Option<serde_json::Value>,
}

impl AcpResponse {
    pub fn result(id: u64, result: serde_json::Value) -> Self {
        Self {
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Build an error response from a JSON-RPC error object
    /// (`{"code":..,"message":..}`).
    pub fn error(id: u64, error: serde_json::Value) -> Self {
        Self {
            id,
            result: None,
            error: Some(error),
        }
    }

    /// Shortcut: error object built from a code + message.
    pub fn error_code(id: u64, code: i64, message: impl Into<String>) -> Self {
        Self::error(
            id,
            serde_json::json!({ "code": code, "message": message.into() }),
        )
    }

    /// Full wire object: `{"jsonrpc":"2.0","id":..,"result":..|"error":..}`.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.id,
            "result": self.result,
            "error": self.error,
        })
    }
}

impl Serialize for AcpResponse {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("AcpResponse", 3)?;
        s.serialize_field("jsonrpc", "2.0")?;
        s.serialize_field("id", &self.id)?;
        match (&self.result, &self.error) {
            (Some(result), None) => s.serialize_field("result", result)?,
            (None, Some(error)) => s.serialize_field("error", error)?,
            _ => {
                return Err(serde::ser::Error::custom(
                    "AcpResponse must carry exactly one of result or error",
                ))
            }
        }
        s.end()
    }
}

/// Frame-level parse outcome: distinguishes unrecoverable framing violations
/// from recoverable content errors whose byte boundary is still known.
#[derive(Debug)]
pub(crate) struct FrameError {
    pub(crate) message: String,
    /// `true`: the stream is desynced or hostile; the connection must end.
    /// `false`: the `consumed` bytes are discarded and framing continues.
    pub(crate) fatal: bool,
    /// Leading bytes to discard on recoverable errors (header end or full
    /// declared body end).
    pub(crate) consumed: usize,
}

impl FrameError {
    fn recoverable(message: impl Into<String>, consumed: usize) -> Self {
        Self {
            message: message.into(),
            fatal: false,
            consumed,
        }
    }

    fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: true,
            consumed: 0,
        }
    }
}

/// Incremental Content-Length frame parser over an accumulated buffer.
///
/// - `Ok(None)`: buffer does not yet hold a complete frame (keep feeding).
/// - `Ok(Some((consumed, value)))`: one complete frame; `consumed` is the
///   number of leading bytes it occupied — leftover bytes belong to the
///   next frame.
/// - `Err`: the buffer is unrecoverably unframed or hostile (bad header,
///   missing/duplicate/invalid `Content-Length`, declared size beyond
///   [`MAX_FRAME_BYTES`], invalid JSON body).
pub fn parse_frame(bytes: &[u8]) -> Result<Option<(usize, serde_json::Value)>, String> {
    parse_frame_detailed(bytes).map_err(|e| e.message)
}

/// Like [`parse_frame`] but with recovery metadata for the serve loop.
pub(crate) fn parse_frame_detailed(
    bytes: &[u8],
) -> Result<Option<(usize, serde_json::Value)>, FrameError> {
    let header_end = match find_terminator(bytes) {
        Some(pos) => pos + FRAME_TERMINATOR.len(),
        None => {
            if bytes.len() > MAX_HEADER_BYTES {
                return Err(FrameError::fatal(format!(
                    "no header terminator within {MAX_HEADER_BYTES} bytes; stream is not Content-Length framed"
                )));
            }
            return Ok(None);
        }
    };
    let header_bytes = &bytes[..header_end];

    let mut content_length: Option<u64> = None;
    let mut lines = 0usize;
    for raw_line in header_bytes.split(|b| *b == b'\r' || *b == b'\n') {
        let line = raw_line.trim_ascii();
        if line.is_empty() {
            continue;
        }
        lines += 1;
        if lines > MAX_HEADER_LINES {
            return Err(FrameError::recoverable(
                format!("more than {MAX_HEADER_LINES} header lines"),
                header_end,
            ));
        }
        let colon = line.iter().position(|b| *b == b':').ok_or_else(|| {
            FrameError::recoverable(
                format!("malformed header line: {:?}", String::from_utf8_lossy(line)),
                header_end,
            )
        })?;
        let (name, value) = (&line[..colon], line[colon + 1..].trim_ascii());
        if name.eq_ignore_ascii_case(&b"content-length"[..]) {
            if content_length.is_some() {
                return Err(FrameError::recoverable(
                    "duplicate Content-Length header",
                    header_end,
                ));
            }
            let text = std::str::from_utf8(value)
                .map_err(|_| FrameError::recoverable("Content-Length is not ASCII", header_end))?;
            let n: u64 = text.trim().parse().map_err(|_| {
                FrameError::recoverable(format!("invalid Content-Length: {text:?}"), header_end)
            })?;
            content_length = Some(n);
        }
        // Unknown headers are tolerated and ignored (LSP/MCP convention).
    }

    let declared = content_length
        .ok_or_else(|| FrameError::recoverable("missing Content-Length header", header_end))?;
    if declared > MAX_FRAME_BYTES as u64 {
        // Hostile declaration: refusing the connection is the only bounded
        // answer (the declared body may never actually arrive).
        return Err(FrameError::fatal(format!(
            "declared Content-Length {declared} exceeds the 16 MiB frame bound"
        )));
    }
    let body_start = header_end;
    let end = body_start
        .checked_add(declared as usize)
        .ok_or_else(|| FrameError::fatal("Content-Length overflows usize".to_string()))?;
    if bytes.len() < end {
        return Ok(None);
    }
    let body = &bytes[body_start..end];
    let value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| FrameError::recoverable(format!("invalid JSON body: {e}"), end))?;
    Ok(Some((end, value)))
}

/// Incremental NDJSON parser over an accumulated buffer.
///
/// - `Ok(None)`: no complete line yet (keep feeding); a buffer with no
///   newline beyond [`MAX_FRAME_BYTES`] is a fatal hostile-line error instead
///   of unbounded buffering.
/// - `Ok(Some((consumed, None)))`: one whitespace-only line; the caller
///   discards `consumed` bytes and keeps parsing (blank lines are legal
///   keep-alives, never messages, and draining them keeps a blank-line flood
///   bounded).
/// - `Ok(Some((consumed, Some(value))))`: one complete JSON message.
/// - `Err`: an oversized line (fatal) or an invalid JSON line
///   (recoverable at the line boundary).
pub(crate) fn parse_ndjson_detailed(
    bytes: &[u8],
) -> Result<Option<(usize, Option<serde_json::Value>)>, FrameError> {
    let Some(newline) = bytes.iter().position(|b| *b == b'\n') else {
        if bytes.len() > MAX_FRAME_BYTES {
            return Err(FrameError::fatal(format!(
                "no newline within {MAX_FRAME_BYTES} bytes; stream is not newline-delimited JSON"
            )));
        }
        return Ok(None);
    };
    let consumed = newline + 1;
    let mut body_end = newline;
    if body_end > 0 && bytes[body_end - 1] == b'\r' {
        body_end -= 1;
    }
    let line = &bytes[..body_end];
    if line.iter().all(u8::is_ascii_whitespace) {
        // Returned per line so the caller drains it: a blank-line flood is
        // consumed read-by-read instead of accumulating.
        return Ok(Some((consumed, None)));
    }
    let value: serde_json::Value = serde_json::from_slice(line)
        .map_err(|e| FrameError::recoverable(format!("invalid JSON line: {e}"), consumed))?;
    Ok(Some((consumed, Some(value))))
}

/// Parse one complete NDJSON message, skipping blank lines. `Ok(None)` means
/// the buffer does not yet hold a complete line; `consumed` includes the
/// blank lines and the terminating newline of the returned message.
pub fn parse_ndjson(bytes: &[u8]) -> Result<Option<(usize, serde_json::Value)>, String> {
    let mut offset = 0usize;
    loop {
        match parse_ndjson_detailed(&bytes[offset..]) {
            Ok(None) => return Ok(None),
            Ok(Some((consumed, None))) => {
                offset += consumed;
                continue;
            }
            Ok(Some((consumed, Some(value)))) => {
                return Ok(Some((offset + consumed, value)));
            }
            Err(e) => return Err(e.message),
        }
    }
}

fn find_terminator(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < FRAME_TERMINATOR.len() {
        return None;
    }
    bytes
        .windows(FRAME_TERMINATOR.len())
        .position(|w| w == FRAME_TERMINATOR)
}

/// Serialize any value into one unframed JSON body (no header, no newline).
pub fn encode_body(value: &serde_json::Value) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|e| format!("encode failed: {e}"))
}

/// Serialize any value into one complete legacy `Content-Length` framed
/// message (header + body).
pub fn encode(value: &serde_json::Value) -> Result<Vec<u8>, String> {
    let body = encode_body(value)?;
    let mut out = Vec::with_capacity(body.len() + 64);
    out.extend_from_slice(format!("Content-Length: {}\r\n\r\n", body.len()).as_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Serialize any value into one complete NDJSON line (body + `\n`).
pub fn encode_line(value: &serde_json::Value) -> Result<Vec<u8>, String> {
    let mut body = encode_body(value)?;
    body.push(b'\n');
    Ok(body)
}

/// Encode a request/response with a numeric id.
pub fn frame(method: String, id: u64, params: serde_json::Value) -> Vec<u8> {
    // Serialization of a well-formed AcpRequest cannot fail.
    encode(&AcpRequest::new(id, method, params).to_json()).expect("request frame encodes")
}

/// Encode a server→client notification into a legacy `Content-Length`
/// frame. JSON-RPC notifications MUST omit `id`: the official SDK treats any
/// message carrying an `id` member (even `null`) as a request and would drop
/// the notification.
pub fn notification_frame(method: String, params: serde_json::Value) -> Vec<u8> {
    encode(&notification_body(method, params)).expect("notification frame encodes")
}

/// The JSON-RPC notification object with the `id` member omitted.
pub fn notification_body(method: String, params: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn json(s: &str) -> serde_json::Value {
        serde_json::from_str(s).unwrap()
    }

    fn complete_frame(body: &[u8]) -> Vec<u8> {
        let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn empty_and_partial_are_incomplete() {
        assert_eq!(parse_frame(b"").unwrap(), None);
        assert_eq!(parse_frame(b"Content-Length: 5\r\n").unwrap(), None);
        assert_eq!(parse_frame(b"Content-Length: 5\r\n\r\nhel").unwrap(), None);
        // Header exactly at the cap, body still pending: incomplete, not error.
        let header = "Content-Length: 100000\r\n\r\n".to_string();
        assert_eq!(parse_frame(header.as_bytes()).unwrap(), None);
    }

    #[test]
    fn hostile_content_length_rejected_without_body() {
        // 20 MiB declaration > 16 MiB cap: error immediately, no buffering.
        let header = format!("Content-Length: {}\r\n\r\n", 20 * 1024 * 1024);
        let err = parse_frame(header.as_bytes()).unwrap_err();
        assert!(err.contains("exceeds"), "{err}");
        // Overflowing and negative declarations are invalid, not huge.
        assert!(parse_frame(b"Content-Length: 99999999999999999999999\r\n\r\n").is_err());
        assert!(parse_frame(b"Content-Length: -5\r\n\r\n").is_err());
        assert!(parse_frame(b"Content-Length: abc\r\n\r\n").is_err());
        assert!(parse_frame(b"Content-Length: \r\n\r\n").is_err());
    }

    #[test]
    fn unframed_garbage_beyond_header_cap_is_error() {
        let junk = vec![b'x'; MAX_HEADER_BYTES + 1];
        let err = parse_frame(&junk).unwrap_err();
        assert!(err.contains("header terminator"), "{err}");
        let junk_below = vec![b'x'; MAX_HEADER_BYTES - 1];
        assert_eq!(parse_frame(&junk_below).unwrap(), None);
    }

    #[test]
    fn missing_or_duplicate_content_length_rejected() {
        let err = parse_frame(b"X-Powered-By: acp\r\n\r\n{}").unwrap_err();
        assert!(err.contains("missing Content-Length"), "{err}");
        let dup = b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}";
        assert!(parse_frame(dup).unwrap_err().contains("duplicate"));
        let garbage_line = b"not-a-header\r\n\r\n{}";
        assert!(parse_frame(garbage_line).is_err());
    }

    #[test]
    fn header_case_whitespace_extra_headers_tolerated() {
        let frame = complete_frame(br#"{"ok":1}"#);
        let mut tolerant = b"content-length: 8\r\nX-Ignored: yep\r\n\r\n".to_vec();
        tolerant.extend_from_slice(br#"{"ok":1}"#);
        let (consumed, value) = parse_frame(&tolerant).unwrap().unwrap();
        assert_eq!(consumed, tolerant.len());
        assert_eq!(value, json(r#"{"ok":1}"#));
        assert_eq!(parse_frame(&frame).unwrap().unwrap().0, frame.len());
    }

    #[test]
    fn malformed_json_body_is_error_but_consumed() {
        let frame = complete_frame(b"{\"broken");
        let err = parse_frame(&frame).unwrap_err();
        assert!(err.contains("invalid JSON body"), "{err}");
        // Body bytes are still accounted; trailing valid frame parses after
        // the caller drops the consumed prefix.
        let mut both = frame;
        both.extend_from_slice(&complete_frame(b"{}"));
        let first = parse_frame(&both).unwrap_err();
        assert!(first.contains("invalid JSON body"));
    }

    #[test]
    fn complete_frame_with_trailing_bytes_reports_consumed_offset() {
        let a = complete_frame(br#"{"a":1}"#);
        let b = complete_frame(br#"{"b":2}"#);
        let mut both = a.clone();
        both.extend_from_slice(&b);
        let (consumed, value) = parse_frame(&both).unwrap().unwrap();
        assert_eq!(consumed, a.len());
        assert_eq!(value, json(r#"{"a":1}"#));
        let rest = &both[consumed..];
        let (consumed2, value2) = parse_frame(rest).unwrap().unwrap();
        assert_eq!(consumed2, rest.len());
        assert_eq!(value2, json(r#"{"b":2}"#));
    }

    #[test]
    fn fragmentation_equivalent_to_single_write() {
        let whole = complete_frame(br#"{"jsonrpc":"2.0","id":7,"method":"session/prompt","params":{"sessionID":"s","text":"hi"}}"#);
        let (_, expect) = parse_frame(&whole).unwrap().unwrap();
        let mut feed = Vec::new();
        for (i, byte) in whole.iter().enumerate() {
            feed.push(*byte);
            if i % 7 == 3 {
                // At arbitrary boundaries the parser must simply say "wait".
                assert_eq!(parse_frame(&feed).unwrap(), None);
            }
        }
        let (consumed, got) = parse_frame(&feed).unwrap().unwrap();
        assert_eq!(consumed, feed.len());
        assert_eq!(got, expect);
    }

    #[test]
    fn json_body_with_embedded_crlf_crlf_is_fine() {
        // Raw control bytes are illegal inside JSON strings, but \r\n\r\n
        // may appear as whitespace BETWEEN tokens; it is inside the declared
        // body so the parser must not treat it as a frame boundary.
        let body = b"{\"a\" : [1,\r\n\r\n2] }";
        let frame = complete_frame(body);
        let (consumed, value) = parse_frame(&frame).unwrap().unwrap();
        assert_eq!(consumed, frame.len());
        assert_eq!(value, json(r#"{"a":[1,2]}"#));
    }

    #[test]
    fn unframed_partial_then_garbage_after_read_chunks() {
        // Fragmented reads must accumulate identically to one big read even
        // when a chunk boundary splits the header terminator itself.
        let frame = complete_frame(b"{}");
        for split in 0..frame.len() {
            let mut a = frame[..split].to_vec();
            let b = frame[split..].to_vec();
            assert_eq!(parse_frame(&a).unwrap(), None);
            a.extend_from_slice(&b);
            assert_eq!(parse_frame(&a).unwrap().unwrap().1, json("{}"));
        }
    }

    #[test]
    fn acp_method_strings_round_trip() {
        let all = [
            AcpMethod::Initialize,
            AcpMethod::Shutdown,
            AcpMethod::SessionNew,
            AcpMethod::SessionLoad,
            AcpMethod::SessionPrompt,
            AcpMethod::SessionCancel,
            AcpMethod::SessionAbort,
            AcpMethod::SessionList,
            AcpMethod::AgentInfo,
            AcpMethod::Authenticate,
            AcpMethod::RequestPermission,
            AcpMethod::FsReadTextFile,
            AcpMethod::FsWriteTextFile,
        ];
        for m in all {
            assert_eq!(m.as_str().parse::<AcpMethod>(), Ok(m));
        }
        assert_eq!(
            "session/cancel".parse::<AcpMethod>(),
            Ok(AcpMethod::SessionCancel)
        );
        assert_eq!(
            "session/abort".parse::<AcpMethod>(),
            Ok(AcpMethod::SessionAbort)
        );
        assert_eq!(
            "session/request_permission".parse::<AcpMethod>(),
            Ok(AcpMethod::RequestPermission)
        );
        assert_eq!("session/close".parse::<AcpMethod>(), Err(()));
    }

    #[test]
    fn wire_serialization_shapes() {
        let request = AcpRequest::new(3, "session/prompt", json(r#"{"sessionID":"s"}"#));
        assert_eq!(
            request.to_json(),
            json(
                r#"{"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionID":"s"}}"#
            )
        );
        let ok = AcpResponse::result(1, json(r#"{"ok":true}"#));
        assert_eq!(
            ok.to_json(),
            json(r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true},"error":null}"#)
        );
        let err = AcpResponse::error_code(1, -32601, "method not found");
        assert_eq!(err.to_json()["error"]["code"], -32601);
        let encoded = frame("initialize".into(), 1, json!({}));
        assert!(encoded.starts_with(b"Content-Length: "));
        assert!(encoded.ends_with(b"}"));
    }

    #[test]
    fn notification_frame_omits_id_member_entirely() {
        let raw = notification_frame("session/update".into(), json!({"sessionID": "s"}));
        let (_, value) = parse_frame(&raw).unwrap().unwrap();
        assert_eq!(value["jsonrpc"], "2.0");
        assert!(
            value.get("id").is_none(),
            "notifications must omit id, got {value}"
        );
        assert_eq!(value["method"], "session/update");
        // Raw bytes: the `"id"` member must not appear anywhere (the
        // official SDK classifies an id-bearing frame as a request).
        let text = String::from_utf8(raw).unwrap();
        assert!(!text.contains("\"id\""), "{text}");
        let line = encode_line(&notification_body(
            "session/update".into(),
            json!({"sessionId": "s"}),
        ))
        .unwrap();
        assert!(line.ends_with(b"\n"));
        let (_, value) = parse_ndjson(&line).unwrap().unwrap();
        assert!(value.get("id").is_none());
    }

    #[test]
    fn framing_detection_reads_first_non_whitespace_byte() {
        assert_eq!(detect_framing(b""), None);
        assert_eq!(detect_framing(b" \r\n\t"), None);
        assert_eq!(
            detect_framing(br#"{"jsonrpc":"2.0""#),
            Some(Framing::Ndjson)
        );
        assert_eq!(detect_framing(b"\r\n  [1,2]"), Some(Framing::Ndjson));
        assert_eq!(
            detect_framing(b"Content-Length: 5\r\n\r\n"),
            Some(Framing::ContentLength)
        );
        assert_eq!(
            detect_framing(b"content-length: 5"),
            Some(Framing::ContentLength)
        );
    }

    #[test]
    fn ndjson_parses_lines_blank_lines_and_fragmentation() {
        let line = encode_line(&json!({"a":1})).unwrap();
        let (consumed, value) = parse_ndjson(&line).unwrap().unwrap();
        assert_eq!(consumed, line.len());
        assert_eq!(value, json!({"a":1}));
        assert_eq!(parse_ndjson(b"{\"a\":1").unwrap(), None);
        // CRLF line endings are accepted; blank lines are skipped and
        // accounted in `consumed`.
        let mut framed = b"\n\r\n  \r\n".to_vec();
        framed.extend_from_slice(&line);
        let (consumed, value) = parse_ndjson(&framed).unwrap().unwrap();
        assert_eq!(consumed, framed.len());
        assert_eq!(value, json!({"a":1}));
        // Trailing bytes past the first message stay unconsumed.
        let mut two = line.clone();
        two.extend_from_slice(&line);
        let (consumed, _) = parse_ndjson(&two).unwrap().unwrap();
        assert_eq!(consumed, line.len());
        // One byte at a time is equivalent to one write.
        let mut feed = Vec::new();
        for (i, byte) in line.iter().enumerate() {
            feed.push(*byte);
            if i + 1 < line.len() {
                assert_eq!(parse_ndjson(&feed).unwrap(), None);
            }
        }
        assert_eq!(parse_ndjson(&feed).unwrap().unwrap().1, json!({"a":1}));
    }

    #[test]
    fn ndjson_invalid_line_is_recoverable_and_oversized_is_fatal() {
        // Blank lines are reported per line so callers drain them.
        assert_eq!(
            parse_ndjson_detailed(b"  \r\n").unwrap().unwrap(),
            (4, None)
        );
        let bad = b"{\"broken\n";
        let err = parse_ndjson_detailed(bad).unwrap_err();
        assert!(!err.fatal, "{err:?}");
        assert_eq!(err.consumed, bad.len());
        // A line with no newline beyond the frame bound is a hostile stream:
        // refuse without buffering the rest of the connection.
        let huge = vec![b'x'; MAX_FRAME_BYTES + 1];
        let err = parse_ndjson_detailed(&huge).unwrap_err();
        assert!(err.fatal, "oversized line must be fatal");
        assert!(err.message.contains("newline"), "{}", err.message);
    }

    #[test]
    fn zero_length_body_is_a_parse_error_not_a_hang() {
        let frame = b"Content-Length: 0\r\n\r\n";
        assert!(parse_frame(frame).is_err());
    }
}
