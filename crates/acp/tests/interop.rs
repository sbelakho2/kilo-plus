//! Independent-parser ACP interop harness.
//!
//! Certification needs "somebody else's parser" acceptance: no Zed or
//! JetBrains binary exists in this environment, so the agent under test is
//! driven by a hand-rolled client that shares NO code with this crate —
//! no `faktor_acp` imports on the client path, no `protocol::` helpers, no
//! `serde_json`. It builds frames with its own serializer, parses frames
//! with its own Content-Length scanner and its own recursive-descent JSON
//! parser, and classifies every server frame against its own understanding
//! of the official ACP v1 kinds (initialize result, session/new result,
//! prompt stopReason, `session/update` with `agent_message_chunk`).
//! Anything else is an unknown kind and is counted, never guessed.
//!
//! Transport: real TCP on 127.0.0.1:0. The acp server runs on a tokio
//! runtime bridged to std sockets by pump tasks (the acp crate compiles
//! without the tokio `net` feature, so the sockets are std::net and the
//! bridge is plain blocking IO on spawn_blocking). The independent client
//! is plain std: one reader thread feeds a bounded channel, the test
//! thread writes requests and parses frames.
//!
//! Framing fact locked here: the acp server frames messages with
//! `Content-Length: N\r\n\r\n<body>` (LSP/MCP style), NOT newline
//! delimited JSON-RPC. The independent client mirrors that exactly and
//! rejects newline-delimited bodies as unframed garbage.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt, DuplexStream};

use faktor_acp::{AcpServer, AcpStreamBackend, PromptCtx};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
const QUIESCE: Duration = Duration::from_millis(300);
const FRAME_DEADLINE: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Independent JSON value, parser and serializer (own code, no serde).
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum J {
    Null,
    Bool(bool),
    I(i64),
    U(u64),
    F(f64),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
}

const MAX_DEPTH: usize = 256;

impl J {
    fn get(&self, key: &str) -> Option<&J> {
        match self {
            J::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            J::Str(s) => Some(s),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            J::I(i) => Some(*i),
            J::U(u) => i64::try_from(*u).ok(),
            J::F(f) if f.fract() == 0.0 && f.is_finite() && *f >= -9.2e18 && *f <= 9.2e18 => {
                Some(*f as i64)
            }
            _ => None,
        }
    }
}

struct Cp<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> Cp<'a> {
    fn ws(&mut self) {
        while let Some(c) = self.peek() {
            if matches!(c, ' ' | '\t' | '\r' | '\n') {
                self.i += c.len_utf8();
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.s[self.i..].chars().next()
    }

    fn expect(&mut self, ch: char) -> Result<(), String> {
        self.ws();
        if self.s[self.i..].starts_with(ch) {
            self.i += ch.len_utf8();
            Ok(())
        } else {
            Err(format!("expected {ch:?} at byte {}", self.i))
        }
    }

    fn lit(&mut self, word: &str) -> Result<(), String> {
        self.ws();
        if self.s[self.i..].starts_with(word) {
            self.i += word.len();
            Ok(())
        } else {
            Err(format!("expected literal {word:?} at byte {}", self.i))
        }
    }

    fn value(&mut self, depth: usize) -> Result<J, String> {
        if depth > MAX_DEPTH {
            return Err(format!("JSON nesting deeper than {MAX_DEPTH}"));
        }
        self.ws();
        let c = self.peek().ok_or("unexpected end of JSON")?;
        match c {
            'n' => {
                self.lit("null")?;
                Ok(J::Null)
            }
            't' => {
                self.lit("true")?;
                Ok(J::Bool(true))
            }
            'f' => {
                self.lit("false")?;
                Ok(J::Bool(false))
            }
            '"' => self.string().map(J::Str),
            '[' => {
                self.i += 1;
                let mut items = Vec::new();
                self.ws();
                if self.peek() == Some(']') {
                    self.i += 1;
                    return Ok(J::Arr(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    if self.peek() == Some(']') {
                        self.i += 1;
                        return Ok(J::Arr(items));
                    }
                    self.expect(',')?;
                }
            }
            '{' => {
                self.i += 1;
                let mut fields = Vec::new();
                self.ws();
                if self.peek() == Some('}') {
                    self.i += 1;
                    return Ok(J::Obj(fields));
                }
                loop {
                    let key = match self.value(depth + 1)? {
                        J::Str(k) => k,
                        other => return Err(format!("object key is not a string: {other:?}")),
                    };
                    self.expect(':')?;
                    let val = self.value(depth + 1)?;
                    fields.push((key, val));
                    self.ws();
                    if self.peek() == Some('}') {
                        self.i += 1;
                        return Ok(J::Obj(fields));
                    }
                    self.expect(',')?;
                }
            }
            '-' | '0'..='9' => self.number(),
            _ => Err(format!("unexpected character {c:?} at byte {}", self.i)),
        }
    }

    fn string(&mut self) -> Result<String, String> {
        debug_assert_eq!(self.peek(), Some('"'));
        self.i += 1;
        let mut out = String::new();
        loop {
            let c = self.peek().ok_or("unterminated string")?;
            match c {
                '"' => {
                    self.i += 1;
                    return Ok(out);
                }
                '\\' => {
                    self.i += 1;
                    let esc = self.peek().ok_or("unterminated escape")?;
                    match esc {
                        '"' | '\\' | '/' => {
                            self.i += 1;
                            out.push(esc);
                        }
                        'b' => {
                            self.i += 1;
                            out.push('\u{8}');
                        }
                        'f' => {
                            self.i += 1;
                            out.push('\u{c}');
                        }
                        'n' => {
                            self.i += 1;
                            out.push('\n');
                        }
                        'r' => {
                            self.i += 1;
                            out.push('\r');
                        }
                        't' => {
                            self.i += 1;
                            out.push('\t');
                        }
                        'u' => {
                            self.i += 1;
                            let hi = self.hex4()?;
                            if (0xd800..=0xdbff).contains(&hi) {
                                if !self.s[self.i..].starts_with("\\u") {
                                    return Err("lone high surrogate".into());
                                }
                                self.i += 2;
                                let lo = self.hex4()?;
                                if !(0xdc00..=0xdfff).contains(&lo) {
                                    return Err("high surrogate not followed by a low one".into());
                                }
                                let code = 0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                                out.push(char::from_u32(code).ok_or("bad surrogate pair")?);
                            } else if (0xdc00..=0xdfff).contains(&hi) {
                                return Err("lone low surrogate".into());
                            } else if let Some(ch) = char::from_u32(hi) {
                                out.push(ch);
                            } else {
                                return Err("escape is not a scalar value".into());
                            }
                        }
                        other => return Err(format!("bad escape \\{other}")),
                    }
                }
                c if (c as u32) < 0x20 => return Err("raw control byte in string".into()),
                c => {
                    out.push(c);
                    self.i += c.len_utf8();
                }
            }
        }
    }

    fn number(&mut self) -> Result<J, String> {
        let start = self.i;
        let rest = &self.s[self.i..];
        let mut len = 0usize;
        for c in rest.chars() {
            if c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E') {
                len += c.len_utf8();
            } else {
                break;
            }
        }
        if len == 0 {
            return Err("empty number".into());
        }
        self.i += len;
        let text = &self.s[start..self.i];
        let is_float = text.contains(['.', 'e', 'E']);
        if is_float {
            match text.parse::<f64>() {
                Ok(f) if f.is_finite() => Ok(J::F(f)),
                _ => Err(format!("bad float {text:?}")),
            }
        } else if let Ok(i) = text.parse::<i64>() {
            Ok(J::I(i))
        } else if let Ok(u) = text.parse::<u64>() {
            Ok(J::U(u))
        } else {
            Err(format!("integer out of range: {text:?}"))
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let end = self.i + 4;
        let digits = self.s.get(self.i..end).ok_or("truncated \\u escape")?;
        let mut acc = 0u32;
        for c in digits.chars() {
            acc = acc * 16 + c.to_digit(16).ok_or("bad hex digit in \\u escape")?;
        }
        self.i = end;
        Ok(acc)
    }
}

fn parse_json(s: &str) -> Result<J, String> {
    let mut cp = Cp { s, i: 0 };
    let v = cp.value(0)?;
    cp.ws();
    if cp.i != s.len() {
        return Err(format!("trailing data at byte {}", cp.i));
    }
    Ok(v)
}

fn escape_into(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_json(j: &J, out: &mut String) {
    match j {
        J::Null => out.push_str("null"),
        J::Bool(true) => out.push_str("true"),
        J::Bool(false) => out.push_str("false"),
        J::I(i) => out.push_str(&i.to_string()),
        J::U(u) => out.push_str(&u.to_string()),
        J::F(f) => out.push_str(&f.to_string()),
        J::Str(s) => escape_into(s, out),
        J::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(item, out);
            }
            out.push(']');
        }
        J::Obj(fields) => {
            out.push('{');
            for (i, (k, v)) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                escape_into(k, out);
                out.push(':');
                write_json(v, out);
            }
            out.push('}');
        }
    }
}

fn jstr(s: &str) -> J {
    J::Str(s.to_string())
}

fn jobj(fields: &[(&str, J)]) -> J {
    J::Obj(
        fields
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

fn jarr(items: &[J]) -> J {
    J::Arr(items.to_vec())
}

fn text_block(text: &str) -> J {
    jobj(&[("type", jstr("text")), ("text", jstr(text))])
}

// ---------------------------------------------------------------------------
// Independent framing: Content-Length scanner mirroring the documented wire.
// ---------------------------------------------------------------------------

const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;
const MAX_HEADER_BYTES: usize = 16 * 1024;
const MAX_HEADER_LINES: usize = 64;
const TERMINATOR: &[u8] = b"\r\n\r\n";

enum Decode {
    Done(usize, J),
    NeedMore,
    Recoverable(String, usize),
    Fatal(String),
}

fn trim_ascii(mut line: &[u8]) -> &[u8] {
    while let [first, rest @ ..] = line {
        if first.is_ascii_whitespace() {
            line = rest;
        } else {
            break;
        }
    }
    while let [rest @ .., last] = line {
        if last.is_ascii_whitespace() {
            line = rest;
        } else {
            break;
        }
    }
    line
}

fn find_terminator(buf: &[u8]) -> Option<usize> {
    let len = buf.len();
    if len < TERMINATOR.len() {
        return None;
    }
    (0..=len - TERMINATOR.len()).find(|&i| &buf[i..i + TERMINATOR.len()] == TERMINATOR)
}

fn decode_one(buf: &[u8]) -> Decode {
    let Some(pos) = find_terminator(buf) else {
        if buf.len() > MAX_HEADER_BYTES {
            return Decode::Fatal(format!(
                "no header terminator within {MAX_HEADER_BYTES} bytes; not Content-Length framed"
            ));
        }
        return Decode::NeedMore;
    };
    let header_end = pos + TERMINATOR.len();
    let header = &buf[..pos];
    let mut content_length: Option<u64> = None;
    let mut lines = 0usize;
    for raw in header.split(|b| *b == b'\r' || *b == b'\n') {
        let line = trim_ascii(raw);
        if line.is_empty() {
            continue;
        }
        lines += 1;
        if lines > MAX_HEADER_LINES {
            return Decode::Recoverable(
                format!("more than {MAX_HEADER_LINES} header lines"),
                header_end,
            );
        }
        let Some(colon) = line.iter().position(|b| *b == b':') else {
            return Decode::Recoverable(
                format!("malformed header line: {:?}", String::from_utf8_lossy(line)),
                header_end,
            );
        };
        let (name, value) = (&line[..colon], trim_ascii(&line[colon + 1..]));
        if name.eq_ignore_ascii_case(b"content-length") {
            if content_length.is_some() {
                return Decode::Recoverable("duplicate Content-Length".to_string(), header_end);
            }
            let text = match std::str::from_utf8(value) {
                Ok(t) => t,
                Err(_) => {
                    return Decode::Recoverable(
                        "Content-Length is not ASCII".to_string(),
                        header_end,
                    )
                }
            };
            let n: u64 = match text.trim().parse() {
                Ok(n) => n,
                Err(_) => {
                    return Decode::Recoverable(
                        format!("invalid Content-Length {text:?}"),
                        header_end,
                    )
                }
            };
            content_length = Some(n);
        }
    }
    let declared = match content_length {
        Some(n) => n,
        None => return Decode::Recoverable("missing Content-Length".to_string(), header_end),
    };
    if declared > MAX_FRAME_BYTES as u64 {
        return Decode::Fatal(format!(
            "declared Content-Length {declared} exceeds the {MAX_FRAME_BYTES}-byte frame bound"
        ));
    }
    let total = match header_end.checked_add(declared as usize) {
        Some(t) => t,
        None => return Decode::Fatal("Content-Length overflows usize".to_string()),
    };
    if buf.len() < total {
        return Decode::NeedMore;
    }
    let body = match std::str::from_utf8(&buf[header_end..total]) {
        Ok(b) => b,
        Err(_) => return Decode::Recoverable("body is not UTF-8".to_string(), total),
    };
    match parse_json(body) {
        Ok(v) => Decode::Done(total, v),
        Err(msg) => Decode::Recoverable(format!("invalid JSON body: {msg}"), total),
    }
}

fn encode_frame(method: &str, id: J, params: &J) -> Vec<u8> {
    let body = jobj(&[
        ("jsonrpc", jstr("2.0")),
        ("id", id),
        ("method", jstr(method)),
        ("params", params.clone()),
    ]);
    let mut s = String::new();
    write_json(&body, &mut s);
    let bytes = s.into_bytes();
    let mut out = format!("Content-Length: {}\r\n\r\n", bytes.len()).into_bytes();
    out.extend_from_slice(&bytes);
    out
}

// ---------------------------------------------------------------------------
// Independent client.
// ---------------------------------------------------------------------------

enum Incoming {
    Bytes(Vec<u8>),
    Eof,
}

fn spawn_reader(mut sock: TcpStream) -> mpsc::Receiver<Incoming> {
    let (tx, rx) = mpsc::sync_channel(2048);
    std::thread::spawn(move || {
        let mut buf = [0u8; 8192];
        loop {
            match sock.read(&mut buf) {
                Ok(0) => {
                    let _ = tx.send(Incoming::Eof);
                    return;
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => {
                    let _ = tx.send(Incoming::Eof);
                    return;
                }
                Ok(n) => {
                    if tx.send(Incoming::Bytes(buf[..n].to_vec())).is_err() {
                        return;
                    }
                }
            }
        }
    });
    rx
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Init,
    New,
    Prompt,
    Cancel,
}

#[derive(Debug)]
enum Ev {
    InitResult { version: i64 },
    NewSession { sid: String },
    PromptEnd { session: String, stop: String },
    CancelAck,
    Error { code: i64, data: Option<J> },
    Chunk { session: String, text: String },
    Unknown(String),
}

#[derive(Debug)]
enum ClientErr {
    Timeout,
    Eof,
    Truncated(String),
    Protocol(String),
    Io(String),
}

impl ClientErr {
    fn is_close(&self) -> bool {
        matches!(
            self,
            ClientErr::Eof | ClientErr::Truncated(_) | ClientErr::Protocol(_)
        )
    }

    fn describe(&self) -> String {
        match self {
            ClientErr::Timeout => "read timed out".to_string(),
            ClientErr::Eof => "clean end of stream".to_string(),
            ClientErr::Truncated(msg) | ClientErr::Protocol(msg) | ClientErr::Io(msg) => {
                msg.clone()
            }
        }
    }
}

struct IndieClient {
    sock: TcpStream,
    rx: mpsc::Receiver<Incoming>,
    buf: Vec<u8>,
    next_id: u64,
    pending: Vec<(u64, Method, Option<String>)>,
    chunks: Vec<(String, String)>,
    terminals: Vec<(String, String)>,
    unknown: Vec<String>,
    closed: bool,
}

impl IndieClient {
    fn connect(addr: SocketAddr) -> Result<Self, ClientErr> {
        let sock = TcpStream::connect_timeout(&addr, Duration::from_secs(5))
            .map_err(|e| ClientErr::Io(format!("connect: {e}")))?;
        let _ = sock.set_nodelay(true);
        let _ = sock.set_read_timeout(Some(Duration::from_secs(2)));
        let _ = sock.set_write_timeout(Some(Duration::from_secs(10)));
        let reader = sock.try_clone().map_err(|e| ClientErr::Io(e.to_string()))?;
        let rx = spawn_reader(reader);
        Ok(Self {
            sock,
            rx,
            buf: Vec::new(),
            next_id: 1,
            pending: Vec::new(),
            chunks: Vec::new(),
            terminals: Vec::new(),
            unknown: Vec::new(),
            closed: false,
        })
    }

    fn send(&mut self, method: &str, id: Option<u64>, params: &J) -> Result<(), ClientErr> {
        let bytes = encode_frame(method, id.map_or(J::Null, |n| J::I(n as i64)), params);
        self.sock
            .write_all(&bytes)
            .map_err(|e| ClientErr::Io(format!("write: {e}")))?;
        self.sock
            .flush()
            .map_err(|e| ClientErr::Io(format!("flush: {e}")))
    }

    fn send_raw(&mut self, bytes: &[u8]) -> Result<(), ClientErr> {
        self.sock
            .write_all(bytes)
            .map_err(|e| ClientErr::Io(format!("write: {e}")))?;
        self.sock
            .flush()
            .map_err(|e| ClientErr::Io(format!("flush: {e}")))
    }

    fn request(
        &mut self,
        method: Method,
        session: Option<String>,
        params: &J,
    ) -> Result<u64, ClientErr> {
        let id = self.next_id;
        self.next_id += 1;
        self.pending.push((id, method, session));
        let wire = match method {
            Method::Init => "initialize",
            Method::New => "session/new",
            Method::Prompt => "session/prompt",
            Method::Cancel => "session/cancel",
        };
        self.send(wire, Some(id), params)?;
        Ok(id)
    }

    fn notify(&mut self, method: &str, params: &J) -> Result<(), ClientErr> {
        self.send(method, None, params)
    }

    fn rpc(&mut self, method: Method, params: &J) -> Result<Ev, ClientErr> {
        let _ = self.request(method, None, params)?;
        loop {
            let ev = self.next_event(DEFAULT_TIMEOUT)?;
            let done = matches!(
                (&ev, method),
                (Ev::InitResult { .. } | Ev::Error { .. }, Method::Init)
                    | (Ev::NewSession { .. } | Ev::Error { .. }, Method::New)
                    | (Ev::Error { .. }, Method::Prompt)
                    | (Ev::CancelAck | Ev::Error { .. }, Method::Cancel)
            );
            if done {
                return Ok(ev);
            }
        }
    }

    fn next_event(&mut self, timeout: Duration) -> Result<Ev, ClientErr> {
        if self.closed {
            return Err(ClientErr::Protocol(
                "connection closed after a framing error".into(),
            ));
        }
        let mut stalled_since: Option<Instant> = None;
        loop {
            match decode_one(&self.buf) {
                Decode::Done(consumed, j) => {
                    self.buf.drain(..consumed);
                    return Ok(self.classify(j));
                }
                Decode::Recoverable(msg, consumed) => {
                    self.buf.drain(..consumed);
                    return Ok(Ev::Unknown(format!("undecodable frame: {msg}")));
                }
                Decode::Fatal(msg) => {
                    self.closed = true;
                    return Err(ClientErr::Protocol(msg));
                }
                Decode::NeedMore => {
                    if stalled_since.is_none() {
                        stalled_since = Some(Instant::now());
                    }
                    if !self.buf.is_empty() && stalled_since.unwrap().elapsed() > FRAME_DEADLINE {
                        self.closed = true;
                        return Err(ClientErr::Truncated("frame never completed".into()));
                    }
                    match self.rx.recv_timeout(timeout) {
                        Ok(Incoming::Bytes(bytes)) => {
                            stalled_since = None;
                            self.buf.extend_from_slice(&bytes);
                        }
                        Ok(Incoming::Eof) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                            if self.buf.is_empty() {
                                return Err(ClientErr::Eof);
                            }
                            self.closed = true;
                            return Err(ClientErr::Truncated("connection closed mid-frame".into()));
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if self.buf.is_empty() {
                                return Err(ClientErr::Timeout);
                            }
                            self.closed = true;
                            return Err(ClientErr::Truncated(
                                "timed out waiting for the rest of a frame".into(),
                            ));
                        }
                    }
                }
            }
        }
    }

    fn classify(&mut self, j: J) -> Ev {
        if j.get("method").is_some() {
            self.classify_notification(&j)
        } else {
            self.classify_response(&j)
        }
    }

    fn classify_notification(&mut self, j: &J) -> Ev {
        let method = match j.get("method").and_then(J::as_str) {
            Some(m) => m,
            None => return self.unknown_kind("notification without a method string"),
        };
        if method != "session/update" {
            return self.unknown_kind(&format!(
                "server notification method {method:?} is not an official ACP kind"
            ));
        }
        let Some(params) = j.get("params") else {
            return self.unknown_kind("session/update without params");
        };
        let Some(sid) = params.get("sessionId").and_then(J::as_str) else {
            return self.unknown_kind("session/update without a string sessionId");
        };
        let Some(update) = params.get("update") else {
            return self.unknown_kind("session/update without an update object");
        };
        if update.get("kind").is_some() {
            let kind = update.get("kind").and_then(J::as_str).unwrap_or("?");
            return self.unknown_kind(&format!(
                "extension update kind {kind:?} arrived although the client declared no extensions"
            ));
        }
        match update.get("sessionUpdate").and_then(J::as_str) {
            Some("agent_message_chunk") => {
                let Some(content) = update.get("content") else {
                    return self.unknown_kind("agent_message_chunk without content");
                };
                if content.get("type").and_then(J::as_str) != Some("text") {
                    return self.unknown_kind("agent_message_chunk content type is not text");
                }
                let Some(text) = content.get("text").and_then(J::as_str) else {
                    return self.unknown_kind("agent_message_chunk content without text");
                };
                let session = sid.to_string();
                let text = text.to_string();
                self.chunks.push((session.clone(), text.clone()));
                Ev::Chunk { session, text }
            }
            other => self.unknown_kind(&format!(
                "sessionUpdate kind {other:?} is not a kind this client knows"
            )),
        }
    }

    fn classify_response(&mut self, j: &J) -> Ev {
        if let Some(error) = j.get("error") {
            let Some(code) = error.get("code").and_then(J::as_i64) else {
                return self.unknown_kind("error response without a numeric code");
            };
            if let Some(id) = j.get("id").and_then(J::as_i64) {
                if let Some(pos) = self
                    .pending
                    .iter()
                    .position(|(pid, _, _)| *pid as i64 == id)
                {
                    self.pending.remove(pos);
                }
            }
            return Ev::Error {
                code,
                data: error.get("data").cloned(),
            };
        }
        let Some(result) = j.get("result") else {
            return self.unknown_kind("response without result or error");
        };
        let Some(id) = j.get("id").and_then(J::as_i64) else {
            return self.unknown_kind("result response without a numeric id");
        };
        let Some(pos) = self
            .pending
            .iter()
            .position(|(pid, _, _)| *pid as i64 == id)
        else {
            return self.unknown_kind(&format!(
                "result response for id {id} this client never sent"
            ));
        };
        let (_pid, method, session) = self.pending.remove(pos);
        match method {
            Method::Init => {
                let Some(version) = result.get("protocolVersion").and_then(J::as_i64) else {
                    return self.unknown_kind("initialize result without numeric protocolVersion");
                };
                Ev::InitResult { version }
            }
            Method::New => {
                let Some(sid) = result.get("sessionId").and_then(J::as_str) else {
                    return self.unknown_kind("session/new result without a string sessionId");
                };
                Ev::NewSession {
                    sid: sid.to_string(),
                }
            }
            Method::Prompt => {
                let Some(session) = session else {
                    return self.unknown_kind("prompt pending without a session");
                };
                let Some(stop) = result.get("stopReason").and_then(J::as_str) else {
                    return self.unknown_kind("prompt result without a string stopReason");
                };
                let stop = stop.to_string();
                self.terminals.push((session.clone(), stop.clone()));
                Ev::PromptEnd { session, stop }
            }
            Method::Cancel => {
                if result.get("stopReason").is_some() {
                    return self.unknown_kind("cancel answered with a stopReason");
                }
                Ev::CancelAck
            }
        }
    }

    fn unknown_kind(&mut self, desc: &str) -> Ev {
        self.unknown.push(desc.to_string());
        Ev::Unknown(desc.to_string())
    }

    fn assert_no_unknown(&self, what: &str) {
        assert!(
            self.unknown.is_empty(),
            "{what}: the client saw {count} frame(s) it cannot classify as official ACP: {seen:?}",
            count = self.unknown.len(),
            seen = self.unknown
        );
    }
}

// ---------------------------------------------------------------------------
// Server seam: fake streaming agent + TCP harness (mirrors tests/acp.rs).
// ---------------------------------------------------------------------------

const BURST_CAP: u64 = 60_000;

#[derive(Clone)]
struct FakeAgent {
    inner: Arc<Inner>,
}

struct Inner {
    next: AtomicU64,
    sessions: Mutex<Vec<String>>,
    emitted: AtomicU64,
}

impl FakeAgent {
    fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                next: AtomicU64::new(0),
                sessions: Mutex::new(Vec::new()),
                emitted: AtomicU64::new(0),
            }),
        }
    }

    fn emitted(&self) -> u64 {
        self.inner.emitted.load(Ordering::SeqCst)
    }
}

impl AcpStreamBackend for FakeAgent {
    fn agent_info(&self) -> Value {
        json!({
            "name": "faktor-interop-agent",
            "version": "0.0.0",
            "capabilities": { "prompt": true, "sessions": true },
        })
    }

    fn create_session(&self, params: &Value) -> Result<String, String> {
        if params.get("fail").is_some() {
            return Err("create refused by backend".into());
        }
        let n = self.inner.next.fetch_add(1, Ordering::SeqCst);
        let id = format!("sess-{n}");
        self.inner.sessions.lock().unwrap().push(id.clone());
        Ok(id)
    }

    fn list_sessions(&self) -> Vec<String> {
        self.inner.sessions.lock().unwrap().clone()
    }

    fn prompt<'a>(
        &'a self,
        _session_id: &'a str,
        ctx: &'a PromptCtx,
        text: &'a str,
    ) -> BoxFuture<'a, Result<Value, String>> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            let parts: Vec<&str> = text.splitn(4, ':').collect();
            match parts.as_slice() {
                ["chunks", n, word] | ["chunks", n, word, "0"] => {
                    let count: u64 = n.parse().map_err(|_| "bad chunk count")?;
                    for i in 0..count {
                        let chunk = format!("{word}#{i};");
                        if ctx.emit_text(&chunk).await.is_err() {
                            break;
                        }
                    }
                    Ok(json!({ "directive": "chunks", "word": word }))
                }
                ["chunks", n, word, ms] => {
                    let count: u64 = n.parse().map_err(|_| "bad chunk count")?;
                    let delay: u64 = ms.parse().map_err(|_| "bad chunk delay")?;
                    for i in 0..count {
                        let chunk = format!("{word}#{i};");
                        if ctx.emit_text(&chunk).await.is_err() {
                            break;
                        }
                        if delay > 0 {
                            tokio::time::sleep(Duration::from_millis(delay)).await;
                        }
                    }
                    Ok(json!({ "directive": "chunks", "word": word }))
                }
                ["park", ..] => {
                    let _ = ctx.emit_text("park#0;").await;
                    ctx.cancelled().await;
                    Ok(json!({ "echo": "parked" }))
                }
                ["burst"] => {
                    let mut i = 0u64;
                    while i < BURST_CAP {
                        let chunk = format!("b{i:06};{}", "x".repeat(1024));
                        match ctx.emit_text(&chunk).await {
                            Ok(()) => {
                                inner.emitted.store(i + 1, Ordering::SeqCst);
                                i += 1;
                            }
                            Err(_) => break,
                        }
                    }
                    Ok(json!({ "directive": "burst" }))
                }
                _ => Err(format!("backend does not understand directive {text:?}")),
            }
        })
    }
}

struct Harness {
    addr: SocketAddr,
    listener: Option<TcpListener>,
    kills: Arc<Mutex<Vec<Option<tokio::sync::oneshot::Sender<()>>>>>,
    conn_threads: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>>,
    accept_thread: Option<std::thread::JoinHandle<()>>,
    stop: Arc<AtomicBool>,
}

impl Harness {
    fn start(backend: FakeAgent) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
        let addr = listener.local_addr().expect("local addr");
        let server = AcpServer::new_streaming(backend);
        let kills = Arc::new(Mutex::new(Vec::new()));
        let conn_threads = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let accept_listener = listener.try_clone().expect("clone listener");
        let _ = accept_listener.set_nonblocking(true);
        let kills_t = Arc::clone(&kills);
        let conn_threads_t = Arc::clone(&conn_threads);
        let stop_t = Arc::clone(&stop);
        let accept_thread = std::thread::spawn(move || loop {
            if stop_t.load(Ordering::SeqCst) {
                break;
            }
            match accept_listener.accept() {
                Ok((sock, _)) => {
                    let (kill_tx, kill_rx) = tokio::sync::oneshot::channel();
                    kills_t.lock().unwrap().push(Some(kill_tx));
                    let server = server.clone();
                    conn_threads_t
                        .lock()
                        .unwrap()
                        .push(std::thread::spawn(move || {
                            conn_thread(sock, server, kill_rx);
                        }));
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(_) => std::thread::sleep(Duration::from_millis(5)),
            }
        });
        Self {
            addr,
            listener: Some(listener),
            kills,
            conn_threads,
            accept_thread: Some(accept_thread),
            stop,
        }
    }

    fn conn_count(&self) -> usize {
        self.kills.lock().unwrap().len()
    }

    fn drop_conn(&self, idx: usize) {
        if let Some(tx) = self
            .kills
            .lock()
            .unwrap()
            .get_mut(idx)
            .and_then(Option::take)
        {
            let _ = tx.send(());
        }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        drop(self.listener.take());
        for tx in self.kills.lock().unwrap().iter_mut() {
            if let Some(tx) = tx.take() {
                let _ = tx.send(());
            }
        }
        if let Some(thread) = self.accept_thread.take() {
            let _ = thread.join();
        }
        for thread in self.conn_threads.lock().unwrap().drain(..) {
            let _ = thread.join();
        }
    }
}

fn conn_thread(sock: TcpStream, server: AcpServer, kill_rx: tokio::sync::oneshot::Receiver<()>) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_time()
        .build()
        .expect("conn runtime");
    rt.block_on(async {
        tokio::select! {
            biased;
            _ = kill_rx => {}
            res = serve_tcp(sock, server) => {
                let _ = res;
            }
        }
    });
    rt.shutdown_timeout(Duration::from_millis(300));
}

async fn serve_tcp(sock: TcpStream, server: AcpServer) -> Result<(), String> {
    let _ = sock.set_nodelay(true);
    let _ = sock.set_nonblocking(false);
    let _ = sock.set_read_timeout(Some(Duration::from_secs(1)));
    let _ = sock.set_write_timeout(Some(Duration::from_secs(5)));
    let (c2s_r, c2s_w) = tokio::io::duplex(1024 * 1024);
    let (s2c_r, s2c_w) = tokio::io::duplex(1024 * 1024);
    let reader = sock.try_clone().map_err(|e| format!("clone: {e}"))?;
    let (done_tx, done_rx) = tokio::sync::watch::channel(());
    let pump_in = tokio::spawn(pump_tcp_to_duplex(reader, c2s_w, done_rx));
    let pump_out = tokio::spawn(pump_duplex_to_tcp(s2c_r, sock, done_tx.subscribe()));
    let res = server.serve_connection(c2s_r, s2c_w).await;
    let _ = done_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(5), async {
        let _ = pump_in.await;
        let _ = pump_out.await;
    })
    .await;
    res
}

async fn pump_tcp_to_duplex(
    mut tcp: TcpStream,
    mut dup: DuplexStream,
    mut done: tokio::sync::watch::Receiver<()>,
) {
    let mut buf = vec![0u8; 65536];
    loop {
        let outcome = tokio::select! {
            biased;
            _ = done.changed() => break,
            outcome = tokio::task::spawn_blocking(move || {
                let r = tcp.read(&mut buf);
                (r, buf, tcp)
            }) => outcome,
        };
        let (read, next_buf, t) = match outcome {
            Ok(triple) => triple,
            Err(_) => break,
        };
        buf = next_buf;
        tcp = t;
        match read {
            Ok(0) => break,
            Ok(n) => {
                if dup.write_all(&buf[..n]).await.is_err() {
                    break;
                }
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(_) => break,
        }
    }
}

async fn pump_duplex_to_tcp(
    mut dup: DuplexStream,
    mut tcp: TcpStream,
    mut done: tokio::sync::watch::Receiver<()>,
) {
    let mut buf = [0u8; 65536];
    loop {
        let read = tokio::select! {
            biased;
            _ = done.changed() => break,
            read = dup.read(&mut buf) => read,
        };
        match read {
            Ok(0) => break,
            Ok(n) => {
                let chunk = buf[..n].to_vec();
                let (written, t) = match tokio::task::spawn_blocking(move || {
                    let r = tcp.write_all(&chunk).and_then(|_| tcp.flush());
                    (r, tcp)
                })
                .await
                {
                    Ok(pair) => pair,
                    Err(_) => break,
                };
                tcp = t;
                if written.is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

// ---------------------------------------------------------------------------
// Test helpers.
// ---------------------------------------------------------------------------

fn wait_until(what: &str, deadline: Duration, mut cond: impl FnMut() -> bool) {
    let start = Instant::now();
    while !cond() {
        assert!(start.elapsed() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn expect_ok(ev: Result<Ev, ClientErr>, what: &str) -> Ev {
    match ev {
        Ok(ev) => ev,
        Err(err) => panic!("{what}: unexpected client error: {}", err.describe()),
    }
}

fn expect_chunk(ev: &Ev, what: &str) -> (String, String) {
    match ev {
        Ev::Chunk { session, text } => (session.clone(), text.clone()),
        other => panic!("{what}: expected a text chunk frame, got {other:?}"),
    }
}

fn expect_seq(ev: &Ev, what: &str, session: &str, word: &str, expect_idx: u64) {
    let (sid, text) = expect_chunk(ev, what);
    assert_eq!(sid, session, "{what}: chunk for the wrong session");
    let prefix = format!("{word}#{expect_idx};");
    assert!(
        text.starts_with(&prefix),
        "{what}: chunk out of order; expected {prefix:?} got {text:?}"
    );
}

fn prompt_params(session: &str, text: &str) -> J {
    jobj(&[
        ("sessionId", jstr(session)),
        ("prompt", jarr(&[text_block(text)])),
    ])
}

fn connect_and_init(addr: SocketAddr) -> IndieClient {
    let mut client = IndieClient::connect(addr).expect("client connects");
    let params = jobj(&[
        ("protocolVersion", J::I(1)),
        (
            "clientInfo",
            jobj(&[
                ("name", jstr("independent-parser")),
                ("version", jstr("0.1.0")),
            ]),
        ),
    ]);
    match client.rpc(Method::Init, &params) {
        Ok(Ev::InitResult { version }) => assert_eq!(version, 1, "protocol version"),
        other => panic!("initialize failed: {other:?}"),
    }
    client
}

fn rpc_new_session(client: &mut IndieClient) -> String {
    match client.rpc(Method::New, &jobj(&[])) {
        Ok(Ev::NewSession { sid }) => sid,
        other => panic!("session/new failed: {other:?}"),
    }
}

fn expect_stop(ev: &Ev, session: &str, stop: &str) {
    match ev {
        Ev::PromptEnd {
            session: s,
            stop: st,
        } => {
            assert_eq!(s, session, "terminal for the wrong session");
            assert_eq!(st, stop, "wrong stopReason");
        }
        other => panic!("expected prompt terminal, got {other:?}"),
    }
}

fn quiesce(client: &mut IndieClient) {
    match client.next_event(QUIESCE) {
        Err(ClientErr::Timeout) => {}
        other => panic!("expected a quiet connection, got {other:?}"),
    }
}

fn prompt_and_drain(
    client: &mut IndieClient,
    session: &str,
    directive: &str,
    word: &str,
    chunks: u64,
) {
    client
        .request(
            Method::Prompt,
            Some(session.to_string()),
            &prompt_params(session, directive),
        )
        .expect("prompt request");
    let mut got = Vec::new();
    loop {
        let ev = expect_ok(client.next_event(DEFAULT_TIMEOUT), "prompt stream");
        match &ev {
            Ev::Chunk { session: s, text } if s == session => {
                expect_seq(&ev, "chunk stream", session, word, got.len() as u64);
                got.push(text.clone());
            }
            Ev::PromptEnd { session: s, .. } if s == session => {
                expect_stop(&ev, session, "end_turn");
                break;
            }
            other => panic!("unexpected frame while draining a prompt: {other:?}"),
        }
    }
    assert_eq!(got.len(), chunks as usize, "chunk count");
}

// ---------------------------------------------------------------------------
// Scenarios.
// ---------------------------------------------------------------------------

#[test]
fn interop_a_initialize_handshake_and_v2_rejection() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = IndieClient::connect(harness.addr).expect("connect");

    let v2 = jobj(&[("protocolVersion", J::I(2))]);
    match client.rpc(Method::Init, &v2) {
        Ok(Ev::Error { code, data }) => {
            assert_eq!(code, -32602, "v2 must be refused with invalid params");
            let data = data.expect("typed version error data");
            assert_eq!(data.get("protocolVersion").and_then(J::as_i64), Some(2));
            assert_eq!(
                data.get("supportedProtocolVersion").and_then(J::as_i64),
                Some(1),
                "no silent fallback: the supported version must be announced"
            );
        }
        other => panic!("protocolVersion 2 must fail loudly, got {other:?}"),
    }

    let legacy = jobj(&[("protocolVersion", jstr("0.1.0"))]);
    match client.rpc(Method::Init, &legacy) {
        Ok(Ev::Error { code, .. }) => assert_eq!(code, -32602, "legacy string version rejected"),
        other => panic!("legacy version string must be rejected, got {other:?}"),
    }

    let Ev::InitResult { version } = expect_ok(
        client.rpc(Method::Init, &jobj(&[("protocolVersion", J::I(1))])),
        "init v1",
    ) else {
        panic!("initialize v1 failed after rejections");
    };
    assert_eq!(version, 1);

    let sid = rpc_new_session(&mut client);
    assert!(!sid.is_empty(), "sessionId must be a non-empty string");
    let sid2 = rpc_new_session(&mut client);
    assert_ne!(sid, sid2, "session ids must be unique");
    client.assert_no_unknown("initialize handshake");
}

#[test]
fn interop_b_session_new_returns_official_session_id() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = connect_and_init(harness.addr);
    let a = rpc_new_session(&mut client);
    let b = rpc_new_session(&mut client);
    assert_ne!(a, b);
    assert!(a.starts_with("sess-"), "{a}");
    assert!(b.starts_with("sess-"), "{b}");

    client
        .request(
            Method::Prompt,
            Some(a.clone()),
            &prompt_params(&a, "chunks:5:hello:0"),
        )
        .expect("prompt request");
    let mut got = Vec::new();
    for idx in 0..5u64 {
        let ev = expect_ok(client.next_event(DEFAULT_TIMEOUT), "streaming prompt");
        expect_seq(&ev, "text chunk stream", &a, "hello", idx);
        if let Ev::Chunk { text, .. } = &ev {
            got.push(text.clone());
        }
    }
    let ev = expect_ok(client.next_event(DEFAULT_TIMEOUT), "terminal");
    expect_stop(&ev, &a, "end_turn");
    quiesce(&mut client);

    let joined: String = got.concat();
    assert_eq!(joined, "hello#0;hello#1;hello#2;hello#3;hello#4;");
    assert_eq!(client.terminals.len(), 1, "exactly one terminal per turn");
    client.assert_no_unknown("session/new + streaming run");
}

#[test]
fn interop_c_cancel_midstream_yields_cancelled_exactly_once() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = connect_and_init(harness.addr);
    let session = rpc_new_session(&mut client);

    client
        .request(
            Method::Prompt,
            Some(session.clone()),
            &prompt_params(&session, "chunks:2000:slow:2"),
        )
        .expect("prompt request");

    let mut seen_first = false;
    let mut ack_count = 0usize;
    let mut chunk_count = 0usize;
    let mut terminal_count = 0usize;
    loop {
        let ev = expect_ok(client.next_event(DEFAULT_TIMEOUT), "cancel scenario");
        match &ev {
            Ev::Chunk { session: s, .. } if s == &session => {
                if !seen_first {
                    seen_first = true;
                    client
                        .request(
                            Method::Cancel,
                            Some(session.clone()),
                            &jobj(&[("sessionId", jstr(&session))]),
                        )
                        .expect("cancel request");
                }
                expect_seq(
                    &ev,
                    "slow stream before cancel",
                    &session,
                    "slow",
                    chunk_count as u64,
                );
                chunk_count += 1;
            }
            Ev::CancelAck => ack_count += 1,
            Ev::PromptEnd { session: s, stop } if s == &session => {
                terminal_count += 1;
                assert_eq!(
                    stop, "cancelled",
                    "terminal must be the official cancelled state"
                );
                break;
            }
            other => panic!("unexpected frame in the cancel scenario: {other:?}"),
        }
    }
    assert!(seen_first, "the turn must have started streaming");
    assert!(
        chunk_count < 2000,
        "cancel must land mid-stream, not after natural completion"
    );
    assert!(
        chunk_count >= 1,
        "at least one chunk arrives before the cancel"
    );
    assert_eq!(ack_count, 1, "cancel request answered exactly once");
    assert_eq!(terminal_count, 1, "cancelled terminal exactly once");
    quiesce(&mut client);

    prompt_and_drain(&mut client, &session, "chunks:3:after:0", "after", 3);
    client.assert_no_unknown("cancel-while-running");
}

#[test]
fn interop_d_server_drop_midstream_then_reconnect_works() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = connect_and_init(harness.addr);
    let session = rpc_new_session(&mut client);
    wait_until(
        "first connection registered",
        Duration::from_secs(5),
        || harness.conn_count() >= 1,
    );

    client
        .request(
            Method::Prompt,
            Some(session.clone()),
            &prompt_params(&session, "park"),
        )
        .expect("park request");
    let ev = expect_ok(client.next_event(DEFAULT_TIMEOUT), "park first chunk");
    expect_seq(&ev, "park chunk", &session, "park", 0);

    harness.drop_conn(0);
    loop {
        match client.next_event(DEFAULT_TIMEOUT) {
            Err(err) if err.is_close() => break,
            Err(err) => panic!("server drop must be a clean close, got {err:?}"),
            Ok(Ev::Chunk { .. }) => continue,
            Ok(other) => panic!("unexpected frame after the server dropped: {other:?}"),
        }
    }
    drop(client);

    let mut client2 = connect_and_init(harness.addr);
    wait_until(
        "second connection registered",
        Duration::from_secs(5),
        || harness.conn_count() >= 2,
    );
    let session2 = rpc_new_session(&mut client2);
    assert_ne!(
        session2, session,
        "a fresh connection must not replay the old session id"
    );
    prompt_and_drain(&mut client2, &session2, "chunks:2:again:0", "again", 2);
    client2.assert_no_unknown("reconnect after server drop");
}

#[test]
fn interop_e_two_connections_no_state_bleed() {
    let harness = Harness::start(FakeAgent::new());
    let mut client1 = connect_and_init(harness.addr);
    let s1 = rpc_new_session(&mut client1);

    client1
        .request(
            Method::Prompt,
            Some(s1.clone()),
            &prompt_params(&s1, "park"),
        )
        .expect("park");
    let ev = expect_ok(client1.next_event(DEFAULT_TIMEOUT), "park chunk");
    expect_seq(&ev, "park", &s1, "park", 0);

    harness.drop_conn(0);
    while client1.next_event(DEFAULT_TIMEOUT).is_ok() {}
    drop(client1);

    let mut client2 = connect_and_init(harness.addr);
    let s2 = rpc_new_session(&mut client2);
    assert_ne!(
        s1, s2,
        "session ids must not be replayed across connections"
    );
    prompt_and_drain(&mut client2, &s2, "chunks:3:beta:0", "beta", 3);
    quiesce(&mut client2);
    assert!(
        client2.chunks.iter().all(|(sid, _)| sid == &s2),
        "no frame of connection one may leak onto connection two"
    );
    client2.assert_no_unknown("sequential connections");
}

#[test]
fn interop_f_multi_session_interleaves_with_per_session_order() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = connect_and_init(harness.addr);
    let sa = rpc_new_session(&mut client);
    let sb = rpc_new_session(&mut client);
    assert_ne!(sa, sb);

    client
        .request(
            Method::Prompt,
            Some(sa.clone()),
            &prompt_params(&sa, "chunks:300:alpha:1"),
        )
        .expect("prompt a");
    client
        .request(
            Method::Prompt,
            Some(sb.clone()),
            &prompt_params(&sb, "chunks:300:beta:1"),
        )
        .expect("prompt b");

    let mut count_a = 0u64;
    let mut count_b = 0u64;
    let mut stop_a = false;
    let mut stop_b = false;
    while !(stop_a && stop_b) {
        let ev = expect_ok(client.next_event(DEFAULT_TIMEOUT), "multi-session stream");
        match &ev {
            Ev::Chunk { session, .. } if session == &sa => {
                expect_seq(&ev, "session a", &sa, "alpha", count_a);
                count_a += 1;
            }
            Ev::Chunk { session, .. } if session == &sb => {
                expect_seq(&ev, "session b", &sb, "beta", count_b);
                count_b += 1;
            }
            Ev::PromptEnd { session, stop } if session == &sa => {
                assert_eq!(stop, "end_turn", "session a terminal");
                stop_a = true;
            }
            Ev::PromptEnd { session, stop } if session == &sb => {
                assert_eq!(stop, "end_turn", "session b terminal");
                stop_b = true;
            }
            other => panic!("unexpected frame: {other:?}"),
        }
    }
    assert_eq!(count_a, 300, "session a chunk count");
    assert_eq!(count_b, 300, "session b chunk count");

    let positions: Vec<&String> = client.chunks.iter().map(|(sid, _)| sid).collect();
    let pos = |target: &str| {
        positions
            .iter()
            .enumerate()
            .filter(|(_, sid)| sid.as_str() == target)
            .map(|(i, _)| i)
            .collect::<Vec<_>>()
    };
    let a = pos(&sa);
    let b = pos(&sb);
    assert!(!a.is_empty() && !b.is_empty());
    assert!(
        a.first().unwrap() < b.last().unwrap() && b.first().unwrap() < a.last().unwrap(),
        "the two sessions must actually interleave on the wire"
    );
    client.assert_no_unknown("multi-session interleave");
}

#[test]
fn interop_g_bad_params_official_errors_connection_stays_usable() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = connect_and_init(harness.addr);
    let session = rpc_new_session(&mut client);

    let broken = b"Content-Length: 10\r\n\r\n{\"broken\":".to_vec();
    client.send_raw(&broken).expect("malformed frame");
    match expect_ok(client.next_event(DEFAULT_TIMEOUT), "malformed JSON") {
        Ev::Error { code, .. } => assert_eq!(code, -32700, "parse error"),
        other => panic!("malformed JSON must be a parse error, got {other:?}"),
    }

    let cases: Vec<(Method, J)> = vec![
        (Method::Prompt, jobj(&[])),
        (
            Method::Prompt,
            jobj(&[
                ("sessionId", J::I(7)),
                ("prompt", jarr(&[text_block("hi")])),
            ]),
        ),
        (
            Method::Prompt,
            jobj(&[("sessionId", jstr(&session)), ("prompt", jstr("hi"))]),
        ),
        (
            Method::Prompt,
            jobj(&[
                ("sessionId", jstr(&session)),
                (
                    "prompt",
                    jarr(&[jobj(&[("type", jstr("image")), ("data", jstr("x"))])]),
                ),
            ]),
        ),
        (Method::Prompt, jarr(&[J::I(1), J::I(2), J::I(3)])),
        (Method::Cancel, jobj(&[])),
        (Method::New, jobj(&[("fail", J::Bool(true))])),
    ];
    for (method, params) in cases {
        let ev = expect_ok(client.rpc(method, &params), "bad params");
        match ev {
            Ev::Error { code, .. } => {
                let expected = match method {
                    Method::New => -32603,
                    _ => -32602,
                };
                assert_eq!(code, expected, "official error code for bad {params:?}");
            }
            other => panic!("bad params must answer with an official error, got {other:?}"),
        }
    }

    let ok = rpc_new_session(&mut client);
    assert_ne!(ok, session);
    prompt_and_drain(&mut client, &ok, "chunks:2:ok:0", "ok", 2);
    client.assert_no_unknown("bad-params resilience");
}

#[test]
fn interop_h_oversized_request_hits_server_bound_close_not_hang() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = connect_and_init(harness.addr);

    client
        .send_raw(b"Content-Length: 20971520\r\n\r\n")
        .expect("hostile header");
    match expect_ok(client.next_event(DEFAULT_TIMEOUT), "server bound") {
        Ev::Error { code, .. } => assert_eq!(code, -32700, "declared 20 MiB body"),
        other => panic!("expected the server's parse error, got {other:?}"),
    }
    match client.next_event(DEFAULT_TIMEOUT) {
        Err(err) => assert!(
            err.is_close(),
            "framing is unrecoverable: the server must close: {err:?}"
        ),
        Ok(other) => panic!("expected close after the fatal framing error, got {other:?}"),
    }
}

#[test]
fn interop_i_slow_client_backpressure_no_deadlock_lossless() {
    let agent = FakeAgent::new();
    let harness = Harness::start(agent.clone());
    let mut client = connect_and_init(harness.addr);
    let session = rpc_new_session(&mut client);

    client
        .request(
            Method::Prompt,
            Some(session.clone()),
            &prompt_params(&session, "burst"),
        )
        .expect("burst prompt");

    let first = expect_ok(client.next_event(DEFAULT_TIMEOUT), "burst first chunk");
    let (first_sid, first_text) = expect_chunk(&first, "burst");
    assert_eq!(first_sid, session);
    assert!(
        first_text.starts_with("b000000;"),
        "first burst chunk must be b000000"
    );

    std::thread::sleep(Duration::from_millis(150));
    client
        .request(
            Method::Cancel,
            Some(session.clone()),
            &jobj(&[("sessionId", jstr(&session))]),
        )
        .expect("cancel while stalled");
    std::thread::sleep(Duration::from_millis(350));

    let started = Instant::now();
    let mut chunks = 1u64;
    let mut ack = false;
    let terminal_stop = loop {
        let ev = expect_ok(
            client.next_event(Duration::from_secs(20)),
            "slow client drain",
        );
        match &ev {
            Ev::Chunk { session: s, text } if s == &session => {
                let expected = format!("b{chunks:06};");
                assert!(
                    text.starts_with(&expected),
                    "chunk {chunks} out of order or corrupted"
                );
                chunks += 1;
            }
            Ev::CancelAck => ack = true,
            Ev::PromptEnd { session: s, stop } if s == &session => break stop.clone(),
            other => panic!("unexpected frame while draining: {other:?}"),
        }
    };
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "drain must complete inside the generous deadline"
    );
    assert!(chunks >= 100, "the burst must have flowed before the stall");
    assert_eq!(
        terminal_stop, "cancelled",
        "cancel during the stall cancels the turn"
    );
    assert!(ack, "cancel ack must arrive");
    assert!(
        chunks < BURST_CAP,
        "cancel must land while the backend is still bursting (got all {chunks})"
    );
    assert_eq!(
        chunks,
        agent.emitted(),
        "lossless: every frame the backend emitted must be delivered intact"
    );

    let fresh = rpc_new_session(&mut client);
    prompt_and_drain(&mut client, &fresh, "chunks:2:tail:0", "tail", 2);
    client.assert_no_unknown("slow-client backpressure");
}

#[test]
fn interop_j_extension_suppression_no_unknown_kinds_full_run() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = connect_and_init(harness.addr);
    let session = rpc_new_session(&mut client);

    prompt_and_drain(&mut client, &session, "chunks:4:base:0", "base", 4);

    client
        .request(
            Method::Prompt,
            Some(session.clone()),
            &prompt_params(&session, "chunks:600:notif:2"),
        )
        .expect("second prompt");
    let ev = expect_ok(
        client.next_event(DEFAULT_TIMEOUT),
        "first chunk of turn two",
    );
    expect_seq(&ev, "turn two", &session, "notif", 0);
    client
        .notify("session/cancel", &jobj(&[("sessionId", jstr(&session))]))
        .expect("notification-form cancel");
    loop {
        let ev = expect_ok(client.next_event(DEFAULT_TIMEOUT), "cancelled terminal");
        match &ev {
            Ev::Chunk { session: s, .. } if s == &session => continue,
            _ => {
                expect_stop(&ev, &session, "cancelled");
                break;
            }
        }
    }
    quiesce(&mut client);

    match client.rpc(Method::Cancel, &jobj(&[("sessionId", jstr(&session))])) {
        Ok(Ev::CancelAck) => {}
        other => panic!("idle cancel must be a clean ack, got {other:?}"),
    }

    assert!(
        client.chunks.iter().all(|(sid, _)| sid == &session),
        "all frames must belong to the declared session"
    );
    client.assert_no_unknown("full base run without any extension declaration");
}

// ---------------------------------------------------------------------------
// Adversarial: hostile server frames against the independent parser.
// ---------------------------------------------------------------------------

fn rogue_server(mut behavior: impl FnMut(TcpStream) + Send + 'static) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("rogue bind");
    let addr = listener.local_addr().expect("rogue addr");
    std::thread::spawn(move || {
        if let Ok((sock, _)) = listener.accept() {
            behavior(sock);
        }
    });
    addr
}

#[test]
fn interop_adv_rogue_truncation_never_hangs_or_panics() {
    let addr = rogue_server(|mut sock| {
        let frame = encode_frame(
            "session/update",
            J::Null,
            &jobj(&[
                ("sessionId", jstr("rogue")),
                (
                    "update",
                    jobj(&[
                        ("sessionUpdate", jstr("agent_message_chunk")),
                        (
                            "content",
                            jobj(&[("type", jstr("text")), ("text", jstr("dribbled#0;"))]),
                        ),
                    ]),
                ),
            ]),
        );
        for byte in frame.iter() {
            let _ = sock.write_all(&[*byte]);
            let _ = sock.flush();
            std::thread::sleep(Duration::from_millis(1));
        }
        std::thread::sleep(Duration::from_millis(50));
        let _ = sock.write_all(b"Content-Length: 200\r\n\r\n{\"truncated\":");
        let _ = sock.flush();
        std::thread::sleep(Duration::from_millis(30));
    });
    let mut client = IndieClient::connect(addr).expect("connect to rogue");

    let ev = expect_ok(client.next_event(DEFAULT_TIMEOUT), "dribbled chunk");
    expect_seq(&ev, "rogue dribble", "rogue", "dribbled", 0);

    match client.next_event(DEFAULT_TIMEOUT) {
        Err(ClientErr::Truncated(msg)) => assert!(msg.contains("mid-frame"), "{msg}"),
        Err(err) if err.is_close() => {}
        other => panic!("rogue truncation must surface as a clean close, got {other:?}"),
    }
}

#[test]
fn interop_adv_rogue_oversized_and_unframed_garbage() {
    let addr = rogue_server(|mut sock| {
        let _ = sock.write_all(b"Content-Length: 104857600\r\n\r\n");
        let _ = sock.flush();
        std::thread::sleep(Duration::from_millis(50));
        let _ = sock.write_all(&vec![b'x'; MAX_HEADER_BYTES + 100]);
        let _ = sock.flush();
        std::thread::sleep(Duration::from_millis(30));
    });
    let mut client = IndieClient::connect(addr).expect("connect to rogue");
    match client.next_event(DEFAULT_TIMEOUT) {
        Err(ClientErr::Protocol(msg)) => {
            assert!(
                msg.contains("exceeds"),
                "the client's own bound must refuse: {msg}"
            )
        }
        other => panic!("declared 100 MiB must be refused by the client's own bound: {other:?}"),
    }
    assert!(client.next_event(DEFAULT_TIMEOUT).is_err());

    let addr = rogue_server(|mut sock| {
        let _ = sock.write_all(&vec![b'x'; MAX_HEADER_BYTES + 100]);
        let _ = sock.flush();
    });
    let mut client = IndieClient::connect(addr).expect("connect to rogue 2");
    match client.next_event(DEFAULT_TIMEOUT) {
        Err(ClientErr::Protocol(msg)) => {
            assert!(msg.contains("not Content-Length framed"), "{msg}")
        }
        other => panic!("unframed stream must be rejected, got {other:?}"),
    }
}

#[test]
fn interop_adv_rogue_unknown_kinds_classified_never_guessed() {
    let addr = rogue_server(|mut sock| {
        let mut write = |method: &str, params: &J| {
            let bytes = encode_frame(method, J::Null, params);
            let _ = sock.write_all(&bytes);
            let _ = sock.flush();
            std::thread::sleep(Duration::from_millis(20));
        };
        write(
            "session/update",
            &jobj(&[
                ("sessionId", jstr("rogue")),
                (
                    "update",
                    jobj(&[
                        ("kind", jstr("agentStateChanged")),
                        ("agentState", jobj(&[("status", jstr("busy"))])),
                    ]),
                ),
            ]),
        );
        write("zed/custom", &jobj(&[("anything", J::I(1))]));
        write(
            "session/update",
            &jobj(&[
                ("sessionId", jstr("rogue")),
                (
                    "update",
                    jobj(&[("sessionUpdate", jstr("some_future_kind"))]),
                ),
            ]),
        );
    });
    let mut client = IndieClient::connect(addr).expect("connect to rogue");
    let mut unknown_kinds = 0usize;
    loop {
        match client.next_event(DEFAULT_TIMEOUT) {
            Ok(Ev::Unknown(desc)) => {
                unknown_kinds += 1;
                assert!(!desc.is_empty(), "unknown-kind description must exist");
            }
            Ok(other) => panic!("rogue frames must classify as unknown, got {other:?}"),
            Err(err) => {
                assert!(
                    err.is_close() || matches!(err, ClientErr::Timeout),
                    "{err:?}"
                );
                break;
            }
        }
    }
    assert_eq!(
        unknown_kinds, 3,
        "every hostile frame counted, none guessed"
    );
}

#[test]
fn interop_adv_client_decoder_corpus_never_panics() {
    let mut corpus: Vec<Vec<u8>> = vec![
        b"".to_vec(),
        b"Content-Length: 0\r\n\r\n".to_vec(),
        b"Content-Length: -5\r\n\r\n".to_vec(),
        b"Content-Length: abc\r\n\r\n".to_vec(),
        b"Content-Length: 99999999999999999999999\r\n\r\n".to_vec(),
        b"no header terminator here at all.........".to_vec(),
        vec![b'x'; MAX_HEADER_BYTES + 1],
        b"Content-Length: 5\r\n\r\nhel".to_vec(),
        b"Content-Length: 2\r\nContent-Length: 2\r\n\r\n{}".to_vec(),
        b"X-Powered-By: acp\r\n\r\n{}".to_vec(),
        b"Content-Length: 8\r\n\r\n{\"ok\":1}".to_vec(),
        b"Content-Length: 3\r\n\r\n   ".to_vec(),
        b"Content-Length: 5\r\n\r\n42abc".to_vec(),
        b"Content-Length: 1\r\n\r\n\"".to_vec(),
        b"Content-Length: 9\r\n\r\n{\"a\":1e999}".to_vec(),
        b"Content-Length: 12\r\n\r\n{\"a\":[1,2,3]}".to_vec(),
        b"Content-Length: 14\r\n\r\n\"\\ud800\\udc00\"".to_vec(),
        b"Content-Length: 9\r\n\r\n\"\\ud800\"".to_vec(),
        b"Content-Length: 8\r\n\r\n\"\\u0001\"".to_vec(),
        b"Content-Length: 8\r\n\r\n\"\\x41\"".to_vec(),
        b"Content-Length: 10\r\n\r\n{\"a\": 01}".to_vec(),
        b"Content-Length: 7\r\n\r\n{\"a\":+1}".to_vec(),
        b"Content-Length: 7\r\n\r\n[1,2,3]".to_vec(),
        b"Content-Length: 2\r\n\r\n{}".to_vec(),
        b"Content-Length: 5\r\n\r\n1.5.5".to_vec(),
        b"Content-Length: 6\r\n\r\n\"\\u12\"".to_vec(),
        b"Content-Length: 8\r\n\r\n[1,2,3,".to_vec(),
    ];
    let deep = format!("{}0{}", "[".repeat(5000), "]".repeat(5000));
    corpus.push(format!("Content-Length: {}\r\n\r\n{}", deep.len(), deep).into_bytes());
    for bytes in &corpus {
        let _ = decode_one(bytes);
        let _ = parse_json(std::str::from_utf8(bytes).unwrap_or(""));
    }
}

#[test]
fn interop_adv_fragmented_request_over_real_tcp() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = connect_and_init(harness.addr);
    let session = rpc_new_session(&mut client);

    let id = client.next_id;
    client.next_id += 1;
    client
        .pending
        .push((id, Method::Prompt, Some(session.clone())));
    let bytes = encode_frame(
        "session/prompt",
        J::I(id as i64),
        &prompt_params(&session, "chunks:3:frag:0"),
    );
    let mut step = 1usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let take = (i + step).min(bytes.len());
        client.send_raw(&bytes[i..take]).expect("dribble");
        std::thread::sleep(Duration::from_millis(1));
        i = take;
        step = step % 3 + 1;
    }
    let mut got = 0u64;
    loop {
        let ev = expect_ok(client.next_event(DEFAULT_TIMEOUT), "fragmented stream");
        match &ev {
            Ev::Chunk { session: s, .. } if s == &session => {
                expect_seq(&ev, "fragmented prompt", &session, "frag", got);
                got += 1;
            }
            Ev::PromptEnd { session: s, .. } if s == &session => {
                expect_stop(&ev, &session, "end_turn");
                break;
            }
            other => panic!("unexpected frame: {other:?}"),
        }
    }
    assert_eq!(got, 3);
    client.assert_no_unknown("fragmented request");
}

#[test]
fn interop_adv_idle_close_server_side_is_a_clean_eof() {
    let harness = Harness::start(FakeAgent::new());
    let mut client = connect_and_init(harness.addr);
    let session = rpc_new_session(&mut client);
    prompt_and_drain(&mut client, &session, "chunks:2:bye:0", "bye", 2);
    quiesce(&mut client);
    drop(harness);
    match client.next_event(DEFAULT_TIMEOUT) {
        Err(err) => assert!(
            err.is_close(),
            "server shutdown must be a clean close: {err:?}"
        ),
        Ok(other) => panic!("expected close after the server went away, got {other:?}"),
    }
}
