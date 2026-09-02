//! Testing harness: a scripted HTTP server that lets adapter tests assert
//! the exact wire shape (no internal option leakage), inject malformed
//! streams, rate limits, and mid-stream deaths.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// One captured request header, (lowercased name, value).
pub type MockHeader = (String, String);

/// One scripted response per (method, path) — or a closure for dynamic
/// behavior (asserting the request body).
#[derive(Clone)]
pub enum MockAction {
    /// Respond with this status + body (content-type json by default).
    Respond { status: u16, body: String },
    /// Read the request, run the assertion, then respond.
    AssertThenRespond {
        status: u16,
        body: String,
        assert: Arc<dyn Fn(&serde_json::Value) + Send + Sync>,
    },
    /// SSE stream: lines of `data: <json>` then `data: [DONE]`.
    Sse { status: u16, events: Vec<String> },
    /// Adversarial stream: each string is delivered as its OWN HTTP chunk,
    /// so boundaries can fall MID-LINE and MID-RUNE (the exact case the
    /// per-chunk `.lines()` bug corrupted). A test with a well-behaved
    /// server whose frames are fragmented this way MUST still reassemble.
    ChunkedSse {
        status: u16,
        /// Raw bytes per HTTP chunk: permits mid-rune and mid-line splits.
        chunks: Vec<Vec<u8>>,
    },
    /// One action per matching request, consumed in order (a provider may
    /// receive several distinct responses over one route — e.g. two tool
    /// responses whose ids must differ). When the list is exhausted the
    /// route behaves like an unrouted path (404), so an out-of-contract
    /// extra request is loud instead of silently repeating a stale script.
    Sequence { actions: Vec<MockAction> },
}

#[derive(Clone, Default)]
pub struct MockServer {
    routes: Arc<Mutex<HashMap<(String, String), MockAction>>>,
    /// Next unconsumed index per Sequence route (one index per method+path).
    sequence_next: Arc<Mutex<HashMap<(String, String), usize>>>,
    requests: Arc<Mutex<Vec<(String, String, String)>>>, // method, path, body
    /// Request headers, index-aligned with `requests` (lowercased names).
    request_headers: Arc<Mutex<Vec<Vec<MockHeader>>>>,
}

impl MockServer {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn route(&self, method: &str, path: &str, action: MockAction) {
        self.routes
            .lock()
            .unwrap()
            .insert((method.to_string(), path.to_string()), action);
    }

    pub fn requests(&self) -> Vec<(String, String, String)> {
        self.requests.lock().unwrap().clone()
    }

    pub fn last_request(&self) -> Option<(String, String, String)> {
        self.requests.lock().unwrap().last().cloned()
    }

    pub fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    /// Headers of the most recent request (lowercased names, insertion
    /// order). Header assertions live here so `requests()` keeps its frozen
    /// 3-tuple shape for existing callers.
    pub fn last_request_headers(&self) -> Vec<(String, String)> {
        self.request_headers
            .lock()
            .unwrap()
            .last()
            .cloned()
            .unwrap_or_default()
    }

    /// Bind and serve until the returned handle is dropped.
    pub async fn serve(self: &Arc<Self>) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let me = self.clone();
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let me = me.clone();
                tokio::spawn(async move {
                    let _ = handle_conn(&mut socket, &me).await;
                });
            }
        });
        (addr, handle)
    }

    /// Serve + build a base_url for reqwest.
    pub async fn base_url(self: &Arc<Self>) -> String {
        let (addr, _handle) = self.serve().await;
        format!("http://{addr}")
    }
}

async fn handle_conn(socket: &mut tokio::net::TcpStream, me: &MockServer) -> std::io::Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end;
    loop {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            header_end = pos + 4;
            break;
        }
        if buf.len() > 64 * 1024 {
            return Ok(());
        }
    }
    let header = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut lines = header.lines();
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split(' ');
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    // Query strings never participate in routing (the request URL carries
    // provider-specific query params like alt=sse).
    let path = path.split('?').next().unwrap_or(&path).to_string();
    let mut content_length = 0usize;
    let mut req_headers: Vec<MockHeader> = Vec::new();
    for line in lines {
        if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
        if let Some((name, value)) = line.split_once(':') {
            req_headers.push((name.trim().to_lowercase(), value.trim().to_string()));
        }
    }
    while buf.len() < header_end + content_length {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = String::from_utf8_lossy(
        &buf[header_end..header_end + content_length.min(buf.len() - header_end)],
    )
    .to_string();
    me.requests
        .lock()
        .unwrap()
        .push((method.clone(), path.clone(), body.clone()));
    me.request_headers.lock().unwrap().push(req_headers);

    let action = lookup_action(me, &method, &path);
    let action = match action {
        Some(a) => a,
        None => {
            let _ = socket
                .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
                .await;
            return Ok(());
        }
    };
    match action {
        MockAction::Respond { status, body: resp } => {
            // Connection: close — one scripted response per connection, so
            // a second request on the same client can never race a pooled
            // keep-alive connection the mock just closed (a real flake
            // source when tests script several sequential responses).
            let resp = format!(
                "HTTP/1.1 {status} X\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                resp.len(),
                resp
            );
            let _ = socket.write_all(resp.as_bytes()).await;
        }
        MockAction::AssertThenRespond {
            status,
            body: resp,
            assert,
        } => {
            if let Ok(json) = serde_json::from_str(&body) {
                assert(&json);
            }
            let resp = format!(
                "HTTP/1.1 {status} X\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                resp.len(),
                resp
            );
            let _ = socket.write_all(resp.as_bytes()).await;
        }
        MockAction::Sse { status, events } => {
            let mut out = format!("HTTP/1.1 {status} X\r\nConnection: close\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n");
            for ev in &events {
                // ev is a full SSE frame (data: ...\n\n); the chunk is the
                // size in hex, CRLF, the data, CRLF.
                out.push_str(&format!("{:x}\r\n{}\r\n", ev.len(), ev));
            }
            // Chunked terminator.
            out.push_str("0\r\n\r\n");
            let _ = socket.write_all(out.as_bytes()).await;
        }
        MockAction::ChunkedSse { status, chunks } => {
            let head = format!("HTTP/1.1 {status} X\r\nConnection: close\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n");
            let _ = socket.write_all(head.as_bytes()).await;
            for chunk in &chunks {
                // Every chunk is one HTTP chunk; flush between chunks so
                // reqwest observes the boundary (loopback may coalesce on
                // some platforms — the test must pass either way).
                let framed = format!("{:x}\r\n", chunk.len());
                let _ = socket.write_all(framed.as_bytes()).await;
                let _ = socket.write_all(chunk).await;
                let _ = socket.write_all(b"\r\n").await;
                let _ = socket.flush().await;
            }
            let _ = socket.write_all(b"0\r\n\r\n").await;
        }
        MockAction::Sequence { .. } => unreachable!("lookup_action unwraps nested Sequences"),
    }
    Ok(())
}

/// Resolve the action for one request. A `Sequence` pops its next action
/// (nested sequences are unwrapped, bounded so a self-referential script can
/// never spin forever); an exhausted sequence yields `None`, i.e. the route
/// behaves like an unrouted path (404).
fn lookup_action(me: &MockServer, method: &str, path: &str) -> Option<MockAction> {
    for _ in 0..16 {
        let routes = me.routes.lock().unwrap();
        let action = match routes.get(&(method.to_string(), path.to_string())) {
            Some(MockAction::Sequence { actions }) => {
                let mut nexts = me.sequence_next.lock().unwrap();
                let key = (method.to_string(), path.to_string());
                let idx = nexts.entry(key).or_insert(0);
                let cur = *idx;
                *idx = idx.saturating_add(1);
                actions.get(cur).cloned()
            }
            Some(other) => Some(other.clone()),
            None => None,
        };
        drop(routes);
        match action {
            Some(MockAction::Sequence { .. }) => continue,
            other => return other,
        }
    }
    None
}

/// Build an SSE body (non-chunked) for plain `Respond` testing.
pub fn sse_body(events: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for e in events {
        out.push_str(&format!("data: {}\n\n", serde_json::to_string(e).unwrap()));
    }
    out.push_str("data: [DONE]\n\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_records_and_responds() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::AssertThenRespond {
                status: 200,
                body: r#"{"ok":true}"#.into(),
                assert: Arc::new(|body: &serde_json::Value| {
                    assert_eq!(body["model"], "qwen3.8");
                    assert!(body["messages"].is_array());
                }),
            },
        );
        let base = server.base_url().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/api/chat"))
            .json(&serde_json::json!({"model": "qwen3.8", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(server.request_count(), 1);
        let (m, p, _b) = server.last_request().unwrap();
        assert_eq!(m, "POST");
        assert_eq!(p, "/api/chat");
    }

    #[tokio::test]
    async fn mock_sse_stream() {
        let server = MockServer::new();
        server.route(
            "POST",
            "/v1/chat",
            MockAction::Sse {
                status: 200,
                events: vec![
                    r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#.into(),
                    r#"data: {"choices":[{"delta":{"content":" there"}}]}"#.into(),
                ],
            },
        );
        let base = server.base_url().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/v1/chat"))
            .body("{}")
            .send()
            .await
            .unwrap();
        let text = resp.text().await.unwrap();
        assert!(text.contains("hi"));
        assert!(text.contains("there"));
    }

    #[tokio::test]
    async fn mock_404_for_unrouted() {
        let server = MockServer::new();
        let base = server.base_url().await;
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{base}/nope"))
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 404);
    }

    #[tokio::test]
    async fn mock_sequence_serves_actions_in_order_then_404() {
        // One route, two scripted responses consumed in order (the second
        // provider response of a test must differ from the first); an extra
        // request past the script is loud (404), never a stale repeat.
        let server = MockServer::new();
        server.route(
            "POST",
            "/api/chat",
            MockAction::Sequence {
                actions: vec![
                    MockAction::Respond {
                        status: 200,
                        body: r#"{"n":1}"#.into(),
                    },
                    MockAction::Respond {
                        status: 200,
                        body: r#"{"n":2}"#.into(),
                    },
                ],
            },
        );
        let base = server.base_url().await;
        let client = reqwest::Client::new();
        for n in ["1", "2"] {
            let resp = client
                .post(format!("{base}/api/chat"))
                .body("{}")
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            assert_eq!(resp.text().await.unwrap(), format!(r#"{{"n":{n}}}"#));
        }
        let resp = client
            .post(format!("{base}/api/chat"))
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            404,
            "an exhausted Sequence must behave like an unrouted path"
        );
    }
}
