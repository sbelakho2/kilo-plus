//! Testing harness: a scripted HTTP server that lets adapter tests assert
//! the exact wire shape (no internal option leakage), inject malformed
//! streams, rate limits, and mid-stream deaths.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
}

#[derive(Clone, Default)]
pub struct MockServer {
    routes: Arc<Mutex<HashMap<(String, String), MockAction>>>,
    requests: Arc<Mutex<Vec<(String, String, String)>>>, // method, path, body
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

async fn handle_conn(
    socket: &mut tokio::net::TcpStream,
    me: &MockServer,
) -> std::io::Result<()> {
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
    for line in lines {
        if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }
    while buf.len() < header_end + content_length {
        let n = socket.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = String::from_utf8_lossy(&buf[header_end..header_end + content_length.min(buf.len() - header_end)]).to_string();
    me.requests.lock().unwrap().push((method.clone(), path.clone(), body.clone()));

    let action = me.routes.lock().unwrap().get(&(method, path)).cloned();
    let action = match action {
        Some(a) => a,
        None => {
            let _ = socket.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n").await;
            return Ok(());
        }
    };
    match action {
        MockAction::Respond { status, body: resp } => {
            let resp = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                resp.len(),
                resp
            );
            let _ = socket.write_all(resp.as_bytes()).await;
        }
        MockAction::AssertThenRespond { status, body: resp, assert } => {
            if let Ok(json) = serde_json::from_str(&body) {
                assert(&json);
            }
            let resp = format!(
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                resp.len(),
                resp
            );
            let _ = socket.write_all(resp.as_bytes()).await;
        }
        MockAction::Sse { status, events } => {
            let mut out = format!("HTTP/1.1 {status} X\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n");
            for ev in &events {
                // ev is a full SSE frame (data: ...\n\n); the chunk is the
                // size in hex, CRLF, the data, CRLF.
                out.push_str(&format!("{:x}\r\n{}\r\n", ev.len(), ev));
            }
            // Chunked terminator.
            out.push_str("0\r\n\r\n");
            let _ = socket.write_all(out.as_bytes()).await;
        }
    }
    Ok(())
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
}
