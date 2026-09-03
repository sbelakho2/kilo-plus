//! Shared HTTP-stream transport hardening for the OpenAI-shaped adapters
//! (openai, deepseek, gateway, ollama, anthropic, google).
//!
//! All adapters consumed HTTP chunked bodies with per-chunk
//! `String::from_utf8_lossy(bytes).lines()`. That is wrong on two counts:
//!
//! 1. A line (or a multibyte UTF-8 rune) split across two network chunks was
//!    silently corrupted — the two fragments parsed as garbage lines and
//!    produced spurious `Malformed` errors against well-behaved servers.
//! 2. There was no cap on a line: a hostile server dribbling one giant line
//!    made the adapter buffer without bound.
//!
//! [`utf8_line_stream`] buffers RAW BYTES until `\n`, decodes each complete
//! line once, strips the trailing `\r` for `\r\n` frames, and bounds the
//! pending buffer at `max_line_bytes` (a breach is a loud `Malformed`
//! error and the stream terminates — memory stays bounded by construction).
//! A final unterminated line still yields (servers may omit the trailing
//! newline), preserving the old `.lines()` semantics.

use std::pin::Pin;

use futures::{Stream, StreamExt};

use crate::{ProviderError, ProviderErrorKind};

/// Hard cap on one SSE/NDJSON text line (a tool-call JSON line is the
/// largest legitimate frame; prompts are bounded at 512 KiB by session
/// bounds, so a megabyte is a generous wire ceiling).
pub const MAX_LINE_BYTES: usize = 1 << 20;

/// Turn an HTTP byte stream into complete UTF-8 lines.
///
/// - Lines are assembled at the byte level, so chunk boundaries never split
///   a line or a multibyte rune.
/// - `\r\n` and `\n` frames both work; empty lines are preserved (callers
///   already skip them) except that a bare `\r` line ending is stripped.
/// - When the unbroken buffer exceeds `max_line_bytes` the stream emits ONE
///   `Malformed` error and ends (bounded memory, loud failure).
/// - Transport errors surface as `Network` errors; the response never
///   decodes line-by-line twice.
pub fn utf8_line_stream<S, B, E>(
    bytes: S,
    max_line_bytes: usize,
) -> impl Stream<Item = Result<String, ProviderError>> + Send
where
    S: Stream<Item = Result<B, E>> + Send + 'static,
    B: AsRef<[u8]> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    enum Buf<S> {
        Active(Pin<Box<S>>, Vec<u8>),
        Dead,
    }

    // Pin<Box<S>> is Unpin even when S is not, so StreamExt::next works.
    futures::stream::unfold(
        Buf::Active(Box::pin(bytes), Vec::<u8>::new()),
        move |state| async move {
            let Buf::Active(mut bytes, mut pending) = state else {
                return None;
            };
            loop {
                // 1. Emit complete lines first — a single chunk may carry many.
                if let Some(break_at) = pending.iter().position(|b| *b == b'\n') {
                    let mut line: Vec<u8> = pending.drain(..=break_at).collect();
                    line.pop(); // '\n'
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    return Some((
                        Ok(String::from_utf8_lossy(&line).into_owned()),
                        Buf::Active(bytes, pending),
                    ));
                }
                // 2. Only the INCOMPLETE tail sits in the buffer here: enforce
                //    the cap before reading more (bounded memory by
                //    construction — a hostile giant line is a loud error and
                //    the stream terminates).
                if pending.len() > max_line_bytes {
                    return Some((
                        Err(ProviderError::new(
                            ProviderErrorKind::Malformed,
                            format!("SSE/NDJSON line exceeds {max_line_bytes} bytes"),
                        )),
                        Buf::Dead,
                    ));
                }
                // 3. Read the next network chunk.
                match bytes.next().await {
                    Some(Ok(chunk)) => {
                        pending.extend_from_slice(chunk.as_ref());
                        // Loop back to the split branch.
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(ProviderError::new(
                                ProviderErrorKind::Network,
                                format!("{e}"),
                            )),
                            Buf::Dead,
                        ));
                    }
                    None => {
                        if !pending.is_empty() {
                            let line = std::mem::take(&mut pending);
                            return Some((
                                Ok(String::from_utf8_lossy(&line).into_owned()),
                                Buf::Dead,
                            ));
                        }
                        return None;
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(
        chunks: Vec<Result<Vec<u8>, String>>,
        max_line: usize,
    ) -> Vec<Result<String, ProviderError>> {
        let stream = futures::stream::iter(chunks);
        let mut lines = Box::pin(utf8_line_stream(stream, max_line));
        let mut out = Vec::new();
        while let Some(item) = futures::executor::block_on(lines.next()) {
            out.push(item);
        }
        out
    }

    fn ok(chunks: Vec<&str>, max_line: usize) -> Vec<String> {
        collect(
            chunks
                .into_iter()
                .map(|c| Ok(c.as_bytes().to_vec()))
                .collect(),
            max_line,
        )
        .into_iter()
        .map(|r| r.unwrap())
        .collect()
    }

    #[test]
    fn line_split_across_chunks_assembles() {
        // The old per-chunk .lines() corrupted this: two fragments of one
        // line parsed as garbage. Byte-level buffering must reassemble.
        let lines = ok(vec!["data: {\"a\":\"part", "ial\"}\n", "data: b\n"], 4096);
        assert_eq!(lines, vec!["data: {\"a\":\"partial\"}", "data: b"]);
    }

    #[test]
    fn crlf_frames_strip_carriage_return() {
        let lines = ok(vec!["data: x\r\ndata: y\r\n"], 4096);
        assert_eq!(lines, vec!["data: x", "data: y"]);
    }

    #[test]
    fn multibyte_rune_split_across_chunks_survives() {
        // "héllo" split between 'é' (2 bytes) halves: fragments are invalid
        // UTF-8 on their own; buffering decodes the joined line correctly.
        let e_utf8 = "é".as_bytes();
        let mut c1 = b"data: h".to_vec();
        c1.extend_from_slice(&e_utf8[..1]);
        let mut c2 = e_utf8[1..].to_vec();
        c2.extend_from_slice(b"llo world\n");
        let lines: Vec<String> = collect(vec![Ok(c1), Ok(c2)], 4096)
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(lines, vec!["data: héllo world"]);
    }

    #[test]
    fn final_unterminated_line_is_still_delivered() {
        let lines = ok(vec!["data: tail-without-newline"], 4096);
        assert_eq!(lines, vec!["data: tail-without-newline"]);
    }

    #[test]
    fn empty_lines_preserved() {
        let lines = ok(vec!["a\n\nb\n"], 4096);
        assert_eq!(lines, vec!["a", "", "b"]);
    }

    #[test]
    fn oversized_line_is_loud_and_stream_terminates() {
        let mut big = Vec::new();
        big.extend_from_slice(b"data: ");
        big.extend(std::iter::repeat_n(b'x', 3000));
        let mut results = collect(vec![Ok(big)], 2048);
        let err = results.remove(0).unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::Malformed);
        assert!(err.message.contains("exceeds"));
        assert!(results.is_empty(), "stream must terminate after the error");
    }

    #[test]
    fn transport_error_is_network_error() {
        let mut results = collect(vec![Err("connection reset".into())], 4096);
        let err = results.remove(0).unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::Network);
        assert!(err.retryable);
        assert!(results.is_empty());
    }

    #[test]
    fn many_lines_inside_one_chunk() {
        let mut body = String::new();
        for i in 0..500 {
            body.push_str(&format!("data: line {i}\n"));
        }
        let lines = ok(vec![&body], 4096);
        assert_eq!(lines.len(), 500);
        assert_eq!(lines[499], "data: line 499");
    }

    #[test]
    fn hostile_dribble_below_cap_never_hangs() {
        // 1-byte chunks each poll; many polls; total under the cap.
        let body = b"data: hello\n";
        let chunks: Vec<Result<Vec<u8>, String>> = body.iter().map(|b| Ok(vec![*b])).collect();
        let lines: Vec<String> = collect(chunks, 4096)
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(lines, vec!["data: hello"]);
    }
}

// ---------------------------------------------------------------- deadlines

/// Per-stream hang controls (audit round 9, P1 "provider calls can still
/// hang"): first-byte, stream-idle, and overall deadlines plus prompt
/// cancellation. Silence from an already-connected server can therefore
/// never hang a turn: the guard errors (retryable Timeout) at the limits.
#[derive(Debug, Clone, Copy)]
pub struct StreamDeadlines {
    /// Time from stream start to the first item.
    pub first_byte_ms: u64,
    /// Maximum silence between consecutive items (reset on genuine data).
    pub idle_ms: u64,
    /// Overall bound; 0 disables (the turn's own lifetime governs).
    pub overall_ms: u64,
}

impl Default for StreamDeadlines {
    fn default() -> Self {
        Self {
            first_byte_ms: 60_000,
            idle_ms: 90_000,
            overall_ms: 0,
        }
    }
}

/// Wrap an adapter's parsed LINE stream with the hang controls. The item
/// stream carries `Result<String, ProviderError>` (adapters parse
/// SSE/NDJSON lines into provider errors), so the guard emits the REAL
/// `Timeout`/`Cancelled` kinds the agent's state-aware retry understands.
pub fn guarded_lines<S>(
    lines: S,
    deadlines: StreamDeadlines,
    cancel: Option<faktor_core::cancellation::CancellationToken>,
) -> impl Stream<Item = Result<String, ProviderError>>
where
    S: Stream<Item = Result<String, ProviderError>> + Send + 'static,
{
    let started = std::time::Instant::now();
    let state = (Box::pin(lines), started, None::<std::time::Instant>);
    futures::stream::unfold(state, move |(mut inner, started, last_seen)| {
        let cancel = cancel.clone(); // CancellationToken is cheap Arc clone
        async move {
            loop {
                if let Some(ref cancel) = cancel {
                    if cancel.is_cancelled() {
                        return Some((
                            Err(ProviderError::new(
                                ProviderErrorKind::Cancelled,
                                "stream cancelled",
                            )),
                            (inner, started, last_seen),
                        ));
                    }
                }
                if deadlines.overall_ms > 0
                    && started.elapsed().as_millis() as u64 >= deadlines.overall_ms
                {
                    return Some((
                        Err(ProviderError::new(
                            ProviderErrorKind::Timeout,
                            format!(
                                "stream exceeded its {} ms overall deadline",
                                deadlines.overall_ms
                            ),
                        )),
                        (inner, started, last_seen),
                    ));
                }
                let anchor = last_seen.unwrap_or(started);
                let limit = if last_seen.is_some() {
                    deadlines.idle_ms
                } else {
                    deadlines.first_byte_ms
                };
                let waited = anchor.elapsed().as_millis() as u64;
                let remaining = limit.saturating_sub(waited);
                let mut ticker = tokio::time::interval(std::time::Duration::from_millis(250));
                tokio::select! {
                    item = inner.next() => {
                        match item {
                            Some(Ok(line)) => {
                                let now = std::time::Instant::now();
                                return Some((Ok(line), (inner, started, Some(now))));
                            }
                            Some(Err(e)) => return Some((Err(e), (inner, started, last_seen))),
                            None => return None,
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(remaining)) => {
                        return Some((
                            Err(ProviderError::new(
                                ProviderErrorKind::Timeout,
                                if last_seen.is_some() {
                                    "provider stream idle timeout"
                                } else {
                                    "no first byte from the provider stream"
                                },
                            )),
                            (inner, started, last_seen),
                        ));
                    }
                    _ = ticker.tick() => {}
                }
            }
        }
    })
}

#[cfg(test)]
mod deadline_tests {
    use super::*;
    use faktor_core::cancellation::CancellationToken;

    #[tokio::test]
    async fn fast_stream_passes_through_untouched() {
        let items: Vec<Result<String, ProviderError>> =
            vec![Ok("a".into()), Ok("b".into()), Ok("c".into())];
        let mut s = Box::pin(guarded_lines(
            futures::stream::iter(items),
            StreamDeadlines::default(),
            None,
        ));
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            out.push(item.expect("ok"));
        }
        assert_eq!(out, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn silent_stream_hits_first_byte_timeout() {
        let dl = StreamDeadlines {
            first_byte_ms: 150,
            idle_ms: 100,
            overall_ms: 0,
        };
        // Eternal silence (never ends, never yields): first-byte timeout.
        let silence = futures::stream::unfold((), |()| async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            Some((Ok::<String, ProviderError>("late".into()), ()))
        });
        let mut s = Box::pin(guarded_lines(silence, dl, None));
        let item = tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
            .await
            .expect("silent stream must terminate via first-byte timeout")
            .expect("an error item");
        let err = item.expect_err("must be an error");
        assert_eq!(err.kind, ProviderErrorKind::Timeout);
        assert!(err.retryable);
    }

    #[tokio::test]
    async fn dribble_with_big_gap_hits_idle_timeout() {
        let dl = StreamDeadlines {
            first_byte_ms: 1000,
            idle_ms: 120,
            overall_ms: 0,
        };
        // One item, then eternal silence (the stream never ends): the idle
        // deadline must fire after the window.
        let lines = futures::stream::unfold(true, |first| async move {
            if first {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                Some((Ok::<String, ProviderError>("first".into()), false))
            } else {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                Some((Ok::<String, ProviderError>("never".into()), false))
            }
        });
        let mut s = Box::pin(guarded_lines(lines, dl, None));
        let first = tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
            .await
            .expect("first item within first-byte window")
            .expect("stream alive")
            .expect("first line ok");
        assert_eq!(first, "first");
        let item = tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
            .await
            .expect("idle timeout must terminate")
            .expect("an error item");
        let err = item.expect_err("must be an error");
        assert_eq!(err.kind, ProviderErrorKind::Timeout);
        assert!(err.message.contains("idle"));
    }

    #[tokio::test]
    async fn slow_but_alive_stream_survives() {
        let dl = StreamDeadlines {
            first_byte_ms: 1000,
            idle_ms: 200,
            overall_ms: 0,
        };
        let lines = futures::stream::unfold(0u32, |i| async move {
            if i >= 4 {
                return None;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            Some((Ok::<String, ProviderError>(format!("item {i}")), i + 1))
        });
        let mut s = Box::pin(guarded_lines(lines, dl, None));
        let mut got = Vec::new();
        for _ in 0..4 {
            let item = tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
                .await
                .expect("slow stream must not time out")
                .expect("stream must not end early")
                .expect("line ok");
            got.push(item);
        }
        assert_eq!(got.len(), 4);
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(3), s.next())
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn cancellation_lands_promptly_during_silence() {
        let dl = StreamDeadlines {
            first_byte_ms: 60_000,
            idle_ms: 60_000,
            overall_ms: 0,
        };
        let cancel = CancellationToken::new();
        let silence = futures::stream::unfold((), |()| async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            Some((Ok::<String, ProviderError>("late".into()), ()))
        });
        let mut s = Box::pin(guarded_lines(silence, dl, Some(cancel.clone())));
        cancel.cancel();
        let item = tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
            .await
            .expect("cancellation must surface promptly")
            .expect("an error item");
        assert_eq!(
            item.expect_err("cancelled").kind,
            ProviderErrorKind::Cancelled
        );
    }

    #[tokio::test]
    async fn overall_deadline_bounds_slow_stream() {
        let dl = StreamDeadlines {
            first_byte_ms: 10_000,
            idle_ms: 500,
            overall_ms: 300,
        };
        let lines = futures::stream::unfold(0u32, |i| async move {
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
            Some((Ok::<String, ProviderError>(format!("item {i}")), i + 1))
        });
        let mut s = Box::pin(guarded_lines(lines, dl, None));
        let mut items = 0usize;
        loop {
            match tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
                .await
                .expect("overall deadline must terminate")
            {
                Some(Ok(_)) => items += 1,
                Some(Err(e)) => {
                    assert_eq!(e.kind, ProviderErrorKind::Timeout);
                    assert!(e.message.contains("overall"));
                    break;
                }
                None => panic!("stream must error at the overall deadline"),
            }
        }
        assert!(items >= 1);
    }
}
