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
//! line once (STRICTLY — invalid UTF-8 is a loud `Malformed` error, never a
//! lossy decode), strips the trailing `\r` for `\r\n` frames, and bounds
//! the pending buffer at `max_line_bytes` — a breach is a loud `Malformed`
//! error and the stream terminates (memory stays bounded by construction;
//! the cap is enforced BEFORE an over-long complete line is emitted, audit
//! round 16). A final unterminated line still yields (servers may omit the
//! trailing newline), preserving the old `.lines()` semantics.
//!
//! [`guarded_lines`] wraps the line stream with first-byte / idle / overall
//! deadlines and prompt cancellation. Cancellation is WAKE-DRIVEN via
//! [`faktor_core::cancellation::CancellationToken::cancelled`] — no timer
//! polling (audit round 14: a recreated 250 ms interval ticked immediately
//! and spun the loop); a terminal error moves the guard to a dead state so
//! exactly one error is emitted and the stream ends (audit round 17).

use std::pin::Pin;

use futures::{Stream, StreamExt};

use crate::{ProviderError, ProviderErrorKind};

/// Hard cap on one SSE/NDJSON text line (a tool-call JSON line is the
/// largest legitimate frame; prompts are bounded at 512 KiB by session
/// bounds, so a megabyte is a generous wire ceiling).
pub const MAX_LINE_BYTES: usize = 1 << 20;

/// Absolute ceiling on the overall stream deadline an adapter derives from
/// `RequestMeta::deadline_ms` (audit round 15). The meta deadline is the
/// operation's remaining budget; the stream must never outlive it, but a
/// misconfigured oversized value is clamped here — no stream is bounded by
/// more than an hour regardless of what the runtime wrote.
pub const PROVIDER_CEILING_MS: u64 = 3_600_000;

/// Turn an HTTP byte stream into complete UTF-8 lines.
///
/// - Lines are assembled at the byte level, so chunk boundaries never split
///   a line or a multibyte rune.
/// - `\r\n` and `\n` frames both work; empty lines are preserved (callers
///   already skip them) except that a bare `\r` line ending is stripped.
/// - Lines decode with strict UTF-8: invalid bytes are a loud `Malformed`
///   error and the stream terminates (never `from_utf8_lossy` on protocol
///   lines — replacement characters would silently corrupt SSE/NDJSON).
/// - A line longer than `max_line_bytes` is a loud `Malformed` error and
///   the stream ends, checked BEFORE an over-long complete line is emitted
///   (bounded memory, loud failure).
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

    fn malformed(msg: impl Into<String>) -> ProviderError {
        ProviderError::new(ProviderErrorKind::Malformed, msg)
    }

    // Pin<Box<S>> is Unpin even when S is not, so StreamExt::next works.
    futures::stream::unfold(
        Buf::Active(Box::pin(bytes), Vec::<u8>::new()),
        move |state| async move {
            let Buf::Active(mut bytes, mut pending) = state else {
                return None;
            };
            loop {
                // 1. Emit complete lines first — a single chunk may carry
                //    many. The cap is enforced HERE, before the drain
                //    (audit round 16): a complete line longer than the cap
                //    is a loud Malformed error, never an emitted giant
                //    line. `break_at` is the `\n` index, hence the line
                //    length without the terminator.
                if let Some(break_at) = pending.iter().position(|b| *b == b'\n') {
                    if break_at > max_line_bytes {
                        return Some((
                            Err(malformed(format!(
                                "SSE/NDJSON line exceeds {max_line_bytes} bytes"
                            ))),
                            Buf::Dead,
                        ));
                    }
                    let mut line: Vec<u8> = pending.drain(..=break_at).collect();
                    line.pop(); // '\n'
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    // Strict decode (audit round 16b): protocol lines never
                    // go through from_utf8_lossy — replacement characters
                    // would corrupt SSE/NDJSON silently.
                    let line = match String::from_utf8(line) {
                        Ok(line) => line,
                        Err(_) => {
                            return Some((
                                Err(malformed("invalid utf-8 in SSE/NDJSON line")),
                                Buf::Dead,
                            ));
                        }
                    };
                    return Some((Ok(line), Buf::Active(bytes, pending)));
                }
                // 2. Only the INCOMPLETE tail sits in the buffer here: enforce
                //    the cap before reading more (bounded memory by
                //    construction — a hostile giant line is a loud error and
                //    the stream terminates).
                if pending.len() > max_line_bytes {
                    return Some((
                        Err(malformed(format!(
                            "SSE/NDJSON line exceeds {max_line_bytes} bytes"
                        ))),
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
                            let line = match String::from_utf8(line) {
                                Ok(line) => line,
                                Err(_) => {
                                    return Some((
                                        Err(malformed("invalid utf-8 in SSE/NDJSON line")),
                                        Buf::Dead,
                                    ));
                                }
                            };
                            return Some((Ok(line), Buf::Dead));
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
    fn oversized_complete_line_with_newline_errors_before_emit() {
        // Audit round 16: the cap was only enforced on INCOMPLETE buffers,
        // so a single chunk carrying an over-long line PLUS its newline was
        // drained and emitted past the cap. The check must now fire before
        // the drain: exactly one Malformed, then the stream ends.
        let mut big = b"data: ".to_vec();
        big.extend(std::iter::repeat_n(b'x', 3 * 1024 * 1024));
        big.push(b'\n');
        let mut results = collect(vec![Ok(big)], 1 << 20);
        let err = results.remove(0).unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::Malformed);
        assert!(err.message.contains("exceeds"));
        assert!(results.is_empty(), "stream must terminate after the error");
    }

    #[test]
    fn line_exactly_at_cap_still_delivered() {
        // Boundary: a line of exactly `max_line_bytes` (+ '\n') is legal.
        let mut line = b"x".repeat(1 << 20);
        line.push(b'\n');
        line.push(b'y');
        line.push(b'\n');
        let results = collect(vec![Ok(line)], 1 << 20);
        let lines: Vec<String> = results.into_iter().map(|r| r.unwrap()).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].len(), 1 << 20);
        assert_eq!(lines[1], "y");
    }

    #[test]
    fn invalid_utf8_line_is_loud_and_stream_terminates() {
        // Audit round 16b: protocol lines never decode lossy — a lone 0xFF
        // after a valid prefix is a loud Malformed error, exactly once.
        let mut chunk = b"data: {\"text\":\"ok\"}\n".to_vec();
        chunk.extend_from_slice(b"data: valid-prefix\xff\n");
        let mut results = collect(vec![Ok(chunk)], 4096);
        assert_eq!(results.remove(0).unwrap(), "data: {\"text\":\"ok\"}");
        let err = results.remove(0).unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::Malformed);
        assert!(err.message.contains("utf-8"), "{}", err.message);
        assert!(results.is_empty(), "stream must terminate after the error");
    }

    #[test]
    fn invalid_utf8_in_final_unterminated_line_is_loud() {
        let mut chunk = b"data: ok\n".to_vec();
        chunk.extend_from_slice(b"tail \xff");
        let mut results = collect(vec![Ok(chunk)], 4096);
        assert_eq!(results.remove(0).unwrap(), "data: ok");
        let err = results.remove(0).unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::Malformed);
        assert!(err.message.contains("utf-8"), "{}", err.message);
        assert!(results.is_empty());
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
///
/// The guard is wake-driven (audit round 14): cancellation resolves through
/// [`faktor_core::cancellation::CancellationToken::cancelled`] (std waker
/// registration, not a timer poll) and deadline sleeps carry the exact
/// remaining time — there is NO periodic ticker, so the stream can never
/// busy-spin. After a terminal error (timeout, cancel, or a dead inner
/// stream) the guard moves to a dead state: exactly ONE error item is
/// emitted and every later poll yields `None` (audit round 17).
pub fn guarded_lines<S>(
    lines: S,
    deadlines: StreamDeadlines,
    cancel: Option<faktor_core::cancellation::CancellationToken>,
) -> impl Stream<Item = Result<String, ProviderError>>
where
    S: Stream<Item = Result<String, ProviderError>> + Send + 'static,
{
    enum Phase<S> {
        Live(Pin<Box<S>>, std::time::Instant, Option<std::time::Instant>),
        Dead,
    }

    let started = std::time::Instant::now();
    futures::stream::unfold(Phase::Live(Box::pin(lines), started, None), move |phase| {
        let cancel = cancel.clone(); // CancellationToken is cheap Arc clone
        async move {
            // Terminal errors move the guard to Dead (audit round 17): the
            // stream must not emit a second error — or keep serving lines —
            // after a timeout, a cancel, or a dead inner stream.
            let terminal = |err: ProviderError| Some((Err(err), Phase::Dead));
            let Phase::Live(mut inner, started, last_seen) = phase else {
                return None;
            };
            if let Some(ref cancel) = cancel {
                if cancel.is_cancelled() {
                    return terminal(ProviderError::new(
                        ProviderErrorKind::Cancelled,
                        "stream cancelled",
                    ));
                }
            }
            let elapsed_ms = started.elapsed().as_millis() as u64;
            if deadlines.overall_ms > 0 && elapsed_ms >= deadlines.overall_ms {
                return terminal(ProviderError::new(
                    ProviderErrorKind::Timeout,
                    format!(
                        "stream exceeded its {} ms overall deadline",
                        deadlines.overall_ms
                    ),
                ));
            }
            let anchor = last_seen.unwrap_or(started);
            let (limit_ms, phase_msg) = if last_seen.is_some() {
                (deadlines.idle_ms, "provider stream idle timeout")
            } else {
                (
                    deadlines.first_byte_ms,
                    "no first byte from the provider stream",
                )
            };
            let waited_ms = anchor.elapsed().as_millis() as u64;
            if waited_ms >= limit_ms {
                return terminal(ProviderError::new(
                    ProviderErrorKind::Timeout,
                    phase_msg.to_string(),
                ));
            }
            // One-shot waits for THIS poll, with the exact remaining time —
            // no ticker, no periodic polling (audit round 14). Cancellation
            // is wake-driven: [`CancellationToken::cancelled`] registers a
            // waker and `cancel()` wakes it. Each arm future is created
            // fresh per poll, so the remaining times are always anchored at
            // the stream/phase start.
            let cancel_wait = async {
                match cancel.as_ref() {
                    Some(c) => c.cancelled().await,
                    None => std::future::pending::<()>().await,
                }
            };
            let overall_wait = async {
                if deadlines.overall_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(
                        deadlines.overall_ms - elapsed_ms,
                    ))
                    .await
                } else {
                    std::future::pending::<()>().await
                }
            };
            let phase_wait =
                tokio::time::sleep(std::time::Duration::from_millis(limit_ms - waited_ms));
            tokio::select! {
                biased;
                _ = cancel_wait => {
                    terminal(ProviderError::new(
                        ProviderErrorKind::Cancelled,
                        "stream cancelled",
                    ))
                }
                _ = overall_wait => {
                    terminal(ProviderError::new(
                        ProviderErrorKind::Timeout,
                        format!(
                            "stream exceeded its {} ms overall deadline",
                            deadlines.overall_ms
                        ),
                    ))
                }
                _ = phase_wait => {
                    terminal(ProviderError::new(
                        ProviderErrorKind::Timeout,
                        phase_msg.to_string(),
                    ))
                }
                item = inner.next() => {
                    match item {
                        Some(Ok(line)) => {
                            let now = std::time::Instant::now();
                            Some((Ok(line), Phase::Live(inner, started, Some(now))))
                        }
                        Some(Err(e)) => terminal(e),
                        None => None,
                    }
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
    async fn already_cancelled_token_errors_immediately() {
        let dl = StreamDeadlines {
            first_byte_ms: 60_000,
            idle_ms: 60_000,
            overall_ms: 0,
        };
        let cancel = CancellationToken::new();
        cancel.cancel();
        let silence = futures::stream::unfold((), |()| async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            Some((Ok::<String, ProviderError>("late".into()), ()))
        });
        let mut s = Box::pin(guarded_lines(silence, dl, Some(cancel)));
        let item = tokio::time::timeout(std::time::Duration::from_secs(1), s.next())
            .await
            .expect("pre-cancelled token must error without any wait")
            .expect("an error item");
        assert_eq!(
            item.expect_err("cancelled").kind,
            ProviderErrorKind::Cancelled
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
        // Cancel while the guard is parked on the wake-driven cancel wait
        // (not before the first poll): the waker registration must surface
        // it immediately — no timer polling.
        let cancel_task = {
            let cancel = cancel.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_millis(80)).await;
                cancel.cancel();
            })
        };
        let t0 = std::time::Instant::now();
        let item = tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
            .await
            .expect("cancellation must surface promptly")
            .expect("an error item");
        assert_eq!(
            item.expect_err("cancelled").kind,
            ProviderErrorKind::Cancelled
        );
        assert!(
            t0.elapsed() < std::time::Duration::from_millis(1500),
            "cancel must wake the parked guard promptly: {:?}",
            t0.elapsed()
        );
        cancel_task.await.unwrap();
        // Terminal error => dead state: no further events.
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(1), s.next())
                .await
                .expect("stream must end after a terminal error")
                .is_none()
        );
    }

    #[tokio::test]
    async fn stream_ends_after_terminal_timeout_error() {
        // Audit round 17: after a terminal timeout the guard moves to a dead
        // state — exactly one error, then the stream ends (the old code
        // kept the live inner stream and could emit another error on the
        // next poll).
        let dl = StreamDeadlines {
            first_byte_ms: 60,
            idle_ms: 40,
            overall_ms: 0,
        };
        let silence = futures::stream::unfold((), |()| async move {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            Some((Ok::<String, ProviderError>("late".into()), ()))
        });
        let mut s = Box::pin(guarded_lines(silence, dl, None));
        let item = tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
            .await
            .expect("first-byte timeout must fire")
            .expect("an error item")
            .expect_err("must be a timeout");
        assert_eq!(item.kind, ProviderErrorKind::Timeout);
        // The guard is terminally dead: the next (and only next) poll ends
        // the stream — no repeated error, no lines from the live inner
        // stream. (A stream must not be polled again after `None`.)
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(300), s.next())
                .await
                .expect("stream must be terminally dead")
                .is_none(),
            "no further events after a terminal error"
        );
    }

    #[tokio::test]
    async fn terminal_error_from_inner_stream_ends_guard() {
        // An error the inner stream raises (network death) is emitted once,
        // then the guard ends — it must not keep polling the dead inner
        // stream and emit a second error.
        let dl = StreamDeadlines {
            first_byte_ms: 60_000,
            idle_ms: 60_000,
            overall_ms: 0,
        };
        let poisoned = futures::stream::iter(vec![
            Ok::<String, ProviderError>("a".into()),
            Err(ProviderError::new(
                ProviderErrorKind::Network,
                "connection reset",
            )),
            Ok::<String, ProviderError>("after-death".into()),
        ]);
        let mut s = Box::pin(guarded_lines(poisoned, dl, None));
        assert_eq!(
            tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap(),
            "a"
        );
        let err = tokio::time::timeout(std::time::Duration::from_secs(5), s.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert_eq!(err.kind, ProviderErrorKind::Network);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(300), s.next())
                .await
                .unwrap()
                .is_none(),
            "the guard must not surface items or errors past the terminal error"
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
