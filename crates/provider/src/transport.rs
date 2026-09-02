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
