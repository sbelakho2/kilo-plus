//! Text-delta coalescing for the global event stream.
//!
//! The durable journal carries one `ModelChunkReceived` event per provider
//! chunk; the wire contract coalesces consecutive text deltas into
//! `session_next_text_delta` frames with a 50ms / 8KB aggregation window so
//! the frontend never sees a frame per token. The journal itself is not
//! changed (separate workstream): this struct sits between the journal read
//! and the SSE push.

use std::collections::HashMap;

use kilop_protocol::v756::GlobalEventPayload;

/// Aggregates per-session text deltas into `SessionNextTextDelta` payloads.
///
/// - A burst of chunks arriving within `window_ms` of the bucket's first
///   chunk merges into one frame.
/// - A delta that pushes the bucket past `max_bytes` flushes the bucket
///   first; a single delta at/over `max_bytes` flushes immediately.
/// - `flush`/`flush_all` never lose the tail: whatever was pushed is emitted
///   exactly once.
#[derive(Debug, Clone)]
pub struct DeltaCoalescer {
    window_ms: u64,
    max_bytes: usize,
    buckets: HashMap<String, Bucket>,
}

#[derive(Debug, Clone)]
struct Bucket {
    text: String,
    first_ms: u64,
    last_ms: u64,
}

impl DeltaCoalescer {
    /// `window_ms` — the aggregation window; `max_bytes` — the per-frame cap.
    pub fn new(window_ms: u64, max_bytes: usize) -> Self {
        assert!(window_ms > 0, "window must be positive");
        assert!(max_bytes > 0, "max_bytes must be positive");
        Self {
            window_ms,
            max_bytes,
            buckets: HashMap::new(),
        }
    }

    pub fn window_ms(&self) -> u64 {
        self.window_ms
    }

    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// True when no session has a pending bucket.
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// Total pending bytes across all sessions (bounds check for callers).
    pub fn pending_bytes(&self) -> usize {
        self.buckets.values().map(|b| b.text.len()).sum()
    }

    /// Push one text delta. Returns any payloads that must be emitted *now*
    /// (a previous bucket expired, or a size boundary was crossed). The
    /// pushed delta itself is never returned before its window closes unless
    /// a size boundary forces it.
    pub fn push(&mut self, session_id: &str, delta: &str, now_ms: u64) -> Vec<GlobalEventPayload> {
        let mut out = Vec::new();
        if delta.is_empty() {
            return out;
        }
        // Window expired since the bucket's first chunk: flush it first so
        // bursts within the window stay merged and nothing crosses windows.
        let expired = self
            .buckets
            .get(session_id)
            .is_some_and(|b| now_ms.saturating_sub(b.first_ms) >= self.window_ms);
        if expired {
            if let Some(f) = self.flush(session_id) {
                out.push(f);
            }
        }
        // Size boundary: the accumulated frame would exceed the cap.
        let pending = self
            .buckets
            .get(session_id)
            .map(|b| b.text.len())
            .unwrap_or(0);
        if pending.saturating_add(delta.len()) > self.max_bytes {
            if let Some(f) = self.flush(session_id) {
                out.push(f);
            }
        }
        let size = {
            let bucket = self
                .buckets
                .entry(session_id.to_string())
                .or_insert_with(|| Bucket {
                    text: String::new(),
                    first_ms: now_ms,
                    last_ms: now_ms,
                });
            bucket.text.push_str(delta);
            bucket.last_ms = now_ms;
            bucket.text.len()
        };
        // A single delta at/over the cap flushes immediately.
        if size >= self.max_bytes {
            if let Some(f) = self.flush(session_id) {
                out.push(f);
            }
        }
        out
    }

    /// Flush one session's pending bucket. Returns the frame, if any.
    pub fn flush(&mut self, session_id: &str) -> Option<GlobalEventPayload> {
        let bucket = self.buckets.remove(session_id)?;
        if bucket.text.is_empty() {
            return None;
        }
        Some(GlobalEventPayload::SessionNextTextDelta {
            session_id: session_id.to_string(),
            delta: bucket.text,
        })
    }

    /// Flush every pending bucket (never loses the tail).
    pub fn flush_all(&mut self) -> Vec<GlobalEventPayload> {
        let ids: Vec<String> = self.buckets.keys().cloned().collect();
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(f) = self.flush(&id) {
                out.push(f);
            }
        }
        out
    }

    /// Flush buckets whose last chunk is older than the window — the safety
    /// net that guarantees a quiet tail is emitted (never lost).
    pub fn flush_stale(&mut self, now_ms: u64) -> Vec<GlobalEventPayload> {
        let stale: Vec<String> = self
            .buckets
            .iter()
            .filter(|(_, b)| now_ms.saturating_sub(b.last_ms) >= self.window_ms)
            .map(|(id, _)| id.clone())
            .collect();
        let mut out = Vec::with_capacity(stale.len());
        for id in stale {
            if let Some(f) = self.flush(&id) {
                out.push(f);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_protocol::v756::GlobalEventPayload;

    fn delta_text(p: &GlobalEventPayload) -> String {
        match p {
            GlobalEventPayload::SessionNextTextDelta { delta, .. } => delta.clone(),
            other => panic!("expected text delta, got {other:?}"),
        }
    }

    #[test]
    fn burst_within_window_merges_into_one_frame() {
        let mut c = DeltaCoalescer::new(50, 8192);
        // Chunks at 10/20/30ms: all inside the 50ms window.
        assert!(c.push("s1", "a", 10).is_empty());
        assert!(c.push("s1", "b", 20).is_empty());
        assert!(c.push("s1", "c", 30).is_empty());
        assert!(!c.is_empty(), "bucket is pending");
        // A push past the window flushes the merged burst first.
        let frames = c.push("s1", "d", 70);
        assert_eq!(frames.len(), 1);
        assert_eq!(delta_text(&frames[0]), "abc");
        assert!(!c.is_empty(), "d is pending in the new bucket");
        // The tail (d) is not lost.
        let frames = c.flush("s1");
        assert_eq!(delta_text(&frames.unwrap()), "d");
        assert!(c.is_empty());
    }

    #[test]
    fn window_boundary_is_inclusive() {
        let mut c = DeltaCoalescer::new(50, 8192);
        c.push("s", "a", 0);
        // Just inside the window: merges.
        assert!(c.push("s", "b", 49).is_empty());
        // Exactly at window_ms from the first chunk: the bucket closes and
        // emits "ab" before "c" starts a fresh bucket.
        let frames = c.push("s", "c", 50);
        assert_eq!(frames.len(), 1);
        assert_eq!(delta_text(&frames[0]), "ab");
        // The tail is not lost.
        assert_eq!(delta_text(&c.flush("s").unwrap()), "c");
        assert!(c.is_empty());
    }

    #[test]
    fn large_chunk_flushes_immediately() {
        let mut c = DeltaCoalescer::new(50, 8);
        // A single chunk at/over the cap is its own frame right away.
        let frames = c.push("s", "abcdefgh", 0);
        assert_eq!(frames.len(), 1);
        assert_eq!(delta_text(&frames[0]), "abcdefgh");
        assert!(c.is_empty(), "nothing may be retained past a forced flush");
        // Over the cap: same behavior.
        let frames = c.push("s", "123456789", 1);
        assert_eq!(frames.len(), 1);
        assert_eq!(delta_text(&frames[0]), "123456789");
    }

    #[test]
    fn size_boundary_flushes_bucket_first() {
        let mut c = DeltaCoalescer::new(50, 10);
        c.push("s", "abcde", 0); // 5 bytes pending
                                 // Pushing 6 more would exceed 10: flush "abcde", then start fresh.
        let frames = c.push("s", "fghijk", 1);
        assert_eq!(frames.len(), 1);
        assert_eq!(delta_text(&frames[0]), "abcde");
        // The new chunk is pending, not lost.
        let frames = c.flush("s");
        assert_eq!(delta_text(&frames.unwrap()), "fghijk");
    }

    #[test]
    fn tail_is_never_lost_across_flushes() {
        let mut c = DeltaCoalescer::new(50, 8192);
        c.push("s1", "x", 0);
        c.push("s2", "y", 0);
        c.push("s1", "z", 1);
        let frames = c.flush_all();
        assert_eq!(frames.len(), 2);
        let mut texts: Vec<String> = frames.iter().map(delta_text).collect();
        texts.sort();
        assert_eq!(texts, vec!["xz", "y"]);
        assert!(c.is_empty());
        // Pushing after a flush starts a fresh bucket.
        assert!(c.push("s1", "w", 2).is_empty());
        assert_eq!(delta_text(&c.flush("s1").unwrap()), "w");
        assert!(c.is_empty());
    }

    #[test]
    fn flush_stale_emits_quiet_tails() {
        let mut c = DeltaCoalescer::new(50, 8192);
        c.push("s", "tail", 100);
        // Nothing stale at +49ms.
        assert!(c.flush_stale(149).is_empty());
        // At +50ms the quiet tail is emitted.
        let frames = c.flush_stale(150);
        assert_eq!(frames.len(), 1);
        assert_eq!(delta_text(&frames[0]), "tail");
        assert!(c.is_empty());
    }

    #[test]
    fn sessions_are_independent_and_ordered() {
        let mut c = DeltaCoalescer::new(50, 8192);
        c.push("s1", "a", 0);
        c.push("s2", "A", 0);
        // s1 expires while s2 stays fresh.
        let frames = c.push("s1", "b", 60);
        assert_eq!(frames.len(), 1);
        assert_eq!(delta_text(&frames[0]), "a");
        assert!(!c.is_empty());
        let frames = c.flush_all();
        assert_eq!(frames.len(), 2, "both sessions flush independently");
    }

    #[test]
    fn empty_deltas_are_ignored() {
        let mut c = DeltaCoalescer::new(50, 8192);
        assert!(c.push("s", "", 0).is_empty());
        assert!(c.push("s", "", 0).is_empty());
        assert!(c.is_empty(), "empty deltas must not create buckets");
        assert_eq!(c.pending_bytes(), 0);
        assert!(c.flush("s").is_none());
        // Empty deltas never emit frames even under size pressure.
        assert!(c.push("s", "", u64::MAX).is_empty());
    }

    #[test]
    fn oversized_stream_is_bounded_frame_by_frame() {
        // Even a pathological infinite stream stays bounded: every frame is
        // ≤ max_bytes and nothing accumulates unboundedly.
        let mut c = DeltaCoalescer::new(50, 1024);
        let mut emitted = 0usize;
        for i in 0..1000u64 {
            emitted += c.push("s", &"z".repeat(700), i).len();
        }
        let tail = c.flush_all().len();
        assert!(
            emitted + tail >= 1000,
            "every chunk is delivered exactly once"
        );
        assert!(c.is_empty());
        assert_eq!(c.pending_bytes(), 0);
        // Each emitted frame respects the cap.
        let mut c = DeltaCoalescer::new(50, 1024);
        for i in 0..50u64 {
            for f in c.push("s", &"q".repeat(500), i) {
                assert!(delta_text(&f).len() <= 1024);
            }
        }
        for f in c.flush_all() {
            assert!(delta_text(&f).len() <= 1024);
        }
    }

    #[test]
    fn zero_or_negative_config_is_rejected_loudly() {
        assert!(std::panic::catch_unwind(|| DeltaCoalescer::new(0, 8)).is_err());
        assert!(std::panic::catch_unwind(|| DeltaCoalescer::new(50, 0)).is_err());
    }
}
