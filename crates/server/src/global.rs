//! The global event bus: the `/global/event` stream source.
//!
//! Sessions keep per-session journals; the real v7.5.6 client subscribes to
//! one global stream with a single `after` cursor. The bus projects every
//! session's journal (in deterministic session-id order, so the global
//! sequence is append-only), coalesces text deltas (see `DeltaCoalescer`),
//! and serves the last `RING_CAPACITY` frames from a bounded ring. The
//! journals are the source of truth; the ring is a bounded replay cache.
//!
//! Polling is lazy: the bus is polled by the connected SSE streams (the
//! same pattern as the per-session journal stream), so there is no
//! background task and no unbounded resource lifetime.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use faktor_core::event::{Event, EventKind};
use faktor_protocol::v756::{GlobalEvent, GlobalEventPayload};
use faktor_session::{SessionHandle, SessionManager};

use crate::coalesce::DeltaCoalescer;

/// Default replay depth of the ring (bounded everything).
pub const RING_CAPACITY: usize = 4096;

/// The aggregation window for text deltas (frozen wire behavior).
pub const DELTA_WINDOW_MS: u64 = 50;
/// Per-frame cap for coalesced text deltas.
pub const DELTA_MAX_BYTES: usize = 8 * 1024;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub struct GlobalEventBus {
    session: Arc<SessionManager>,
    directory: Option<String>,
    capacity: usize,
    next_id: AtomicU64,
    state: Mutex<BusState>,
}

struct BusState {
    ring: VecDeque<(u64, GlobalEvent)>,
    cursors: HashMap<u64, u64>,
    text_len: HashMap<(u64, i64), usize>,
    coalescer: DeltaCoalescer,
}

impl GlobalEventBus {
    pub fn new(session: Arc<SessionManager>, directory: Option<String>) -> Self {
        Self::with_capacity(session, directory, RING_CAPACITY)
    }

    pub fn with_capacity(
        session: Arc<SessionManager>,
        directory: Option<String>,
        capacity: usize,
    ) -> Self {
        assert!(capacity > 0, "ring capacity must be positive");
        Self {
            session,
            directory,
            capacity,
            next_id: AtomicU64::new(0),
            state: Mutex::new(BusState {
                ring: VecDeque::new(),
                cursors: HashMap::new(),
                text_len: HashMap::new(),
                coalescer: DeltaCoalescer::new(DELTA_WINDOW_MS, DELTA_MAX_BYTES),
            }),
        }
    }

    /// Scan every session's journal from its last seen cursor, project onto
    /// the global envelope, coalesce text deltas, and append to the ring.
    /// Idempotent: cursors make re-polling a no-op for unchanged journals.
    pub fn poll_once(&self) {
        let mut st = self.state.lock().unwrap();
        let mut rows = self.session.list_sessions(None).unwrap_or_default();
        // Deterministic global order: sessions by ascending id (new sessions
        // sort after existing ones, so the global sequence is append-only).
        rows.sort_by_key(|r| r.id().raw());
        for row in rows {
            let sid = row.id();
            let last = st.cursors.get(&sid.raw()).copied().unwrap_or(0);
            let Ok(Some(handle)) = self.session.get_session(sid) else {
                continue;
            };
            let Ok(events) = handle.events_range(last.saturating_add(1), None) else {
                continue;
            };
            for e in events {
                self.project(&mut st, &e);
                st.cursors.insert(sid.raw(), e.seq.raw());
            }
            // The current journal's chunk events carry only `text_len`; the
            // actual text lands in the message parts table. Re-diff the
            // streamed messages every poll so recovered deltas are emitted
            // even when no chunk event arrives in the same window.
            self.recover_text(&mut st, &handle, sid.raw());
        }
        // Quiet tails are emitted one window after their last chunk.
        for payload in st.coalescer.flush_stale(now_ms()) {
            self.emit(&mut st, self.wrap(payload));
        }
    }

    /// Project one journal event: deltas go through the coalescer, everything
    /// else flushes the session's pending delta first (per-session ordering).
    fn project(&self, st: &mut BusState, e: &Event) {
        let sid = e.session_id.to_string();
        let ts = e.ts_ms.max(0) as u64;
        if e.kind == EventKind::ModelChunkReceived {
            match GlobalEvent::from_journal_event(e, self.directory.clone()) {
                Some(ge) => match &ge.payload {
                    GlobalEventPayload::SessionNextTextDelta { delta, .. } => {
                        for payload in st.coalescer.push(&sid, delta, ts) {
                            self.emit(st, self.wrap(payload));
                        }
                    }
                    _ => {
                        if let Some(payload) = st.coalescer.flush(&sid) {
                            self.emit(st, self.wrap(payload));
                        }
                        self.emit(st, ge);
                    }
                },
                None => {
                    // Text_len-only chunk: register the message so
                    // recover_text diffs its parts (the text is not in the
                    // journal payload).
                    if let Some(mid) = e
                        .payload
                        .as_ref()
                        .and_then(|p| p.get("message_id"))
                        .and_then(|v| v.as_i64())
                    {
                        st.text_len.entry((e.session_id.raw(), mid)).or_insert(0);
                    }
                }
            }
            return;
        }
        if let Some(payload) = st.coalescer.flush(&sid) {
            self.emit(st, self.wrap(payload));
        }
        if let Some(ge) = GlobalEvent::from_journal_event(e, self.directory.clone()) {
            self.emit(st, ge);
        }
    }

    /// Diff the durable text parts of every streamed message and feed the
    /// recovered bytes (never previously seen) to the coalescer.
    fn recover_text(&self, st: &mut BusState, handle: &SessionHandle, session_raw: u64) {
        let keys: Vec<i64> = st
            .text_len
            .keys()
            .filter(|(s, _)| *s == session_raw)
            .map(|(_, m)| *m)
            .collect();
        for mid in keys {
            let cumulative = handle
                .parts_of(mid)
                .ok()
                .map(|parts| {
                    parts
                        .iter()
                        .filter(|p| p.kind == "text")
                        .filter_map(|p| p.data.get("text").and_then(|v| v.as_str()))
                        .collect::<String>()
                })
                .unwrap_or_default();
            let key = (session_raw, mid);
            let prev = st.text_len.get(&key).copied().unwrap_or(0);
            st.text_len.insert(key, cumulative.len());
            if cumulative.len() >= prev {
                if cumulative.len() > prev {
                    let delta = &cumulative[prev..];
                    let sid = handle.id().to_string();
                    for payload in st.coalescer.push(&sid, delta, now_ms()) {
                        self.emit(st, self.wrap(payload));
                    }
                }
            } else {
                // The message was rewritten (shrunk): treat the full text as
                // the delta rather than lose it.
                if !cumulative.is_empty() {
                    let sid = handle.id().to_string();
                    for payload in st.coalescer.push(&sid, &cumulative, now_ms()) {
                        self.emit(st, self.wrap(payload));
                    }
                }
            }
        }
    }

    fn wrap(&self, payload: GlobalEventPayload) -> GlobalEvent {
        GlobalEvent {
            directory: self.directory.clone(),
            project: None,
            workspace: None,
            payload,
        }
    }

    fn emit(&self, st: &mut BusState, ge: GlobalEvent) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        st.ring.push_back((id, ge));
        while st.ring.len() > self.capacity {
            st.ring.pop_front();
        }
    }

    /// Push ONE live chunk frame onto the ring (audit round 11): the agent
    /// forwards streaming text/reasoning through [`ChunkEvent`]; this gives
    /// subscribers true low-latency `session.next.*.delta` frames instead of
    /// waiting for the journal re-diff window. Envelope/type fields mirror
    /// the journal-projected variants exactly.
    pub fn push_chunk(&self, chunk: faktor_agent::ChunkEvent) {
        let payload = match chunk.kind {
            "reasoning" => GlobalEventPayload::SessionNextReasoningDelta {
                session_id: chunk.session_id.to_string(),
                delta: chunk.text,
            },
            "tool" => GlobalEventPayload::SessionNextToolCalled {
                session_id: chunk.session_id.to_string(),
                tool: chunk.text,
            },
            _ => GlobalEventPayload::SessionNextTextDelta {
                session_id: chunk.session_id.to_string(),
                delta: chunk.text,
            },
        };
        let mut st = self.state.lock().unwrap();
        let ge = self.wrap(payload);
        self.emit(&mut st, ge);
    }

    /// The id of the newest emitted frame (0 when nothing was emitted).
    pub fn latest_id(&self) -> u64 {
        self.next_id.load(Ordering::Relaxed)
    }

    pub fn ring_len(&self) -> usize {
        self.state.lock().unwrap().ring.len()
    }

    /// All frames with id > `after` the ring can still serve. Values below
    /// the ring front are clamped: the client gets everything the ring has,
    /// never an error (oversized `after` simply returns nothing).
    pub fn frames_after(&self, after: u64) -> Vec<(u64, GlobalEvent)> {
        let st = self.state.lock().unwrap();
        st.ring
            .iter()
            .filter(|(id, _)| *id > after)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::id::OpId;
    use faktor_core::state::AgentState;

    fn manager(dir: &std::path::Path) -> Arc<SessionManager> {
        SessionManager::open(dir.join("store"), dir.join("cas"), true).unwrap()
    }

    fn session(m: &Arc<SessionManager>) -> SessionHandle {
        let ws = m.create_workspace("/w").unwrap();
        m.create_session(ws, "t", "fake", "m").unwrap()
    }

    fn to_streaming(s: &SessionHandle) {
        s.append_event(
            EventKind::PromptReceived,
            AgentState::Preparing,
            Some(OpId::new(7)),
            None,
        )
        .unwrap();
        s.append_event(
            EventKind::ContextPrepared,
            AgentState::BuildingContext,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            EventKind::ModelStarted,
            AgentState::WaitingForModel,
            None,
            None,
        )
        .unwrap();
    }

    fn chunk(s: &SessionHandle, payload: serde_json::Value) {
        s.append_event(
            EventKind::ModelChunkReceived,
            AgentState::Streaming,
            Some(OpId::new(7)),
            Some(payload),
        )
        .unwrap();
    }

    #[tokio::test]
    async fn projects_journal_into_global_envelope_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path());
        let s = session(&m);
        to_streaming(&s);
        // The legal completion chain: Streaming (chunk) → Validating →
        // UpdatingMemory → ReadyForNextTurn.
        chunk(&s, serde_json::json!({"message_id": 99, "text_len": 0}));
        s.append_event(
            EventKind::TurnCompleted,
            AgentState::Validating,
            Some(OpId::new(7)),
            None,
        )
        .unwrap();
        s.append_event(
            EventKind::TurnCompleted,
            AgentState::UpdatingMemory,
            Some(OpId::new(7)),
            None,
        )
        .unwrap();
        s.append_event(
            EventKind::TurnCompleted,
            AgentState::ReadyForNextTurn,
            Some(OpId::new(7)),
            None,
        )
        .unwrap();
        let bus = GlobalEventBus::new(m, Some("/w".into()));
        bus.poll_once();
        let frames = bus.frames_after(0);
        // session_created, turn_open, state(×3), turn_close(×3) — ids ascend.
        let mut ids = Vec::new();
        let mut types = Vec::new();
        for (id, ge) in &frames {
            ids.push(*id);
            types.push(ge.payload.type_name());
            assert_eq!(ge.directory.as_deref(), Some("/w"));
            assert_eq!(ge.project, None);
            assert_eq!(ge.workspace, None);
        }
        assert_eq!(
            ids,
            (1..=ids.len() as u64).collect::<Vec<_>>(),
            "ids are contiguous"
        );
        assert_eq!(types[0], "session_created");
        assert_eq!(types[1], "session_turn_open");
        assert_eq!(
            types.iter().filter(|t| **t == "session_turn_close").count(),
            3,
            "every TurnCompleted closes the turn"
        );
        // Turn open/close pair on the same op id.
        let open = &frames[1].1.payload;
        let close = frames
            .iter()
            .find(|(_, ge)| matches!(ge.payload, GlobalEventPayload::SessionTurnClose { .. }))
            .map(|(_, ge)| &ge.payload)
            .unwrap();
        match (open, close) {
            (
                GlobalEventPayload::SessionTurnOpen { turn_id: a, .. },
                GlobalEventPayload::SessionTurnClose { turn_id: b, .. },
            ) => assert_eq!(a, b),
            other => panic!("expected turn pair, got {other:?}"),
        }
        // Idempotence: a second poll emits nothing new.
        let latest = bus.latest_id();
        bus.poll_once();
        assert!(bus.frames_after(latest).is_empty());
    }

    #[tokio::test]
    async fn text_deltas_are_coalesced_across_polls_and_never_lost() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path());
        let s = session(&m);
        to_streaming(&s);
        chunk(&s, serde_json::json!({"message_id": 1, "text": "hel"}));
        chunk(&s, serde_json::json!({"message_id": 1, "text": "lo wo"}));
        let bus = GlobalEventBus::new(m, None);
        bus.poll_once();
        // Both chunks arrived within the window: pending, not emitted.
        let frames = bus.frames_after(0);
        assert_eq!(
            frames.len(),
            4,
            "created/turn_open/state/state only: {frames:?}"
        );
        let latest = bus.latest_id();
        // After the window elapses, the burst is one merged frame.
        tokio::time::sleep(std::time::Duration::from_millis(DELTA_WINDOW_MS + 40)).await;
        bus.poll_once();
        let frames = bus.frames_after(latest);
        assert_eq!(frames.len(), 1, "burst must merge into one frame");
        match &frames[0].1.payload {
            GlobalEventPayload::SessionNextTextDelta { delta, .. } => {
                assert_eq!(delta, "hello wo", "hel + lo wo concatenate")
            }
            other => panic!("expected text delta, got {other:?}"),
        }
        // The tail after a quiet period is flushed too.
        let latest = bus.latest_id();
        chunk(&s, serde_json::json!({"message_id": 1, "text": "rld"}));
        bus.poll_once();
        assert!(
            bus.frames_after(latest).is_empty(),
            "fresh chunk is pending"
        );
        tokio::time::sleep(std::time::Duration::from_millis(DELTA_WINDOW_MS + 40)).await;
        bus.poll_once();
        let frames = bus.frames_after(latest);
        assert_eq!(frames.len(), 1);
        match &frames[0].1.payload {
            GlobalEventPayload::SessionNextTextDelta { delta, .. } => assert_eq!(delta, "rld"),
            other => panic!("expected text delta, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn large_delta_flushes_immediately() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path());
        let s = session(&m);
        to_streaming(&s);
        let big = "x".repeat(DELTA_MAX_BYTES);
        chunk(&s, serde_json::json!({"message_id": 1, "text": big}));
        let bus = GlobalEventBus::new(m, None);
        bus.poll_once();
        let frames = bus.frames_after(0);
        // The oversized delta is its own frame right away (no window wait).
        let delta = frames
            .iter()
            .find(|(_, ge)| matches!(ge.payload, GlobalEventPayload::SessionNextTextDelta { .. }))
            .unwrap();
        match &delta.1.payload {
            GlobalEventPayload::SessionNextTextDelta { delta, .. } => {
                assert_eq!(delta.len(), DELTA_MAX_BYTES)
            }
            other => panic!("expected text delta, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn recovers_delta_text_from_message_parts() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path());
        let s = session(&m);
        to_streaming(&s);
        let mid = s
            .put_message(1, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        chunk(&s, serde_json::json!({"message_id": mid, "text_len": 5}));
        s.put_text_part(mid, "hello").unwrap();
        let bus = GlobalEventBus::new(m, None);
        bus.poll_once();
        let latest = bus.latest_id();
        tokio::time::sleep(std::time::Duration::from_millis(DELTA_WINDOW_MS + 40)).await;
        bus.poll_once();
        let frames = bus.frames_after(latest);
        assert_eq!(frames.len(), 1);
        match &frames[0].1.payload {
            GlobalEventPayload::SessionNextTextDelta { delta, .. } => assert_eq!(delta, "hello"),
            other => panic!("expected recovered text delta, got {other:?}"),
        }
        // Another part lands later: only the new text is emitted.
        let latest = bus.latest_id();
        s.put_text_part(mid, " world").unwrap();
        bus.poll_once();
        tokio::time::sleep(std::time::Duration::from_millis(DELTA_WINDOW_MS + 40)).await;
        bus.poll_once();
        let frames = bus.frames_after(latest);
        assert_eq!(frames.len(), 1);
        match &frames[0].1.payload {
            GlobalEventPayload::SessionNextTextDelta { delta, .. } => assert_eq!(delta, " world"),
            other => panic!("expected recovered text delta, got {other:?}"),
        }
    }

    #[test]
    fn ring_is_bounded_and_resume_clamps() {
        let dir = tempfile::tempdir().unwrap();
        let m = manager(dir.path());
        let ws = m.create_workspace("/w").unwrap();
        // 5 sessions × 2 events each = 10 frames; ring capacity is 8.
        let sessions: Vec<SessionHandle> = (0..5)
            .map(|_| m.create_session(ws, "t", "fake", "m").unwrap())
            .collect();
        for s in &sessions {
            s.append_event(EventKind::PromptReceived, AgentState::Preparing, None, None)
                .unwrap();
        }
        let bus = GlobalEventBus::with_capacity(m, None, 8);
        bus.poll_once();
        assert_eq!(bus.ring_len(), 8, "ring must be bounded");
        assert_eq!(bus.latest_id(), 10);
        // Clamped replay: everything the ring can still serve.
        let frames = bus.frames_after(0);
        assert_eq!(frames.len(), 8);
        let first = frames.first().unwrap().0;
        assert!(first > 2, "the oldest frames fell off the ring");
        // Resume from the newest id: nothing.
        assert!(bus.frames_after(10).is_empty());
        // Oversized after is clamped, not an error: u64::MAX returns nothing.
        assert!(bus.frames_after(u64::MAX).is_empty());
        // Resume from a live cursor only gets the newer frames.
        let frames = bus.frames_after(7);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames.first().unwrap().0, 8);
    }
}
