//! Bounded output ring shared by both PTY backends (the unix reader thread
//! and the Windows ConPTY reader thread). Oldest bytes are dropped so memory
//! stays bounded regardless of how much output a hostile child produces.

use std::collections::VecDeque;

/// Bounded output ring (bytes kept, oldest dropped) — RAM stays bounded
/// for hostile or huge output.
pub(crate) const RING_MAX_BYTES: usize = 256 * 1024;

pub(crate) struct Ring {
    buf: VecDeque<u8>,
    total: u64,
}

impl Ring {
    pub(crate) fn new() -> Self {
        Self {
            buf: VecDeque::new(),
            total: 0,
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.buf.push_back(*b);
        }
        while self.buf.len() > RING_MAX_BYTES {
            self.buf.pop_front();
        }
        self.total = self.total.saturating_add(bytes.len() as u64);
    }

    /// Drain the currently available bytes.
    pub(crate) fn drain(&mut self) -> Vec<u8> {
        self.buf.drain(..).collect()
    }

    pub(crate) fn snapshot(&self) -> Vec<u8> {
        self.buf.iter().copied().collect()
    }

    /// Total bytes ever pushed (before ring eviction).
    pub(crate) fn total(&self) -> u64 {
        self.total
    }
}
