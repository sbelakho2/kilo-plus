//! Pure parser harnesses for seeded pseudo-fuzzing (P0-75).
//!
//! Every harness has the SAME shape: `fn(&[u8]) -> Outcome`, never panics
//! on its own, and performs its invariants internally — a violation is
//! returned as [`Outcome::Violation`] with the evidence, never silently
//! accepted. A real libFuzzer run therefore only needs to call the harness
//! and treat a panic (or a violation the wrapper turns into a panic) as a
//! finding.
//!
//! Targets:
//! 1. [`harness_provider_line_framing`] — `faktor-provider`'s byte-level
//!    SSE/NDJSON line framing driven over DETERMINISTICALLY DIFFERENT chunk
//!    splits of the same bytes (chunk boundaries must never change the
//!    emitted stream: lines and multibyte runes split across chunks must
//!    reassemble identically; strict UTF-8 rejections and the line cap must
//!    be split-independent).
//! 2. [`harness_sse_frame_parser`] — the frozen v7.5.6 SSE frame parser:
//!    arbitrary text and truncated/mutated frames must never panic, and
//!    every successfully parsed frame must round-trip (parse(encode(x)) ==
//!    x).
//! 3. [`harness_destination_policy_parser`] — the destination allowlist
//!    parser (`DestinationRule`/`DestinationPolicy`): arbitrary rule text
//!    must never panic, a successful multi-line parse must agree with the
//!    per-entry parse, and reparsing must be idempotent.
//! 4. [`harness_path_normalization`] — workspace-relative path resolution
//!    (traversal/symlink-safe): arbitrary path text must never panic and a
//!    successful resolution must NEVER land outside the workspace root
//!    (escapes are typed denials).
//! 5. [`harness_event_payload_decode`] — the journal event payload decoder
//!    across every `EventKind` and hostile schema versions: arbitrary JSON
//!    must never panic; decode outcomes are either typed rejections or
//!    typed decodes.
//!
//! # Deterministic seeded tests
//!
//! One normal-mode test per target drives 2000 deterministic LCG
//! iterations (mixed structured + hostile byte inputs). These are NOT a
//! libFuzzer corpus; they are a stable, seed-reproducible pseudo-fuzz gate.
//!
//! # libFuzzer wrapper point
//!
//! Real coverage-guided fuzzing needs a nightly toolchain (`cargo-fuzz` /
//! `libfuzzer-sys` compile with nightly-only flags) and this workspace pins
//! `stable` (`rust-toolchain.toml`) — so the setup is NOT feasible here and
//! is deliberately skipped. When run under nightly, add a `fuzz/`
//! cargo-fuzz member that depends on this crate and call the harnesses
//! directly:
//!
//! ```rust,ignore
//! // fuzz/fuzz_targets/line_framing.rs  (cargo-fuzz, nightly only)
//! #![no_main]
//! use libfuzzer_sys::fuzz_target;
//! fuzz_target!(|data: &[u8]| {
//!     if let fuzz_targets::Outcome::Violation(v) =
//!         fuzz_targets::harness_provider_line_framing(data)
//!     {
//!         panic!("invariant violation: {v}");
//!     }
//! });
//! ```
//!
//! Every harness is `pub`, takes exactly `&[u8]`, and is deterministic
//! (no process-global state), so the identical function runs under the
//! pseudo-fuzz test and under libFuzzer. Harnesses that need a filesystem
//! (`path_normalization`) lazily build ONE throwaway workspace per process
//! and keep it for the process lifetime.

use std::fmt;

/// One deterministic parse outcome. Every harness returns exactly one of
/// these and never panics on hostile input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The input was exercised; all invariants held.
    Clean,
    /// The input was not applicable to the target's domain (invalid UTF-8,
    /// not JSON, ...) — still exercised, still panic-free.
    NotApplicable,
    /// An invariant was breached. The payload is the evidence.
    Violation(String),
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Clean => write!(f, "clean"),
            Outcome::NotApplicable => write!(f, "not-applicable"),
            Outcome::Violation(v) => write!(f, "violation: {v}"),
        }
    }
}

impl Outcome {
    /// A harness result acceptable to a fuzz gate.
    pub fn is_ok(&self) -> bool {
        matches!(self, Outcome::Clean | Outcome::NotApplicable)
    }
}

// ---------------------------------------------------------------------------
// Deterministic LCG (same constants as the scheduler modelcheck LCG).
// ---------------------------------------------------------------------------

pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1) | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32 as u64
    }

    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        self.next_u64() % n
    }

    pub fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

pub mod destination_policy;
pub mod event_payload;
pub mod line_framing;
pub mod path_normalization;
pub mod sse_frame;

pub use destination_policy::harness_destination_policy_parser;
pub use event_payload::harness_event_payload_decode;
pub use line_framing::harness_provider_line_framing;
pub use path_normalization::harness_path_normalization;
pub use sse_frame::harness_sse_frame_parser;
