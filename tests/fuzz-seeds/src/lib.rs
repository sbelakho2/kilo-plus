//! Parser fuzz harnesses (audits P0-75/P0-81) with two consumers:
//!
//! 1. a deterministic seeded campaign (`cargo test -p
//!    faktor-tests-fuzz-seeds`, [`campaign`]) that replays checked-in
//!    fixtures plus generated hostile inputs with a bounded wall cap, and
//! 2. coverage-guided libFuzzer through the sibling `fuzz/` cargo-fuzz
//!    workspace, whose `fuzz_target!` entries call the `#[cfg(fuzzing)]`
//!    wrappers in [`fuzz_entry`].
//!
//! Every harness has the SAME shape: `fn(&[u8]) -> Outcome`, never panics
//! on its own, and performs its invariants internally — a violation is
//! returned as [`Outcome::Violation`] with the evidence, never silently
//! accepted. A real libFuzzer run therefore only needs to call the harness
//! and treat a panic (the wrapper turns a violation into one) as a finding.
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
//! 3. [`harness_acp_frame_decoder`] — the ACP Content-Length frame
//!    decoder: arbitrary byte streams (hostile headers, oversized
//!    declarations, truncated bodies) must never panic; a decoded frame
//!    must consume no more than the input and re-encode into a frame that
//!    decodes back to the identical JSON-RPC value.
//! 4. [`harness_compat_dto_decode`] — the frozen v7.5.6 compat DTOs: every
//!    hostile JSON shape decodes to a typed accept/reject, an accepted
//!    decode re-serializes and re-decodes to the identical value, and the
//!    legacy `Handshake` line round-trips.
//! 5. [`harness_tool_json_repair_parse`] — tool-call JSON repair/parse:
//!    the single deterministic repair pass and every `ToolCallMode` parse
//!    are panic-free, bounded (≤64 calls) and deterministic; repaired
//!    values must reparse exactly.
//! 6. [`harness_path_normalization`] — workspace-relative path resolution
//!    (traversal/symlink-safe): arbitrary path text must never panic and a
//!    successful resolution must NEVER land outside the workspace root
//!    (escapes are typed denials).
//! 7. [`harness_event_payload_decode`] — the journal event payload decoder
//!    across every `EventKind` and hostile schema versions: arbitrary JSON
//!    must never panic; decode outcomes are either typed rejections or
//!    typed decodes.
//! 8. [`harness_destination_policy_parser`] — the destination allowlist
//!    parser (`DestinationRule`/`DestinationPolicy`): arbitrary rule text
//!    must never panic, a successful multi-line parse must agree with the
//!    per-entry parse, and reparsing must be idempotent.
//! 9. [`harness_tokenizer_pricing_parse`] — the pure model → tokenizer
//!    mapping and the pricing-snapshot/pricing-state decoders: arbitrary
//!    model strings and hostile pricing JSON must never panic; accepted
//!    snapshots round-trip, the mapping is total and deterministic, and an
//!    `Unknown` authority never fabricates a settled cost.
//!
//! # Seeds
//!
//! [`fixtures`] loads the checked-in `compat/kilo-v756/` and
//! `fixtures/providers/` corpora; the campaign replays them (also mutated)
//! and `fuzz/seed-corpus.sh` regenerates the libFuzzer corpora from the
//! same files.
//!
//! # Real fuzzer interface
//!
//! The `fuzz/` cargo-fuzz workspace (NOT a member of this workspace; it
//! declares its own `[workspace]`) holds one `fuzz_target!` per harness.
//! `cargo fuzz` builds with `--cfg fuzzing`; [`fuzz_entry`] exists only
//! then and panics on [`Outcome::Violation`] so libFuzzer reports it as a
//! crash. The CI nightly job runs each target under a time budget when a
//! nightly toolchain is present and records a skip otherwise.
//!
//! Every harness is `pub`, takes exactly `&[u8]`, and is deterministic
//! (no process-global mutable state), so the identical function runs under
//! the pseudo-fuzz campaign and under libFuzzer. Harnesses that need a
//! filesystem (`path_normalization`) lazily build ONE throwaway workspace
//! per process and keep it for the process lifetime.

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

/// Decode-and-round-trip helper shared by the DTO harnesses: `Ok(false)`
/// means the bytes were a typed rejection (not decodable as `T`); `Ok(true)`
/// means they decoded and the value survived serialize → decode unchanged;
/// `Err` is an invariant breach.
pub(crate) fn json_roundtrip<T>(bytes: &[u8]) -> Result<bool, String>
where
    T: serde::de::DeserializeOwned + serde::Serialize + PartialEq + fmt::Debug,
{
    let Ok(value) = serde_json::from_slice::<T>(bytes) else {
        return Ok(false);
    };
    let encoded = serde_json::to_vec(&value)
        .map_err(|e| format!("re-serialize of decoded value failed: {e}"))?;
    match serde_json::from_slice::<T>(&encoded) {
        Ok(again) if again == value => Ok(true),
        Ok(again) => Err(format!(
            "round-trip mismatch: decoded {value:?} but its own encoding decoded back to {again:?}"
        )),
        Err(e) => Err(format!(
            "re-decode of a value's own serialization failed: {e}"
        )),
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

pub mod acp_frame;
pub mod campaign;
pub mod compat_dto;
pub mod destination_policy;
pub mod event_payload;
pub mod fixtures;
pub mod line_framing;
pub mod path_normalization;
pub mod sse_frame;
pub mod tokenizer_pricing;
pub mod tool_json;

#[cfg(all(test, unix))]
pub mod supply_chain;

pub use acp_frame::harness_acp_frame_decoder;
pub use campaign::{run_campaign, CampaignConfig, CampaignReport, HARNESSES};
pub use compat_dto::harness_compat_dto_decode;
pub use destination_policy::harness_destination_policy_parser;
pub use event_payload::harness_event_payload_decode;
pub use line_framing::harness_provider_line_framing;
pub use path_normalization::harness_path_normalization;
pub use sse_frame::{harness_sse_frame_parser, harness_sse_frame_truncation};
pub use tokenizer_pricing::harness_tokenizer_pricing_parse;
pub use tool_json::harness_tool_json_repair_parse;

/// libFuzzer entry points (`cargo fuzz` builds the crate with `--cfg
/// fuzzing`). Each one runs the shared harness and panics on a violation so
/// libFuzzer records a crash; the fuzz targets in the sibling `fuzz/`
/// workspace call exactly these functions.
#[cfg(fuzzing)]
pub mod fuzz_entry {
    use super::*;

    fn require_clean(name: &'static str, outcome: Outcome) {
        if let Outcome::Violation(evidence) = outcome {
            panic!("{name}: invariant violation: {evidence}");
        }
    }

    pub fn no_panic_provider_line_framing(data: &[u8]) {
        require_clean("provider_line_framing", harness_provider_line_framing(data));
    }

    pub fn no_panic_sse_frame(data: &[u8]) {
        require_clean("sse_frame", harness_sse_frame_parser(data));
        require_clean("sse_frame_truncation", harness_sse_frame_truncation(data));
    }

    pub fn no_panic_acp_frame(data: &[u8]) {
        require_clean("acp_frame", harness_acp_frame_decoder(data));
    }

    pub fn no_panic_compat_dto(data: &[u8]) {
        require_clean("compat_dto", harness_compat_dto_decode(data));
    }

    pub fn no_panic_tool_json(data: &[u8]) {
        require_clean("tool_json", harness_tool_json_repair_parse(data));
    }

    pub fn no_panic_path_normalization(data: &[u8]) {
        require_clean("path_normalization", harness_path_normalization(data));
    }

    pub fn no_panic_event_payload(data: &[u8]) {
        require_clean("event_payload", harness_event_payload_decode(data));
    }

    pub fn no_panic_destination_policy(data: &[u8]) {
        require_clean(
            "destination_policy",
            harness_destination_policy_parser(data),
        );
    }

    pub fn no_panic_tokenizer_pricing(data: &[u8]) {
        require_clean("tokenizer_pricing", harness_tokenizer_pricing_parse(data));
    }
}
