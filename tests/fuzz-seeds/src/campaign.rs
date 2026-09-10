//! Bounded deterministic campaign runner.
//!
//! One run drives every harness round-robin with a deterministic stream of
//! inputs (checked-in fixture seeds, mutated fixtures, raw bytes, JSON/line
//! soup and framed bodies) for at most `iterations_target` iterations OR
//! `wall_cap`, whichever comes first. The runner reports coverage-ish
//! counters per harness: iterations, clean parses, typed rejections
//! (`NotApplicable`), invariant violations and panics (counted with
//! `catch_unwind`, so a panicking harness is reported as `panics > 0`
//! instead of aborting the campaign).
//!
//! The same runner backs the `cargo test -p faktor-tests-fuzz-seeds` smoke:
//! a deterministic 2000-iteration gate and the 50k bounded campaign.

use std::collections::BTreeMap;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

use crate::fixtures::{fixture_seeds, FixtureSeed};
use crate::{
    harness_acp_frame_decoder, harness_compat_dto_decode, harness_destination_policy_parser,
    harness_event_payload_decode, harness_path_normalization, harness_provider_line_framing,
    harness_sse_frame_parser, harness_tokenizer_pricing_parse, harness_tool_json_repair_parse, Lcg,
    Outcome,
};

/// One named harness under campaign.
pub struct HarnessCase {
    pub name: &'static str,
    pub run: fn(&[u8]) -> Outcome,
}

/// Every registered harness, in stable order.
pub const HARNESSES: &[HarnessCase] = &[
    HarnessCase {
        name: "line_framing",
        run: harness_provider_line_framing,
    },
    HarnessCase {
        name: "sse_frame",
        run: harness_sse_frame_parser,
    },
    HarnessCase {
        name: "acp_frame",
        run: harness_acp_frame_decoder,
    },
    HarnessCase {
        name: "compat_dto",
        run: harness_compat_dto_decode,
    },
    HarnessCase {
        name: "tool_json",
        run: harness_tool_json_repair_parse,
    },
    HarnessCase {
        name: "path_normalization",
        run: harness_path_normalization,
    },
    HarnessCase {
        name: "event_payload",
        run: harness_event_payload_decode,
    },
    HarnessCase {
        name: "destination_policy",
        run: harness_destination_policy_parser,
    },
    HarnessCase {
        name: "tokenizer_pricing",
        run: harness_tokenizer_pricing_parse,
    },
];

/// Campaign bounds.
#[derive(Debug, Clone, Copy)]
pub struct CampaignConfig {
    pub seed: u64,
    /// Hard iteration ceiling across all harnesses.
    pub iterations_target: u64,
    /// Hard wall-clock ceiling; the campaign stops at the first loop top
    /// after this elapses.
    pub wall_cap: Duration,
}

impl Default for CampaignConfig {
    fn default() -> Self {
        Self {
            seed: 0x5EED_F011,
            iterations_target: 50_000,
            wall_cap: Duration::from_secs(20),
        }
    }
}

/// Per-harness counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TargetCounters {
    pub iterations: u64,
    /// Inputs the harness exercised end-to-end (`Outcome::Clean`).
    pub clean: u64,
    /// Typed rejections (`Outcome::NotApplicable`).
    pub not_applicable: u64,
    /// Invariant breaches (`Outcome::Violation`).
    pub violations: u64,
    /// Harness panics (counted, not fatal to the campaign).
    pub panics: u64,
}

/// What one campaign observed.
#[derive(Debug, Clone)]
pub struct CampaignReport {
    pub iterations: u64,
    pub clean: u64,
    pub not_applicable: u64,
    pub violations: u64,
    pub panics: u64,
    pub wall: Duration,
    pub stopped_by_wall_cap: bool,
    pub per_target: BTreeMap<&'static str, TargetCounters>,
    pub fixture_seed_count: usize,
    /// First piece of evidence seen (panic payload or violation message).
    pub first_failure: Option<String>,
}

impl CampaignReport {
    /// Typed rejections across all harnesses.
    pub fn typed_rejections(&self) -> u64 {
        self.not_applicable
    }

    /// One-line report for logs and test failures.
    pub fn summary_line(&self) -> String {
        format!(
            "iterations={} clean={} typed_rejections={} violations={} panics={} seeds={} wall_ms={} stopped_by_wall_cap={}",
            self.iterations,
            self.clean,
            self.not_applicable,
            self.violations,
            self.panics,
            self.fixture_seed_count,
            self.wall.as_millis(),
            self.stopped_by_wall_cap
        )
    }
}

impl fmt::Display for CampaignReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary_line())
    }
}

/// Run the bounded deterministic campaign.
pub fn run_campaign(config: &CampaignConfig) -> CampaignReport {
    let started = Instant::now();
    let seeds = fixture_seeds();
    let mut lcg = Lcg::new(config.seed);
    let mut report = CampaignReport {
        iterations: 0,
        clean: 0,
        not_applicable: 0,
        violations: 0,
        panics: 0,
        wall: Duration::ZERO,
        stopped_by_wall_cap: false,
        per_target: BTreeMap::new(),
        fixture_seed_count: seeds.len(),
        first_failure: None,
    };
    let mut iteration = 0u64;
    while iteration < config.iterations_target {
        if started.elapsed() >= config.wall_cap {
            report.stopped_by_wall_cap = true;
            break;
        }
        let case = &HARNESSES[(iteration as usize) % HARNESSES.len()];
        let input = next_input(&mut lcg, seeds, iteration);
        let outcome = catch_unwind(AssertUnwindSafe(|| (case.run)(&input)));
        let counters = report.per_target.entry(case.name).or_default();
        counters.iterations += 1;
        report.iterations += 1;
        match outcome {
            Ok(Outcome::Clean) => {
                counters.clean += 1;
                report.clean += 1;
            }
            Ok(Outcome::NotApplicable) => {
                counters.not_applicable += 1;
                report.not_applicable += 1;
            }
            Ok(Outcome::Violation(evidence)) => {
                counters.violations += 1;
                report.violations += 1;
                if report.first_failure.is_none() {
                    report.first_failure = Some(format!("{}: violation: {evidence}", case.name));
                }
            }
            Err(payload) => {
                counters.panics += 1;
                report.panics += 1;
                if report.first_failure.is_none() {
                    let message = payload
                        .downcast_ref::<&str>()
                        .map(|s| (*s).to_string())
                        .or_else(|| payload.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "panic".to_string());
                    report.first_failure = Some(format!("{}: panic: {message}", case.name));
                }
            }
        }
        iteration += 1;
    }
    report.wall = started.elapsed();
    report
}

// ---------------------------------------------------------------------------
// Deterministic hostile-input generator.
// ---------------------------------------------------------------------------

fn random_bytes(lcg: &mut Lcg, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(lcg.next_u64() as u8);
    }
    out
}

fn ascii_soup(lcg: &mut Lcg, n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(match lcg.below(5) {
            0 => b'\n',
            1 => b' ',
            2 => b'"',
            3 => b'\\',
            _ => b'a' + lcg.below(26) as u8,
        });
    }
    out
}

fn json_soup(lcg: &mut Lcg, n: usize) -> Vec<u8> {
    let atoms = [
        "{",
        "}",
        "[",
        "]",
        "\"",
        ":",
        ",",
        "null",
        "true",
        "false",
        "\"name\"",
        "\"input\"",
        "\"sessionID\"",
        "\"payload\"",
        "\"type\"",
        "\"authority\"",
        "\"quote\"",
        "12345678901234567890",
        "-1",
        "1e309",
        "\\u0000",
    ];
    let mut text = String::new();
    for _ in 0..n {
        text.push_str(atoms[lcg.below(atoms.len() as u64) as usize]);
    }
    text.into_bytes()
}

fn line_soup(lcg: &mut Lcg, n: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for _ in 0..n {
        match lcg.below(6) {
            0 => out.extend_from_slice(b"data: {}\n"),
            1 => out.extend_from_slice(b"event: message\r\n"),
            2 => {
                let body = format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":{},\"method\":\"session/prompt\"}}",
                    lcg.next_u64() % 1000
                );
                out.extend_from_slice(
                    format!("Content-Length: {}\r\n\r\n{body}", body.len()).as_bytes(),
                );
            }
            3 => {
                let n = 1 + lcg.below(24) as usize;
                out.extend_from_slice(&asphalt(lcg, n));
            }
            4 => out.extend_from_slice("λμν".as_bytes()),
            _ => out.push(b'\n'),
        }
    }
    out
}

/// A hostile-but-plausible non-UTF-8 token.
fn asphalt(lcg: &mut Lcg, n: usize) -> Vec<u8> {
    (0..n)
        .map(|_| 0x80 | (lcg.next_u64() as u8 & 0x7f))
        .collect()
}

fn mutate(seed: &FixtureSeed, lcg: &mut Lcg) -> Vec<u8> {
    let mut bytes = seed.bytes.clone();
    if bytes.is_empty() {
        return bytes;
    }
    for _ in 0..=lcg.below(8) {
        let pos = (lcg.next_u64() as usize) % bytes.len();
        bytes[pos] ^= 1 << (lcg.below(8) as u8);
    }
    if lcg.chance(25) {
        let cut = (lcg.next_u64() as usize) % bytes.len();
        bytes.truncate(cut);
    }
    bytes
}

fn next_input(lcg: &mut Lcg, seeds: &[FixtureSeed], iteration: u64) -> Vec<u8> {
    let choose = lcg.below(8);
    if (choose == 3 || choose == 4) && !seeds.is_empty() {
        let seed = &seeds[(iteration as usize) % seeds.len()];
        return if choose == 3 {
            seed.bytes.clone()
        } else {
            mutate(seed, lcg)
        };
    }
    match choose {
        0 => {
            let n = 1 + lcg.below(1024) as usize;
            random_bytes(lcg, n)
        }
        1 => {
            let n = 1 + lcg.below(768) as usize;
            ascii_soup(lcg, n)
        }
        2 => {
            let n = 1 + lcg.below(64) as usize;
            json_soup(lcg, n)
        }
        5 => {
            let n = 1 + lcg.below(16) as usize;
            line_soup(lcg, n)
        }
        6 => {
            let byte = lcg.next_u64() as u8;
            let n = lcg.below(2048) as usize;
            vec![byte; n]
        }
        _ => {
            // A framed JSON-RPC body with arbitrary inner bytes.
            let n = lcg.below(256) as usize;
            let body = random_bytes(lcg, n);
            let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
            out.extend_from_slice(&body);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The deterministic 2000-iteration smoke is reproducible: same seed and
    /// generous cap produce identical counters.
    #[test]
    fn seeded_campaign_2000_is_deterministic() {
        let config = CampaignConfig {
            seed: 0x5EED_0000_0000_000A,
            iterations_target: 2_000,
            wall_cap: Duration::from_secs(120),
        };
        let first = run_campaign(&config);
        let second = run_campaign(&config);
        assert!(!first.stopped_by_wall_cap, "{}", first.summary_line());
        assert_eq!(first.iterations, 2_000);
        assert_eq!(first.panics, 0, "{}", first.summary_line());
        assert_eq!(first.violations, 0, "{:?}", first.first_failure);
        assert_eq!(first.iterations, first.clean + first.not_applicable);
        assert_eq!(first.per_target.len(), HARNESSES.len());
        for (name, counters) in &first.per_target {
            assert_eq!(
                counters.clean + counters.not_applicable + counters.violations + counters.panics,
                counters.iterations,
                "{name} counters must partition"
            );
        }
        assert_eq!(first.iterations, second.iterations);
        assert_eq!(first.clean, second.clean);
        assert_eq!(first.not_applicable, second.not_applicable);
        assert_eq!(first.violations, second.violations);
        assert_eq!(first.panics, second.panics);
        assert_eq!(first.per_target, second.per_target);
    }

    /// The 50k bounded campaign: hard iteration ceiling, hard wall cap,
    /// zero panics/violations, typed rejections recorded.
    #[test]
    fn campaign_50k_across_harnesses_is_bounded_and_panic_free() {
        let config = CampaignConfig {
            seed: 0x5EED_0000_0000_000B,
            iterations_target: 50_000,
            wall_cap: Duration::from_secs(20),
        };
        let report = run_campaign(&config);
        eprintln!("[fuzz-campaign] {}", report.summary_line());
        assert_eq!(report.panics, 0, "{}", report.summary_line());
        assert_eq!(
            report.violations, 0,
            "first failure: {:?}",
            report.first_failure
        );
        assert!(report.iterations > 0);
        assert!(report.iterations <= 50_000);
        assert_eq!(
            report.iterations,
            report.clean + report.not_applicable + report.violations + report.panics
        );
        assert!(
            report.typed_rejections() > 0,
            "typed rejections must be counted"
        );
        assert_eq!(report.per_target.len(), HARNESSES.len());
        assert!(
            report.fixture_seed_count > 0,
            "fixture seeds must be replayed"
        );
        assert!(report.summary_line().contains("panics=0"));
    }

    /// A tiny wall cap stops the campaign promptly and is reported.
    #[test]
    fn campaign_respects_wall_cap() {
        let config = CampaignConfig {
            seed: 0x5EED_0000_0000_000C,
            iterations_target: u64::MAX,
            wall_cap: Duration::from_millis(50),
        };
        let report = run_campaign(&config);
        assert!(report.stopped_by_wall_cap, "{}", report.summary_line());
        assert!(
            report.wall < Duration::from_secs(5),
            "wall cap not respected: {:?}",
            report.wall
        );
        assert!(report.iterations < 100_000, "cap must bound the work");
    }
}
