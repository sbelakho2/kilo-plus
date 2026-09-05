//! Measurable prefix-cache stability (audits 65-66, prefix-cache slice).
//!
//! Provider-side prompt caching turns a stable prompt prefix into a cheaper
//! call. When a session rewrites its evidence order or regenerates its head
//! every turn, the provider cache misses silently and every turn pays the
//! uncached input price. This module makes that failure MODESTLY measurable
//! from the only data the routing layer can honestly see per turn:
//!
//! ```text
//! TurnPrefix {
//!     turn_id: u64,            // caller's durable turn identity
//!     prefix_hash: [u8; 32],   // digest of the exact cacheable-prefix bytes sent
//!     prefix_tokens: u32,      // token count of that prefix
//! }
//! ```
//!
//! The digest is an equality oracle, not a substring oracle: from a digest
//! pair alone the byte-level longest common prefix (LCP) is NOT recoverable
//! (hash-and-count is information-theoretically insufficient). The metric
//! therefore anchors on what digests DO prove and uses the token counts only
//! where they can prove or disprove append-consistency:
//!
//! Per-turn stability, consecutive observations (i-1, i), i ≥ 2:
//!
//! ```text
//! stability(i) = 1.0                          turn 1 (nothing precedes it)
//!              = 1.0                          either prefix is EMPTY (0 tokens):
//!                                              an empty prefix destabilizes nothing
//!                                              (documented convention; also keeps the
//!                                              denominator safe)
//!              = 1.0                          prefix_hash(i) == prefix_hash(i-1):
//!                                              byte-identical prefixes — every byte of
//!                                              the shorter sits in the longer, so the
//!                                              provider cache from turn i-1 covers turn i
//!              = t(i-1) / t(i)                t(i) > t(i-1): STRICT GROWTH is the only
//!                                              byte relation consistent with "the whole
//!                                              previous prefix is still there, bytes were
//!                                              only appended" (appending can never
//!                                              shrink a token count). Growth sessions
//!                                              therefore keep ~1.0, and the value is the
//!                                              exact cache-coverage fraction of the
//!                                              current prefix under that hypothesis.
//!              = 0.0                          otherwise: same-length or shorter prefix
//!                                              with different bytes is a REWRITE — the
//!                                              classic reorder/churn signature that
//!                                              invalidates provider caches.
//! ```
//!
//! Session level: the mean over turns with the population standard deviation
//! ([`prefix_stability`], [`stability_stats`]). An empty or single-observation
//! history is defined fully stable (1.0, σ = 0): nothing was observed, so
//! nothing can be judged churning.
//!
//! Honest limitation (documented, never guessed): a session that rewrites its
//! prefix AND still grows token-for-token is byte-wise indistinguishable from
//! an append-only session from digests alone; such sessions score the growth
//! ratio. Exact per-prefix LCP needs the raw bytes, which live at the
//! settlement site (fill-site wiring, see the store migration notes) — the
//! digest rule above is the deterministic, adversarial-safe approximation the
//! routing layer can reproduce from durable rows alone.
//!
//! Churn advisory ([`prefix_churn`]): when per-turn stability drops below a
//! configurable floor (default [`DEFAULT_STABILITY_FLOOR`] = 0.8) for
//! [`CHURN_WINDOW`] = 3 consecutive turns, an advisory `prefix_churn` signal
//! fires on the turn that COMPLETES the run. It stays quiet for every further
//! low turn of the same run (no spam per settlement) and re-arms only after a
//! recovery turn (stability ≥ floor).
//!
//! Cost integration ([`churn_penalty`], [`apply_churn_penalty`]): the
//! router's expected-cost model charges a risk premium when the session's
//! LAST recorded stability is below the floor: `estimate *= (1 + penalty)`
//! with `penalty = MAX_CHURN_PENALTY * (floor - stability) / floor`,
//! bounded in [0, 0.25], monotone in (floor - stability), never negative, and
//! 0.0 for non-finite inputs (a NaN stability is not evidence of churn and
//! must never leak into integer micro-unit math).

/// One per-turn prefix observation, as persisted by the settlement layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnPrefix {
    pub turn_id: u64,
    /// Digest of the exact cacheable-prefix byte string sent in that turn.
    pub prefix_hash: [u8; 32],
    /// Token count of that prefix (0 = no cacheable prefix was sent).
    pub prefix_tokens: u32,
}

impl TurnPrefix {
    pub fn new(turn_id: u64, prefix_hash: [u8; 32], prefix_tokens: u32) -> Self {
        Self {
            turn_id,
            prefix_hash,
            prefix_tokens,
        }
    }
}

/// Default stability floor: below it a turn counts as churning.
pub const DEFAULT_STABILITY_FLOOR: f64 = 0.8;

/// A churn run completes after this many consecutive low-stability turns.
pub const CHURN_WINDOW: usize = 3;

/// Upper bound of the churn cost premium: `estimate *= 1 + 0.25` at worst.
pub const MAX_CHURN_PENALTY: f64 = 0.25;

fn empty_or_zero(t: u32) -> bool {
    t == 0
}

fn pair_stability(prev: &TurnPrefix, cur: &TurnPrefix) -> f64 {
    // Empty-prefix convention (documented above): 0 tokens destabilize nothing.
    if empty_or_zero(prev.prefix_tokens) || empty_or_zero(cur.prefix_tokens) {
        return 1.0;
    }
    // Digest equality is byte truth: identical prefixes are fully cache-stable.
    if prev.prefix_hash == cur.prefix_hash {
        return 1.0;
    }
    let (p, c) = (prev.prefix_tokens, cur.prefix_tokens);
    if c > p {
        // Strict growth is append-consistent: cache coverage of the current
        // prefix is the previous prefix's share of it.
        f64::from(p) / f64::from(c)
    } else {
        // Same-length or shrinking prefix with different bytes: rewritten
        // head/content order — the provider cache was invalidated.
        0.0
    }
}

/// Per-turn stability of every observation (index 0 = turn 1 = 1.0 by
/// definition). Deterministic and pure.
pub fn turn_stabilities(turns: &[TurnPrefix]) -> Vec<f64> {
    let mut out = Vec::with_capacity(turns.len());
    for (i, t) in turns.iter().enumerate() {
        let v = match i {
            0 => 1.0,
            _ => pair_stability(&turns[i - 1], t),
        };
        out.push(v);
    }
    out
}

/// Session-level prefix-cache stability: mean per-turn stability over the
/// series. Empty history is defined 1.0 (documented above).
pub fn prefix_stability(turns: &[TurnPrefix]) -> f64 {
    stability_stats(turns).mean
}

/// Session-level mean, population standard deviation and observation count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StabilityStats {
    pub mean: f64,
    /// Population standard deviation (divide by n); 0.0 when n ≤ 1.
    pub std_dev: f64,
    pub n: usize,
}

/// Deterministic session aggregate; empty histories are defined fully stable.
pub fn stability_stats(turns: &[TurnPrefix]) -> StabilityStats {
    let vals = turn_stabilities(turns);
    let n = vals.len();
    if n == 0 {
        return StabilityStats {
            mean: 1.0,
            std_dev: 0.0,
            n: 0,
        };
    }
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
    StabilityStats {
        mean,
        std_dev: var.max(0.0).sqrt(),
        n,
    }
}

/// A fired advisory: the session's prefix became churn-unstable for
/// `CHURN_WINDOW` consecutive turns. `stability` is the per-turn stability
/// of the firing turn (the one completing the run).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChurnSignal {
    /// Index into the input series of the turn that completed the run.
    pub turn_index: usize,
    /// Caller's durable turn identity of that turn.
    pub turn_id: u64,
    /// Fired only on the turn completing a run; quantized to the window
    /// granularity so the signal is reproducible from durable rows.
    pub stability_below_floor: bool,
}

/// Churn detector: fires on the turn completing the 3rd consecutive
/// low-stability turn (below `floor`), stays quiet for further lows of the
/// same run, and re-arms after a recovery (stability ≥ floor). Deterministic.
pub fn prefix_churn(turns: &[TurnPrefix], floor: f64) -> Vec<ChurnSignal> {
    let mut out = Vec::new();
    let mut run = 0usize;
    for (i, s) in turn_stabilities(turns).iter().enumerate() {
        if i == 0 {
            continue; // turn 1 is 1.0 by definition and can never be low
        }
        if *s < floor {
            if run < CHURN_WINDOW {
                run += 1;
            }
            if run == CHURN_WINDOW {
                out.push(ChurnSignal {
                    turn_index: i,
                    turn_id: turns[i].turn_id,
                    stability_below_floor: true,
                });
                // Stay quiet (run parked past the window) for every further
                // low of this run; a recovery resets it below.
                run = CHURN_WINDOW + 1;
            }
        } else {
            run = 0;
        }
    }
    out
}

/// Churn cost premium in [0, 0.25]: 0 at/above the floor, linear up to
/// [`MAX_CHURN_PENALTY`] at stability 0, monotone in (floor - stability),
/// never negative. Non-finite stability (NaN/inf) scores 0.0 — it is not
/// evidence of churn and must never poison integer cost math. A floor of 0
/// disables the penalty (nothing is ever below it).
pub fn churn_penalty(stability: f64, floor: f64) -> f64 {
    let floor = floor.clamp(0.0, 1.0);
    if !stability.is_finite() || floor == 0.0 {
        return 0.0;
    }
    let stability = stability.clamp(0.0, 1.0);
    if stability >= floor {
        return 0.0;
    }
    MAX_CHURN_PENALTY * (floor - stability) / floor
}

/// `estimate *= (1 + churn_penalty)` in integer micro-units, rounded UP
/// (never understate a cost prediction), saturating at u64::MAX.
pub fn apply_churn_penalty(cost_micro: u64, stability: f64, floor: f64) -> u64 {
    let penalty = churn_penalty(stability, floor);
    if penalty == 0.0 {
        return cost_micro;
    }
    let extra = ((cost_micro as f64) * penalty).ceil();
    if extra >= u64::MAX as f64 {
        return u64::MAX;
    }
    cost_micro.saturating_add(extra as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only digest: four independent FNV-1a lanes over the prefix bytes
    /// (no external hash dependency in this crate; deterministic). Distinct
    /// byte strings yield distinct digests for every fixture below.
    fn hv(bytes: &[u8]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (k, basis) in [
            0xcbf29ce484222325u64,
            0x84222325,
            0x9e3779b97f4a7c15,
            0x100000001b3,
        ]
        .iter()
        .enumerate()
        {
            let mut h = *basis ^ 0xdead_beef_1234_5678u64.wrapping_mul(k as u64 + 1);
            for &b in bytes {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x100000001b3);
            }
            let lane = h.to_le_bytes();
            out[k * 8..k * 8 + 8].copy_from_slice(&lane);
        }
        out
    }

    fn t(id: u64, bytes: &[u8]) -> TurnPrefix {
        TurnPrefix {
            turn_id: id,
            prefix_hash: hv(bytes),
            prefix_tokens: bytes.len() as u32,
        }
    }

    #[test]
    fn identical_consecutive_prefixes_score_one() {
        let p = b"stable prefix bytes that never change";
        let turns = vec![t(1, p), t(2, p)];
        assert_eq!(turn_stabilities(&turns), vec![1.0, 1.0]);
        assert_eq!(prefix_stability(&turns), 1.0);
        // Long identical chains stay 1.0.
        let chain: Vec<TurnPrefix> = (1..=50).map(|i| t(i, p)).collect();
        assert_eq!(prefix_stability(&chain), 1.0);
        let s = stability_stats(&chain);
        assert_eq!(s.std_dev, 0.0);
    }

    #[test]
    fn totally_different_same_length_prefixes_score_zero() {
        let a = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let b = b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        assert_ne!(hv(a), hv(b));
        let turns = vec![t(1, a), t(2, b)];
        assert_eq!(turn_stabilities(&turns)[1], 0.0);
        // And the reverse direction (shrink-to-equal is a rewrite too).
        let turns = vec![t(1, b), t(2, a)];
        assert_eq!(turn_stabilities(&turns)[1], 0.0);
    }

    #[test]
    fn partial_overlap_scores_the_exact_coverage_fraction() {
        // Contained growth: the whole 600-token prefix reappears at the head
        // of a 1000-token prefix. Cache coverage of the current prefix is
        // exactly 600/1000 = 0.6.
        let head = vec![b'x'; 600];
        let mut grown = head.clone();
        grown.extend_from_slice(&vec![b'y'; 400]);
        let turns = vec![t(1, &head), t(2, &grown)];
        assert_ne!(turns[0].prefix_hash, turns[1].prefix_hash);
        assert_eq!(turn_stabilities(&turns)[1], 0.6);
        // Shrinking the SAME way is a rewrite (the head is not provably
        // intact): a shrink can never be append-consistent.
        let turns = vec![t(1, &grown), t(2, &head)];
        assert_eq!(turn_stabilities(&turns)[1], 0.0);
    }

    #[test]
    fn turn_one_and_empty_prefixes_are_one_by_definition() {
        assert_eq!(turn_stabilities(&[t(1, b"anything")]), vec![1.0]);
        // 0-token prefixes destabilize nothing, in either direction.
        let empty = TurnPrefix {
            turn_id: 2,
            prefix_hash: hv(b""),
            prefix_tokens: 0,
        };
        let full = t(1, b"some prefix");
        for pair in [vec![full, empty], vec![empty, full], vec![empty, empty]] {
            assert_eq!(turn_stabilities(&pair)[1], 1.0, "{pair:?}");
        }
        // Empty histories are defined fully stable.
        assert_eq!(prefix_stability(&[]), 1.0);
        assert_eq!(stability_stats(&[]).std_dev, 0.0);
        assert_eq!(stability_stats(&[t(1, b"x")]).std_dev, 0.0);
    }

    #[test]
    fn churn_fires_exactly_on_the_third_consecutive_low_turn() {
        // Per-turn stabilities: t1 = 1.0 (definition), t2 = identical -> 1.0,
        // then equal-length rewrites (different bytes, same token count)
        // score 0.0 for as long as they repeat.
        let stable = b"identical-content-every-time"; // 28 bytes
        let rewritten = |i: u64| format!("rewritten-content-number-{i:03}").into_bytes();
        let mut turns: Vec<TurnPrefix> = vec![t(1, stable), t(2, stable)];
        turns.push(t(3, &rewritten(1)));
        turns.push(t(4, &rewritten(2)));
        let st = turn_stabilities(&turns);
        assert_eq!(st[0], 1.0);
        assert_eq!(st[1], 1.0);
        assert_eq!(st[2], 0.0);
        assert_eq!(st[3], 0.0);
        // Two lows only: no signal yet.
        assert!(prefix_churn(&turns, DEFAULT_STABILITY_FLOOR).is_empty());
        // Third consecutive low (turn 5) fires exactly once.
        turns.push(t(5, &rewritten(3)));
        let sig = prefix_churn(&turns, DEFAULT_STABILITY_FLOOR);
        assert_eq!(sig.len(), 1);
        assert_eq!(sig[0].turn_index, 4); // turns[4] == turn_id 5
        assert_eq!(sig[0].turn_id, 5);
        // Fourth and fifth consecutive lows stay quiet (no per-settlement spam):
        // a 20-turn unbroken run fires exactly once.
        for i in 6..=25 {
            turns.push(t(i, &rewritten(i)));
        }
        assert_eq!(prefix_churn(&turns, DEFAULT_STABILITY_FLOOR).len(), 1);
        // Recovery re-arms: byte-identical to the last rewritten turn scores
        // 1.0, then two lows are quiet and the third low of the NEW run fires.
        let recovery = rewritten(25);
        turns.push(t(26, &recovery)); // identical to turn 25 -> 1.0
        turns.push(t(27, &rewritten(100)));
        turns.push(t(28, &rewritten(101)));
        turns.push(t(29, &rewritten(102)));
        let sig = prefix_churn(&turns, DEFAULT_STABILITY_FLOOR);
        assert_eq!(sig.len(), 2);
        assert_eq!(sig[1].turn_index, 28);
        assert_eq!(sig[1].turn_id, 29);
        // The first run never re-fires later: still exactly the two runs.
        assert_eq!(prefix_churn(&turns, DEFAULT_STABILITY_FLOOR).len(), 2);
    }

    #[test]
    fn penalty_is_bounded_monotone_and_never_negative() {
        // At/above the floor: zero. Just below: tiny. At 0: the cap.
        assert_eq!(churn_penalty(0.8, 0.8), 0.0);
        assert_eq!(churn_penalty(1.0, 0.8), 0.0);
        assert_eq!(churn_penalty(0.0, 0.8), 0.25);
        assert!(churn_penalty(0.79, 0.8) > 0.0);
        assert!(churn_penalty(0.79, 0.8) < 0.25);
        // Bounded over the entire domain, monotone in (floor - stability).
        let mut last = f64::INFINITY;
        for st100 in 0..=100 {
            let s = st100 as f64 / 100.0;
            for f100 in 1..=100 {
                let f = f100 as f64 / 100.0;
                let p = churn_penalty(s, f);
                assert!((0.0..=0.25).contains(&p), "out of range: {s} {f} {p}");
                // Monotone: lowering stability (more churn) never lowers the
                // premium at a fixed floor.
                if s > 0.0 {
                    let lower = churn_penalty(s - 0.01, f);
                    assert!(
                        lower >= p - 1e-15,
                        "penalty must not fall as stability falls: {s} {f}"
                    );
                }
            }
            assert!(churn_penalty(s, 0.8) <= last + 1e-15);
            last = churn_penalty(s, 0.8);
        }
        // Hostile inputs: NaN and infinities are 0.0, never NaN/negative.
        assert_eq!(churn_penalty(f64::NAN, 0.8), 0.0);
        assert_eq!(churn_penalty(f64::INFINITY, 0.8), 0.0);
        assert_eq!(churn_penalty(f64::NEG_INFINITY, 0.8), 0.0);
        assert_eq!(churn_penalty(-1e9, 0.8), 0.25);
        assert_eq!(churn_penalty(2.0, 0.8), 0.0);
        // Floor 0 disables the premium entirely.
        assert_eq!(churn_penalty(0.0, 0.0), 0.0);
    }

    #[test]
    fn penalty_rounds_up_and_saturates_in_micro_units() {
        // stability 0 + floor 0.8 -> penalty 0.25 -> estimate * 1.25, ceil.
        assert_eq!(apply_churn_penalty(1000, 0.0, 0.8), 1250);
        assert_eq!(apply_churn_penalty(1001, 0.0, 0.8), 1252); // ceil(1001*1.25)
                                                               // Healthy stability and empty-history conventions never charge.
        assert_eq!(apply_churn_penalty(1000, 1.0, 0.8), 1000);
        assert_eq!(apply_churn_penalty(1000, f64::NAN, 0.8), 1000);
        // Saturation at u64::MAX; exact-zero never understates.
        assert_eq!(apply_churn_penalty(u64::MAX, 0.0, 0.8), u64::MAX);
        assert_eq!(apply_churn_penalty(1, 0.0, 0.8), 2); // ceil(1.25) = 2
    }

    #[test]
    fn append_only_session_scores_about_one_rewriting_session_scores_low() {
        // (f) Append-only: the prefix grows ONE byte per turn. Every
        // consecutive digest differs, yet stability stays ~1.0 because each
        // turn is append-consistent (strict token growth).
        let mut base = vec![b'a'; 1000];
        let mut turns = vec![t(1, &base)];
        for i in 1..2000u32 {
            base.push(b'x');
            turns.push(TurnPrefix {
                turn_id: u64::from(i + 1),
                prefix_hash: hv(&base),
                prefix_tokens: base.len() as u32,
            });
        }
        let append_mean = prefix_stability(&turns);
        assert!(
            append_mean >= 0.999,
            "append-only must stay ~1.0: {append_mean}"
        );
        // Every consecutive digest really differs: the ~1.0 verdict is not
        // trivially "identical prefixes".
        for w in turns.windows(2) {
            assert_ne!(w[0].prefix_hash, w[1].prefix_hash);
        }
        // No churn fires on a healthy growing session.
        assert!(prefix_churn(&turns, DEFAULT_STABILITY_FLOOR).is_empty());

        // (f) Rewriting: the SAME evidence blocks reordered every turn.
        // Lengths are constant, digests differ — the rewrite signature.
        let blocks: Vec<Vec<u8>> = (0..8)
            .map(|b| format!("evidence-block-{b}-").repeat(50).into_bytes())
            .collect();
        let mut turns2 = Vec::new();
        let mut perm = [0usize, 1, 2, 3, 4, 5, 6, 7];
        for turn in 1..=16u64 {
            // Deterministic permutation that differs from the previous one.
            perm.rotate_left(3);
            let mut bytes = b"HEADER".to_vec();
            for &b in &perm {
                bytes.extend_from_slice(&blocks[b]);
            }
            turns2.push(TurnPrefix {
                turn_id: turn,
                prefix_hash: hv(&bytes),
                prefix_tokens: bytes.len() as u32,
            });
        }
        for w in turns2.windows(2) {
            assert_ne!(w[0].prefix_hash, w[1].prefix_hash);
        }
        let reorder_mean = prefix_stability(&turns2);
        assert!(reorder_mean < 0.1, "reordering must be low: {reorder_mean}");
        // Churn fires on the reordering session (3rd consecutive low turn).
        let sig = prefix_churn(&turns2, DEFAULT_STABILITY_FLOOR);
        assert!(!sig.is_empty());
    }

    #[test]
    fn ten_k_turns_compute_fast_and_deterministically() {
        // (e) 10k synthetic turns, deterministic hashes: the session metric
        // must complete far under 50 ms and reproduce bit-identical results.
        let turns: Vec<TurnPrefix> = (0..10_000u64)
            .map(|i| {
                let mut h = [0u8; 32];
                h[..8].copy_from_slice(&i.to_le_bytes());
                TurnPrefix {
                    turn_id: i,
                    prefix_hash: h,
                    prefix_tokens: (i % 2000) as u32, // up/down -> mixed verdicts
                }
            })
            .collect();
        let started = std::time::Instant::now();
        let a = stability_stats(&turns);
        let churn = prefix_churn(&turns, DEFAULT_STABILITY_FLOOR);
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(50),
            "10k turns must stay <50ms, took {elapsed:?}"
        );
        // Determinism: identical input, bit-identical output.
        let b = stability_stats(&turns);
        assert_eq!(a.mean.to_bits(), b.mean.to_bits());
        assert_eq!(a.std_dev.to_bits(), b.std_dev.to_bits());
        assert_eq!(churn, prefix_churn(&turns, DEFAULT_STABILITY_FLOOR));
        assert!(a.n == 10_000 && a.mean.is_finite() && a.std_dev.is_finite());
    }

    #[test]
    fn stability_verdicts_mirror_the_documented_pair_rules() {
        // A mixed series exercises every documented branch of the definition.
        let p1 = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let p2 = b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"; // rewrite, same length -> 0
        let mut grown = p1.to_vec();
        grown.extend_from_slice(p2); // append-consistent -> 32/64 = 0.5
        let shrunk = grown[..40].to_vec(); // shrink after growth -> 0
        let mut empty_turn = t(3, &grown);
        empty_turn.prefix_tokens = 0; // empty convention -> 1.0
        let turns = vec![t(1, p1), t(2, p2), t(3, &grown), t(4, &shrunk), empty_turn];
        let got = turn_stabilities(&turns);
        let want = vec![1.0, 0.0, 0.5, 0.0, 1.0];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-12, "got {got:?} want {want:?}");
        }
    }
}
