//! Deterministic crash-certification campaigns (P0-76).
//!
//! A campaign certifies, across `seeds` seeded runs, that a layer's crash
//! recovery converges to the interrupted-free reference state for EVERY
//! deterministic durability boundary the layer declares — and that the
//! interrupted state itself lands in exactly the declared equality class
//! (`PreOp` == the state before the op, `FullyCommitted` == the op's writes
//! are durable, `Ambiguous` == only where the layer declares the residue
//! unclassifiable and refuses to guess).
//!
//! Isolation contract: every seed and every (seed, boundary) pair runs on
//! its own throwaway tempdir; no campaign mutates shared state. Every
//! failure surfaces the `(seed, boundary, expected-vs-actual)` triple
//! through [`CampaignFailure`]; hostile input (zero/huge seed counts,
//! empty or duplicate boundary tables) is a typed rejection, never a
//! silent skip.

use std::fmt;

mod cas;
mod edit_txn;
mod scheduler_dag;
mod store_journal;

/// The declared equality class of one crash boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashClass {
    /// The interrupted op's writes were fully durable (or recovery
    /// re-completed them); the recovered world equals the reference world.
    FullyCommitted,
    /// The interrupted op rolled back completely; the durable world equals
    /// the pre-op state (recovery re-issues the op to reach the reference).
    PreOp,
    /// Only the layer itself may declare a residue ambiguous. Recovery must
    /// refuse it with a typed error and never clobber or silently accept.
    Ambiguous,
}

/// One deterministic crash boundary of a campaign. `name` is the campaign's
/// key for arming the layer seam and must be unique within the campaign.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundarySpec {
    pub name: &'static str,
    pub class: CrashClass,
}

/// Canonical world-state view after crash + recovery (or uninterrupted run).
/// Lines are sorted/ordered deterministically per campaign so string
/// equality IS state equality for the campaign's comparison scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorldState {
    pub lines: Vec<String>,
}

impl WorldState {
    // Kept as the canonical world-state rendering API for campaign
    // comparison tooling (the lib-test harness exercises some campaigns
    // more than others; the API stays uniform across them).
    #[allow(dead_code)]
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    fn diff(&self, other: &WorldState) -> String {
        let mut out = Vec::new();
        for (i, (a, b)) in self.lines.iter().zip(other.lines.iter()).enumerate() {
            if a != b {
                out.push(format!("line {i}: recovered={a:?} reference={b:?}"));
                if out.len() >= 8 {
                    break;
                }
            }
        }
        let (a, b) = (self.lines.len(), other.lines.len());
        if a != b {
            out.push(format!("line count: recovered={a} reference={b}"));
        }
        if out.is_empty() {
            out.push("states differ outside the first 8 lines".into());
        }
        out.join("\n")
    }
}

/// A full crash campaign. Every fn runs on a FRESH isolated state per call
/// (seed-derived content, throwaway tempdirs).
pub struct Campaign {
    pub name: &'static str,
    pub boundaries: &'static [BoundarySpec],
    /// Run the operation uninterrupted; return the reference world state.
    pub reference: fn(u64) -> Result<WorldState, String>,
    /// Build a fresh state, crash at `boundary`, run the layer's recovery
    /// (including the re-issue the class semantics require), and return the
    /// recovered world state.
    pub crash_run: fn(u64, &BoundarySpec) -> Result<WorldState, String>,
    /// Class-aware comparison of the recovered world against the reference
    /// (and, for `Ambiguous` boundaries, the boundary name the residue was
    /// declared for).
    pub check: fn(&WorldState, &WorldState, &BoundarySpec) -> Result<(), String>,
}

/// Upper bound on one campaign's seed count (typed rejection above it).
pub const MAX_CAMPAIGN_SEEDS: u64 = 100_000;

/// Typed failure of [`run_campaign`]. The triple (campaign, seed, boundary)
/// is always present except on whole-campaign rejections.
#[derive(Debug)]
pub enum CampaignFailure {
    /// Zero or absurd seed counts are rejected typed.
    SeedsInvalid {
        campaign: &'static str,
        got: u64,
    },
    /// A boundary table with duplicates is a typed rejection (a duplicate
    /// crash point would make the arm table ambiguous).
    DuplicateBoundary {
        campaign: &'static str,
        boundary: &'static str,
    },
    EmptyBoundaries {
        campaign: &'static str,
    },
    /// The reference run is not reproducible for this seed.
    Nondeterministic {
        campaign: &'static str,
        seed: u64,
        detail: String,
    },
    /// (seed, boundary, expected-vs-actual).
    Mismatch {
        campaign: &'static str,
        seed: u64,
        boundary: &'static str,
        class: CrashClass,
        detail: String,
    },
    /// The crashed run itself failed (typed layer refusal of a boundary
    /// that must crash, an unpredicted durable residue, ...).
    RunFailed {
        campaign: &'static str,
        seed: u64,
        boundary: &'static str,
        detail: String,
    },
}

impl fmt::Display for CampaignFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CampaignFailure::SeedsInvalid { campaign, got } => write!(
                f,
                "campaign {campaign}: hostile seed count {got} rejected typed (valid: 1..={MAX_CAMPAIGN_SEEDS})"
            ),
            CampaignFailure::DuplicateBoundary { campaign, boundary } => write!(
                f,
                "campaign {campaign}: duplicate crash point {boundary:?} rejected typed"
            ),
            CampaignFailure::EmptyBoundaries { campaign } => {
                write!(f, "campaign {campaign}: empty boundary table rejected typed")
            }
            CampaignFailure::Nondeterministic {
                campaign,
                seed,
                detail,
            } => write!(
                f,
                "campaign {campaign} seed {seed:#x}: reference run is not reproducible: {detail}"
            ),
            CampaignFailure::Mismatch {
                campaign,
                seed,
                boundary,
                class,
                detail,
            } => write!(
                f,
                "campaign {campaign}: (seed {seed:#x}, boundary {boundary:?}, class {class:?}) expected-vs-actual mismatch:\n{detail}"
            ),
            CampaignFailure::RunFailed {
                campaign,
                seed,
                boundary,
                detail,
            } => write!(
                f,
                "campaign {campaign}: (seed {seed:#x}, boundary {boundary:?}) crashed run failed: {detail}"
            ),
        }
    }
}

impl std::error::Error for CampaignFailure {}

/// Execute `campaign` over `seeds` seeds: for each seed the reference world
/// is built TWICE (a determinism proof) and then every boundary is crashed,
/// recovered and compared through the campaign's class-aware check.
///
/// Returns the number of (seed, boundary) checks executed. Fails fast on
/// the first typed rejection or mismatch.
pub fn run_campaign(campaign: &Campaign, seeds: u64) -> Result<u64, CampaignFailure> {
    if seeds == 0 || seeds > MAX_CAMPAIGN_SEEDS {
        return Err(CampaignFailure::SeedsInvalid {
            campaign: campaign.name,
            got: seeds,
        });
    }
    if campaign.boundaries.is_empty() {
        return Err(CampaignFailure::EmptyBoundaries {
            campaign: campaign.name,
        });
    }
    let mut seen = std::collections::HashSet::new();
    for b in campaign.boundaries {
        if !seen.insert(b.name) {
            return Err(CampaignFailure::DuplicateBoundary {
                campaign: campaign.name,
                boundary: b.name,
            });
        }
    }
    let mut checks = 0u64;
    for seed in 0..seeds {
        let reference = match (campaign.reference)(seed) {
            Ok(r) => r,
            Err(detail) => {
                return Err(CampaignFailure::RunFailed {
                    campaign: campaign.name,
                    seed,
                    boundary: "<reference>",
                    detail,
                })
            }
        };
        match (campaign.reference)(seed) {
            Ok(again) if again == reference => {}
            Ok(_) => {
                return Err(CampaignFailure::Nondeterministic {
                    campaign: campaign.name,
                    seed,
                    detail: "second reference run diverged from the first".into(),
                })
            }
            Err(detail) => {
                return Err(CampaignFailure::RunFailed {
                    campaign: campaign.name,
                    seed,
                    boundary: "<reference>",
                    detail,
                })
            }
        }
        for boundary in campaign.boundaries {
            let recovered = match (campaign.crash_run)(seed, boundary) {
                Ok(r) => r,
                Err(detail) => {
                    return Err(CampaignFailure::RunFailed {
                        campaign: campaign.name,
                        seed,
                        boundary: boundary.name,
                        detail,
                    })
                }
            };
            if let Err(detail) = (campaign.check)(&recovered, &reference, boundary) {
                return Err(CampaignFailure::Mismatch {
                    campaign: campaign.name,
                    seed,
                    boundary: boundary.name,
                    class: boundary.class,
                    detail,
                });
            }
            checks += 1;
        }
    }
    Ok(checks)
}

// ---------------------------------------------------------------------------
// Deterministic seeded LCG (numerical-recipe constants, u64 wrapping only:
// deterministic across platforms, exactly like the scheduler modelcheck LCG)
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

    /// Uniform in `0..n` (`n > 0`).
    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        self.next_u64() % n
    }

    // Kept as the seeded-draw API for campaign fault injection; the
    // lib-test harness exercises the campaigns it needs per run.
    #[allow(dead_code)]
    pub fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

// ---------------------------------------------------------------------------
// Generic assertions used by every campaign's class-aware check
// ---------------------------------------------------------------------------

/// Equality check for `FullyCommitted` / `PreOp` boundaries: after recovery
/// (and the re-issue the class semantics require) the recovered world must
/// EQUAL the reference world.
pub fn check_equals(
    recovered: &WorldState,
    reference: &WorldState,
    boundary: &BoundarySpec,
) -> Result<(), String> {
    if recovered == reference {
        Ok(())
    } else {
        Err(format!(
            "boundary {boundary:?} must converge to the reference world:\n{}",
            recovered.diff(reference)
        ))
    }
}

/// A boundary whose crash must actually have fired: when the armed seam
/// point is never crossed the operation completes and the run is a silent
/// no-op certification gap — a typed failure.
pub fn expect_crash_fired(caught: std::thread::Result<()>) -> Result<(), String> {
    match caught {
        Ok(()) => Err("seam boundary was never reached: the op completed without crashing".into()),
        Err(_) => Ok(()),
    }
}

// Keep a compact representation of the campaign table for the manual
// runbook (used by the [fault]-gated tests to self-describe).
#[allow(dead_code)]
pub fn describe(campaign: &Campaign, seeds: u64) -> String {
    let boundaries: Vec<&str> = campaign.boundaries.iter().map(|b| b.name).collect();
    format!(
        "campaign {}: {} seeds, {} boundaries ({})",
        campaign.name,
        seeds,
        campaign.boundaries.len(),
        boundaries.join(", ")
    )
}

// ---------------------------------------------------------------------------
// Runner unit tests: typed rejection of hostile input
// ---------------------------------------------------------------------------

fn dummy_campaign(boundaries: &'static [BoundarySpec]) -> Campaign {
    fn reference(_seed: u64) -> Result<WorldState, String> {
        Ok(WorldState { lines: vec![] })
    }
    fn crash_run(_seed: u64, _b: &BoundarySpec) -> Result<WorldState, String> {
        Ok(WorldState { lines: vec![] })
    }
    fn check(_r: &WorldState, _x: &WorldState, _b: &BoundarySpec) -> Result<(), String> {
        Ok(())
    }
    Campaign {
        name: "dummy",
        boundaries,
        reference,
        crash_run,
        check,
    }
}

#[test]
fn hostile_seed_counts_rejected_typed() {
    let c = dummy_campaign(&[BoundarySpec {
        name: "b",
        class: CrashClass::FullyCommitted,
    }]);
    for bad in [0u64, u64::MAX, MAX_CAMPAIGN_SEEDS + 1] {
        match run_campaign(&c, bad) {
            Err(CampaignFailure::SeedsInvalid { got, .. }) => assert_eq!(got, bad),
            other => panic!("hostile seed count {bad} must be a typed rejection, got {other:?}"),
        }
    }
}

#[test]
fn duplicate_boundaries_rejected_typed() {
    let c = dummy_campaign(&[
        BoundarySpec {
            name: "dup",
            class: CrashClass::PreOp,
        },
        BoundarySpec {
            name: "dup",
            class: CrashClass::FullyCommitted,
        },
    ]);
    match run_campaign(&c, 3) {
        Err(CampaignFailure::DuplicateBoundary { boundary, .. }) => assert_eq!(boundary, "dup"),
        other => panic!("duplicate crash points must be a typed rejection, got {other:?}"),
    }
}

#[test]
fn empty_boundary_table_rejected_typed() {
    let c = dummy_campaign(&[]);
    match run_campaign(&c, 3) {
        Err(CampaignFailure::EmptyBoundaries { .. }) => {}
        other => panic!("an empty boundary table must be a typed rejection, got {other:?}"),
    }
}

#[test]
fn boundary_table_of_every_campaign_is_unique_and_nonempty() {
    for c in [
        store_journal::campaign(),
        cas::campaign(),
        edit_txn::campaign(),
        scheduler_dag::campaign(),
    ] {
        let mut seen = std::collections::HashSet::new();
        for b in c.boundaries {
            assert!(
                seen.insert(b.name),
                "campaign {} declares the boundary {:?} twice",
                c.name,
                b.name
            );
        }
        assert!(
            !c.boundaries.is_empty(),
            "campaign {} has no boundaries",
            c.name
        );
    }
}
