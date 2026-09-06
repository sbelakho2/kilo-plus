//! Campaign (b): CAS put/read across a crash mid-write (P0-76, 500 seeds).
//!
//! Rigid per-seed recipe over a real `faktor-cas` (temp+fsync+rename write
//! path):
//!
//! ```text
//! op 0  put A (fresh write)
//! op 1  put_reader B (streaming write path)
//! op 2  strict read of A
//! op 3  external adversary: blob A's file is replaced by garbage bytes
//! op 4  put A again (the dedup path must detect corruption and REPAIR)
//! op 5  strict read of A (verified bytes back)
//! ```
//!
//! Boundaries are the cas seam's durability crossings: after the temp blob
//! is fsynced but before the atomic rename (no blob may exist at the
//! address; the leftover temp must never be served), and after the rename
//! (the blob is fully in place — a partial file can never appear under a
//! valid-looking address). A and B are seed-derived and always distinct
//! (a dedup hit would skip the write path and make the boundary
//! unreachable); content is hostile-inclusive (empty blobs, binary, up to
//! ~32 KiB, max-int-derived bytes).
//!
//! Certification per (seed, boundary):
//! 1. reopen the crashed cas and prove the DURABLE world equals the
//!    expected prefix view — exactly the declared class (`PreOp` = the
//!    in-flight blob is absent, or the corruption stays corrupt when the
//!    crashed op was the REPAIR; `FullyCommitted` = the blob is valid);
//!    a strict read of a corrupt blob is a typed error, never silent
//!    garbage;
//! 2. replay the op tail (re-issue from the durable cursor);
//! 3. the recovered world EQUALS the uninterrupted reference world, and
//!    tmp/ holds at most the crashed run's own debris (never an address).

use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};

use faktor_cas::{Cas, CrashArm};
use faktor_core::hash::FileHash;

use super::{check_equals, BoundarySpec, Campaign, CrashClass, Lcg, WorldState};

/// Seeds of the full [fault]-gated campaign.
pub const FULL_SEEDS: u64 = 500;
/// Seeds of the normal-mode smoke run.
pub const SMOKE_SEEDS: u64 = 3;

const OPS: usize = 6;

// ---------------------------------------------------------------------------
// Seed-derived content
// ---------------------------------------------------------------------------

fn blob_a(seed: u64) -> Vec<u8> {
    let mut lcg = Lcg::new(seed ^ 0xA11CE);
    if seed.is_multiple_of(7) {
        return Vec::new();
    }
    let n = 1 + lcg.below(32 * 1024) as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(lcg.next_u64() as u8);
    }
    // Stable distinguishing head byte: A can never equal B.
    out[0] = 0xA5;
    out
}

fn blob_b(seed: u64) -> Vec<u8> {
    let mut a = blob_a(seed ^ 0x8000_0001);
    a.splice(0..0, b"B".iter().copied());
    a
}

fn garbage(seed: u64) -> Vec<u8> {
    let mut lcg = Lcg::new(seed ^ 0x6AAB);
    let n = 1 + lcg.below(64) as usize;
    (0..n).map(|_| lcg.next_u64() as u8).collect()
}

fn hash_of(bytes: &[u8]) -> FileHash {
    FileHash::from(blake3::hash(bytes).into())
}

// ---------------------------------------------------------------------------
// Recipe ops
// ---------------------------------------------------------------------------

fn blob_path(cas: &Cas, hash: FileHash) -> std::path::PathBuf {
    cas.root()
        .join(&hash.to_hex()[..2])
        .join(&hash.to_hex()[2..])
}

fn exec_op(cas: &Cas, seed: u64, op: usize, a: FileHash) {
    match op {
        0 => assert_eq!(cas.put(&blob_a(seed)).expect("put A"), a),
        1 => {
            let hb = cas
                .put_reader(std::io::Cursor::new(blob_b(seed)))
                .expect("put_reader B");
            assert_ne!(hb, a, "B content must be distinct (recipe invariant)");
        }
        2 => {
            let got = cas.get_verified_now(a).expect("strict read A");
            assert_eq!(got, blob_a(seed), "A reads back exactly");
        }
        3 => {
            fs::write(blob_path(cas, a), garbage(seed)).expect("adversary corrupts A");
        }
        4 => {
            let repaired = cas.put(&blob_a(seed)).expect("repair put A");
            assert_eq!(repaired, a, "repair addresses the same content");
        }
        5 => {
            let got = cas.get_verified_now(a).expect("A after repair");
            assert_eq!(got, blob_a(seed));
            assert!(cas.verify_integrity().is_empty(), "store healthy");
        }
        _ => unreachable!("op index {op} out of the rigid recipe"),
    }
}

// ---------------------------------------------------------------------------
// Canonical world-state: every shard blob at its address with its VERIFIED
// content. tmp/ debris is deliberately outside the state (it is asserted
// separately); corrupt blobs are a typed view, never silent content.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum BlobView {
    /// The blob at the address is valid and decodes to these bytes.
    Valid(Vec<u8>),
    /// The blob at the address exists but FAILS strict verification.
    Corrupt,
    /// No blob at the address.
    Absent,
}

fn view_of(cas: &Cas, hash: FileHash) -> BlobView {
    match cas.get_verified_now(hash) {
        Ok(bytes) => BlobView::Valid(bytes),
        Err(faktor_cas::CasError::NotFound(_)) => BlobView::Absent,
        Err(_) => BlobView::Corrupt,
    }
}

fn tmp_debris(cas: &Cas) -> Vec<String> {
    let mut out: Vec<String> = fs::read_dir(cas.root().join("tmp"))
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

fn dump_world(cas: &Cas, a: FileHash, b: FileHash) -> WorldState {
    let mut lines = Vec::new();
    for (tag, h) in [("A", a), ("B", b)] {
        match view_of(cas, h) {
            BlobView::Valid(bytes) => lines.push(format!("blob:{tag}:valid:{}", hex(&bytes))),
            BlobView::Corrupt => lines.push(format!("blob:{tag}:corrupt")),
            BlobView::Absent => lines.push(format!("blob:{tag}:absent")),
        }
    }
    WorldState { lines }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn completed_count(crashed_op: usize, class: &CrashClass) -> usize {
    match class {
        CrashClass::PreOp => crashed_op,
        CrashClass::FullyCommitted => crashed_op + 1,
        CrashClass::Ambiguous => unreachable!("campaign (b) declares no ambiguous boundary"),
    }
}

fn replay_from(crashed_op: usize, class: &CrashClass) -> usize {
    completed_count(crashed_op, class)
}

// ---------------------------------------------------------------------------
// The campaign
// ---------------------------------------------------------------------------

/// (name, crashed op index, class, seam point, seam ordinal)
const BOUNDARIES: &[(&str, usize, CrashClass, &str, u64)] = &[
    ("fresh.tmp", 0, CrashClass::PreOp, "cas_tmp", 0),
    (
        "fresh.renamed",
        0,
        CrashClass::FullyCommitted,
        "cas_renamed",
        0,
    ),
    ("stream.tmp", 1, CrashClass::PreOp, "cas_stream_tmp", 0),
    (
        "stream.renamed",
        1,
        CrashClass::FullyCommitted,
        "cas_stream_renamed",
        0,
    ),
    // The repair put of corrupt A crashes before its rename: the corrupt
    // blob must STILL be there, and the address must never hold a partial.
    ("repair.tmp", 4, CrashClass::PreOp, "cas_tmp", 1),
    (
        "repair.renamed",
        4,
        CrashClass::FullyCommitted,
        "cas_renamed",
        1,
    ),
];

static BOUNDARY_SPECS: &[BoundarySpec] = &[
    BoundarySpec {
        name: "fresh.tmp",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "fresh.renamed",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "stream.tmp",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "stream.renamed",
        class: CrashClass::FullyCommitted,
    },
    BoundarySpec {
        name: "repair.tmp",
        class: CrashClass::PreOp,
    },
    BoundarySpec {
        name: "repair.renamed",
        class: CrashClass::FullyCommitted,
    },
];

fn reference(seed: u64) -> Result<WorldState, String> {
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let cas = Cas::open(dir.path().join("cas")).map_err(|e| e.to_string())?;
    let a = hash_of(&blob_a(seed));
    for op in 0..OPS {
        exec_op(&cas, seed, op, a);
    }
    let hb = hash_of(&blob_b(seed));
    let world = dump_world(&cas, a, hb);
    assert!(tmp_debris(&cas).is_empty(), "reference leaves no debris");
    drop(cas);
    Ok(world)
}

/// Durable prefix oracle: the expected world after exactly `completed` ops
/// (no crash debris — the crashed run's own debris is asserted separately).
fn prefix_world_dump(seed: u64, completed: usize, a: FileHash) -> WorldState {
    let dir = tempfile::tempdir().expect("tempdir");
    let cas = Cas::open(dir.path().join("cas")).expect("cas opens");
    for op in 0..completed {
        exec_op(&cas, seed, op, a);
    }
    let hb = hash_of(&blob_b(seed));
    let world = dump_world(&cas, a, hb);
    drop(cas);
    world
}

fn crash_run(seed: u64, boundary: &BoundarySpec) -> Result<WorldState, String> {
    let (_, crashed_op, class, point, ordinal) = BOUNDARIES
        .iter()
        .find(|(name, ..)| *name == boundary.name)
        .ok_or_else(|| format!("unknown boundary {:?}", boundary.name))?;
    let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
    let a = hash_of(&blob_a(seed));
    let hb = hash_of(&blob_b(seed));

    // --- crashed run: fresh cas, armed seam, ops through the crashed op.
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let cas = Cas::open(dir.path().join("cas")).expect("cas opens");
        cas.crash_arm(CrashArm {
            point,
            ordinal: *ordinal,
        });
        for op in 0..=*crashed_op {
            exec_op(&cas, seed, op, a);
        }
    }));
    super::expect_crash_fired(caught)?;

    // --- reopen: the durable world == the expected prefix (declared class).
    let reopened = Cas::open(dir.path().join("cas")).map_err(|e| e.to_string())?;
    let durable = dump_world(&reopened, a, hb);
    let expected = prefix_world_dump(seed, completed_count(*crashed_op, class), a);
    if durable != expected {
        return Err(format!(
            "durable cas world after crash diverges from the declared {} class:\n{}",
            class_name(class),
            durable.diff(&expected)
        ));
    }
    // The crashed run's own debris is bounded and never at an address: a
    // temp file name embeds the hash it was writing, but it lives under
    // tmp/ — the ADDRESS itself never holds a partial file (the durable
    // view checks above prove absent-or-valid-or-exact-corrupt-residue).
    let debris = tmp_debris(&reopened);
    if debris.len() > 1 {
        return Err(format!(
            "at most the crashed temp file may remain, got {debris:?}"
        ));
    }
    // A PreOp crash of the REPAIR must leave a strict read failing LOUDLY
    // (typed error — the corrupt bytes are never served).
    if *crashed_op == 4 && *class == CrashClass::PreOp {
        match reopened.get_verified_now(a) {
            Err(e) if !matches!(e, faktor_cas::CasError::NotFound(_)) => {}
            other => {
                return Err(format!(
                    "strict read after a PreOp repair crash must be a typed corruption error, got {other:?}"
                ))
            }
        }
    }

    // --- recovery contract: replay the op tail from the durable cursor.
    for op in replay_from(*crashed_op, class)..OPS {
        exec_op(&reopened, seed, op, a);
    }
    let recovered = dump_world(&reopened, a, hb);
    Ok(recovered)
}

fn class_name(c: &CrashClass) -> &'static str {
    match c {
        CrashClass::PreOp => "PreOp",
        CrashClass::FullyCommitted => "FullyCommitted",
        CrashClass::Ambiguous => "Ambiguous",
    }
}

pub fn campaign() -> Campaign {
    Campaign {
        name: "cas-put-read-crash-mid-write",
        boundaries: BOUNDARY_SPECS,
        reference,
        crash_run,
        check: check_equals,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn smoke_cas_crash_mid_write() {
    let c = campaign();
    let checks = super::run_campaign(&c, SMOKE_SEEDS).expect("smoke must pass");
    assert_eq!(checks, SMOKE_SEEDS * c.boundaries.len() as u64);
}

#[test]
#[ignore = "[fault] CAS put/read crash at every mid-write boundary, 500 seeds"]
fn full_cas_crash_mid_write() {
    let c = campaign();
    let checks = super::run_campaign(&c, FULL_SEEDS).expect("full campaign must pass");
    assert_eq!(checks, FULL_SEEDS * c.boundaries.len() as u64);
}
