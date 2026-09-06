//! Executable coding benchmark (audit follow-up P0-87/88/83): real
//! checked-in mini repositories across languages, driven through the ACTUAL
//! daemon (the built `faktor-cli serve` binary, native HTTP API), scored by
//! repository-native verification plus deterministic criteria coverage,
//! with cost-to-verified-success accounting.
//!
//! Normal (`cargo test`) mode is a FAKE, in-process, no-model harness: it
//! never spawns the daemon and never contacts a provider. It proves the
//! mechanics — corpus loading, immutable corpus copies, toolchain
//! skip-vs-fail gates, `verify.sh` execution with timeout + output cap +
//! process-tree kill, deterministic criteria scoring, and the cost
//! aggregation — against the pristine (buggy) repos, which must FAIL
//! verification. Real-model runs are `#[ignore]`-gated (`tests/real.rs`)
//! and need provider keys; see `README.md` in this crate.
//!
//! Determinism/adversarial rules implemented here:
//! - the checked-in corpus is never written to (always copied to a fresh
//!   temp workspace first; a test asserts byte-identical snapshots);
//! - hostile corpus entries (traversal ids, symlinks, oversized metadata,
//!   malformed criteria) are rejected by the loader;
//! - `verify.sh` output is bounded (cap + retained tail), wall-bounded
//!   (timeout kills the whole process group), and every child of the
//!   harness dies with its parent run;
//! - a missing toolchain is a documented skip, never a failure;
//! - concurrent runs are isolated: every task workspace and every daemon
//!   data dir is its own `tempdir`; no global mutable state exists.

pub mod corpus;
pub mod daemon;
pub mod fsutil;
pub mod process;
pub mod report;
pub mod score;
pub mod toolchain;
pub mod verify;

pub use corpus::{Corpus, CorpusError, Lang, Task, CORPUS_DIR};
pub use report::{Aggregate, CorpusReport, TaskResult};
pub use score::{Criterion, CriterionResult, RecordCriteria};
pub use verify::{run_verify_sh, VerifyOptions, VerifyOutcome};

/// The corpus directory of THIS crate (checked-in task repositories).
pub fn corpus_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CORPUS_DIR)
}
