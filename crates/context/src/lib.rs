//! kilop-context — bounded context construction, the durable task ledger, and
//! a compaction engine that cannot death-spiral.
//!
//! Five memory classes (spec §8): immutable instructions, durable task state,
//! repository knowledge, recent conversation, historical artifacts. The
//! budget is enforced BEFORE anything is sent to a provider (spec §9); a
//! successful compaction must achieve the configured minimum reduction.

pub mod artifact;
pub mod assembler;
pub mod budget;
pub mod compactor;
pub mod estimator;
pub mod ledger;
pub mod wire_plan;

pub use artifact::{ArtifactRef, ArtifactWriter};
pub use assembler::{Evidence, RecentTurn};
pub use budget::ContextBudget;
pub use compactor::{CompactionPlan, CompactionRequest, CompactionStrategy, Compactor, Summarizer};
pub use estimator::Estimator;
pub use ledger::{TaskLedger, TurnSummary};
pub use wire_plan::{plan_wire_request, WirePlan};
