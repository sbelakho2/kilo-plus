//! kilop-core — pure core types for the Kilo+ runtime.
//!
//! This crate has **no workspace dependencies** and no I/O. Every other crate
//! depends on these types; dependencies point inward. It provides:
//!
//! - IDs (`SessionId`, `WorkspaceId`, `WorktreeId`, `TaskId`, `OpId`, `EventSeq`)
//! - typed errors with retryability classification
//! - the explicit session state machine (no implicit async state)
//! - the append-only event journal contract
//! - operation metadata (deadline, retry policy, cancellation, recovery)
//! - cancellation tokens (std-only, no tokio)
//! - injectable clocks and deadlines
//! - resource classes and budgets
//! - sandbox capabilities and permission decisions
//! - model capabilities (provider behavior lives here, not in the agent)

pub mod cancellation;
pub mod capability;
pub mod error;
pub mod event;
pub mod hash;
pub mod id;
pub mod model;
pub mod op;
pub mod resource;
pub mod retry;
pub mod state;
pub mod time;

pub use cancellation::CancellationToken;
pub use capability::{Capability, NetworkPolicy, PermissionDecision};
pub use error::{Error, ErrorKind, Result};
pub use event::{Event, EventKind};
pub use hash::FileHash;
pub use id::{EventSeq, OpId, SessionId, TaskId, WorkspaceId, WorktreeId};
pub use model::{ModelCapabilities, ReasoningMode};
pub use op::{EffectStatus, OpMeta, OpState, RecoveryStrategy};
pub use resource::{ResourceClass, ResourceLimits};
pub use retry::{RetryClass, RetryPolicy};
pub use state::{AgentState, SessionLifecycle, StateMachine};
pub use time::{Clock, Deadline, SystemClock, TestClock};

/// The Kilo+ daemon version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The frozen protocol version this build speaks.
pub const PROTOCOL_V756: &str = "v756";
/// The frozen baseline UX this build targets.
pub const UX_BASELINE: &str = "kilo-v7.5.6";

/// Every file/tool call explicitly carries its workspace identity.
/// There is no global mutable "current directory" in the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkspaceIdentity {
    pub workspace_id: WorkspaceId,
    pub worktree_id: WorktreeId,
    pub task_id: TaskId,
}

impl WorkspaceIdentity {
    pub const fn new(workspace_id: WorkspaceId, worktree_id: WorktreeId, task_id: TaskId) -> Self {
        Self {
            workspace_id,
            worktree_id,
            task_id,
        }
    }
}

/// Zero is never a valid identifier; ID newtype constructors assert on 0.
#[allow(dead_code)]
pub(crate) fn reject_zero(raw: u64, what: &str) -> u64 {
    assert!(raw != 0, "kilo-core invariant violated: {what} cannot be 0");
    raw
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_zero_is_enforced() {
        let panicked = std::panic::catch_unwind(|| reject_zero(0, "test")).is_err();
        assert!(panicked);
        assert_eq!(reject_zero(1, "test"), 1);
    }
}
