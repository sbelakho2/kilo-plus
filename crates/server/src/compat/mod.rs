//! v7.5.6 / SDK compatibility glue (frozen migration surface).
//!
//! These handlers are the deliberately separate legacy surface (architecture
//! §16): `sdk` carries the SDK-shaped REST aliases, `v756` the frozen wire
//! contract. Compatibility code may depend on the native layer's shared
//! glue; the reverse dependency is forbidden and source-scan tested.

use faktor_core::state::AgentState;

pub(crate) mod sdk;
pub(crate) mod v756;

pub(crate) use sdk::*;
pub(crate) use v756::*;

/// Submit a prompt synchronously so the HTTP response carries the TRUE
/// queued state, then spawn the turn (or the queue runner) detached
/// (audit round 6). Returns the receipt's queued flag.
pub(crate) fn submit_and_run(
    agent: &std::sync::Arc<faktor_agent::AgentRuntime>,
    session: faktor_core::id::SessionId,
    prompt: &str,
    files: &[String],
    model: Option<String>,
) -> faktor_core::Result<faktor_session::PromptReceipt> {
    let receipt = agent.submit(session, prompt, files)?;
    let queued = receipt.queued;
    let agent2 = agent.clone();
    if queued {
        // The prompt durably queued behind the active logical turn; the
        // per-session runner delivers it after that turn completes.
        tokio::spawn(async move {
            agent2.run_session_queue(session).await;
        });
    } else {
        let handle = match agent2.deps().session.get_session(session) {
            Ok(Some(h)) => h,
            _ => return Ok(receipt),
        };
        let receipt2 = receipt.clone();
        tokio::spawn(async move {
            let _ = agent2.drive_receipt(&handle, receipt2, model).await;
        });
    }
    Ok(receipt)
}

/// The states that mean "a logical turn is occupying the session machine"
/// (the wait condition of `POST /session/{id}/message`). Everything else —
/// Idle, ReadyForNextTurn, Completed, Cancelled, FailedRecoverable/
/// FailedPermanent, NeedsUserInput, Suspended — means the accepted turn has
/// finished (or never started).
pub(crate) fn turn_machine_busy(s: AgentState) -> bool {
    matches!(
        s,
        AgentState::Preparing
            | AgentState::BuildingContext
            | AgentState::WaitingForModel
            | AgentState::Streaming
            | AgentState::ToolRequested
            | AgentState::WaitingForPermission
            | AgentState::ExecutingTool
            | AgentState::Validating
            | AgentState::UpdatingMemory
    )
}
