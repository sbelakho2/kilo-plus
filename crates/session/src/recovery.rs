//! Crash recovery (architecture spec section 7): unfinished operations are
//! reconstructed from durable state, never blindly re-run.
//!
//! `recover_all` scans durable `tool_run` rows still in `running` status and
//! applies their recorded `RecoveryStrategy`:
//!
//! - `VerifyHash { path, expected }` — hash the file (via the injected
//!   [`FileHasher`]); if it matches `expected`, the deterministic FS op
//!   completed before the crash (`completed`/`verified`); if not, it truly
//!   never ran (`failed`/`failed`).
//! - `MarkUnknown` — record `effect_status = unknown` and force verification
//!   instead of re-running.
//! - `Idempotent` — safe to re-run; mark failed so the scheduler may rerun.
//! - `Manual` — never re-run automatically; requires a human.
//! - `None` — no recovery action.
//!
//! The sweep is idempotent: finished rows are never re-scanned, and a second
//! `recover_all` appends nothing.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use kilop_cas::Cas;
use kilop_core::event::EventKind;
use kilop_core::hash::FileHash;
use kilop_core::id::{OpId, SessionId};
use kilop_core::op::{EffectStatus, RecoveryStrategy};
use kilop_core::state::AgentState;
use kilop_store::ToolRunRow;

use crate::handle::{SessionHandle, is_op_active};
use crate::process::OwnedProcess;
use crate::{SessionError, effect_str, MAX_VERIFY_BYTES};

/// Hashes a file for deterministic-verification recovery. Injectable so tests
/// can simulate crash states without touching the filesystem.
pub trait FileHasher: Send + Sync {
    fn hash_file(&self, path: &Path) -> kilop_core::Result<FileHash>;
}

/// Production hasher: reads the file (bounded by `max_bytes`) and computes
/// its BLAKE3 identity **through the CAS** — the workspace's canonical hashing
/// implementation. The verified content becomes a durable CAS blob as a
/// recovery audit artifact (deduplicated; bounded by the read cap).
#[derive(Debug, Clone)]
pub struct SystemFileHasher {
    cas: Arc<Cas>,
    max_bytes: usize,
}

impl SystemFileHasher {
    pub fn new(cas: Arc<Cas>) -> Self {
        Self {
            cas,
            max_bytes: MAX_VERIFY_BYTES,
        }
    }

    pub fn with_limit(cas: Arc<Cas>, max_bytes: usize) -> Self {
        Self { cas, max_bytes }
    }
}

impl FileHasher for SystemFileHasher {
    fn hash_file(&self, path: &Path) -> kilop_core::Result<FileHash> {
        let meta = fs::metadata(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                SessionError::NotFound(format!("file {path:?} does not exist"))
            } else {
                SessionError::Internal(format!("stat {path:?}: {e}"))
            }
        })?;
        if meta.len() > self.max_bytes as u64 {
            return Err(SessionError::Oversized(format!(
                "file {path:?} is {} bytes, verification bound is {}",
                meta.len(),
                self.max_bytes
            ))
            .into());
        }
        let bytes = fs::read(path)
            .map_err(|e| SessionError::Internal(format!("read {path:?}: {e}")))?;
        if bytes.len() > self.max_bytes {
            return Err(SessionError::Oversized(format!(
                "file {path:?} is {} bytes, verification bound is {}",
                bytes.len(),
                self.max_bytes
            ))
            .into());
        }
        self.cas.put(&bytes).map_err(|e| SessionError::from(e)).map_err(Into::into)
    }
}

/// What recovery decided for one crashed operation.
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryAction {
    /// Deterministic FS op: the file matches the expected hash — it
    /// completed before the crash.
    Verified { expected: FileHash, actual: FileHash },
    /// The file does not match (or does not exist): the op truly never ran.
    NotApplied { expected: FileHash, actual: Option<FileHash> },
    /// `MarkUnknown`: effects unknown, verification forced before reuse.
    UnknownEffect,
    /// `Idempotent`: safe to re-run.
    RerunAllowed,
    /// `Manual`: a human must decide.
    NeedsHuman,
    /// `None`: no recovery action.
    NoAction,
}

/// The state a crashed session may honestly land on. `FailedRecoverable` is
/// preferred; the core machine does not permit it from `ToolRequested` or
/// `WaitingForPermission`, so recovery falls back to `WaitingForPermission`
/// (the permission request is durable and resumable from that state only;
/// `NeedsUserInput` cannot be resolved back to `ExecutingTool`), and finally
/// stays put.
fn crash_target(current: AgentState) -> AgentState {
    let mut m = kilop_core::state::StateMachine::new(current);
    for t in [
        AgentState::FailedRecoverable,
        AgentState::WaitingForPermission,
        AgentState::NeedsUserInput,
    ] {
        if m.transition(t).is_ok() {
            return t;
        }
    }
    current
}

/// One recovered tool run.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveredOp {
    pub op_id: OpId,
    pub tool: String,
    /// Final durable `tool_run.status`.
    pub status: String,
    pub effect: EffectStatus,
    pub action: RecoveryAction,
}

/// The outcome of a recovery sweep over one session.
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryReport {
    pub session_id: SessionId,
    /// Session state after recovery.
    pub state: AgentState,
    pub crashed_ops: Vec<RecoveredOp>,
    /// Children owned at recovery time (post-restart: presumed dead or
    /// re-parented; reported so the runtime cannot silently lose them).
    pub orphans: Vec<OwnedProcess>,
    /// True when an op-active session had no running tool rows: the turn
    /// itself was interrupted mid-flight.
    pub interrupted_turn: bool,
    /// True when tool rows were pending while the journal said the session
    /// was idle/suspended/terminal — the rows are fixed, the state stands.
    pub contradiction: bool,
    /// True when any durable change was made.
    pub applied: bool,
}

fn parse_recovery(row: &ToolRunRow) -> Result<RecoveryStrategy, SessionError> {
    serde_json::from_value(row.recovery.clone()).map_err(|e| {
        SessionError::Malformed(format!(
            "tool_run {} carries an invalid recovery strategy: {e}",
            row.op_id
        ))
    })
}

fn apply_strategy(
    s: &SessionHandle,
    row: &ToolRunRow,
    strategy: &RecoveryStrategy,
    hasher: &dyn FileHasher,
) -> Result<RecoveredOp, SessionError> {
    let (status, effect, action) = match strategy {
        RecoveryStrategy::VerifyHash { path, expected } => {
            // The expected_hash column is redundant durability: a mismatch is
            // tampering and must be loud.
            if let Some(col) = &row.expected_hash {
                if col != &expected.to_hex() {
                    return Err(SessionError::Malformed(format!(
                        "tool_run {} expected_hash column {} disagrees with strategy {}",
                        row.op_id,
                        col,
                        expected.to_hex()
                    )));
                }
            }
            match hasher.hash_file(Path::new(path)) {
                Ok(actual) if actual == *expected => {
                    ("completed", EffectStatus::Verified, RecoveryAction::Verified { expected: *expected, actual })
                }
                Ok(actual) => (
                    "failed",
                    EffectStatus::Failed,
                    RecoveryAction::NotApplied { expected: *expected, actual: Some(actual) },
                ),
                Err(_) => (
                    // Unreadable/missing file: the op never ran.
                    "failed",
                    EffectStatus::Failed,
                    RecoveryAction::NotApplied { expected: *expected, actual: None },
                ),
            }
        }
        RecoveryStrategy::MarkUnknown => ("interrupted", EffectStatus::Unknown, RecoveryAction::UnknownEffect),
        RecoveryStrategy::Idempotent => ("failed", EffectStatus::Unknown, RecoveryAction::RerunAllowed),
        RecoveryStrategy::Manual => ("interrupted", EffectStatus::Unknown, RecoveryAction::NeedsHuman),
        RecoveryStrategy::None => ("interrupted", EffectStatus::Unknown, RecoveryAction::NoAction),
    };
    s.manager()
        .store()
        .finish_tool_run(row.session_id, row.op_id, status, effect_str(effect))
        .map_err(|e| crate::map_store_err(e))?;
    Ok(RecoveredOp {
        op_id: row.op_id,
        tool: row.tool.clone(),
        status: status.to_string(),
        effect,
        action,
    })
}

impl SessionHandle {
    /// Recover this session with the production file hasher.
    pub fn recover_all(&self) -> kilop_core::Result<RecoveryReport> {
        let hasher = self.system_hasher();
        self.recover_all_with(hasher.as_ref())
    }

    /// Recover this session, injecting a file hasher (tests).
    pub fn recover_all_with(&self, hasher: &dyn FileHasher) -> kilop_core::Result<RecoveryReport> {
        let _guard = self.command_guard();
        let session_id = self.id;
        let current = self.state()?;
        let pending = self.pending_tool_runs()?;

        // Children owned when the world stopped: after a restart they are
        // presumed dead or deliberately re-parented by the OS. Report and
        // clear — the runtime must never pretend to own zombies.
        let orphans = self.processes().drain();

        let mut report = RecoveryReport {
            session_id,
            state: current,
            crashed_ops: Vec::new(),
            orphans,
            interrupted_turn: false,
            contradiction: false,
            applied: false,
        };

        if pending.is_empty() {
            if is_op_active(current) {
                // The turn itself was interrupted (no tool row survives it).
                // Journal CrashDetected and land on the honest recovery target
                // so the agent may re-plan; never re-run the turn blindly.
                let target = crash_target(current);
                tracing::warn!(session = %session_id, state = ?current, target = ?target, "interrupted turn detected");
                self.transition_locked(
                    EventKind::CrashDetected,
                    target,
                    None,
                    Some(serde_json::json!({ "recovered_from": crate::state_tag(current) })),
                )?;
                report.state = target;
                report.interrupted_turn = true;
                report.applied = true;
            }
            return Ok(report);
        }

        // Pending tool runs: the journal alone cannot tell how far they got.
        let (crash_state, contradiction) = if is_op_active(current) {
            (crash_target(current), false)
        } else {
            // Idle/Suspended/terminal with running rows: the journal and the
            // ledger disagree. Fix the rows, keep the state, say so.
            (current, true)
        };
        report.contradiction = contradiction;

        self.transition_locked(
            EventKind::CrashDetected,
            crash_state,
            None,
            Some(serde_json::json!({
                "pending_ops": pending.len(),
                "contradiction": contradiction,
            })),
        )?;

        for row in &pending {
            let strategy = parse_recovery(row)?;
            let recovered = apply_strategy(self, row, &strategy, hasher)?;
            let action_tag = match &recovered.action {
                RecoveryAction::Verified { .. } => "verified",
                RecoveryAction::NotApplied { .. } => "not_applied",
                RecoveryAction::UnknownEffect => "unknown_effect",
                RecoveryAction::RerunAllowed => "rerun_allowed",
                RecoveryAction::NeedsHuman => "needs_human",
                RecoveryAction::NoAction => "no_action",
            };
            tracing::warn!(
                session = %session_id,
                op = %recovered.op_id,
                tool = %recovered.tool,
                action = action_tag,
                "recovered crashed operation"
            );
            self.transition_locked(
                EventKind::RecoveryApplied,
                crash_state,
                Some(recovered.op_id),
                Some(serde_json::json!({
                    "op_id": recovered.op_id.raw(),
                    "tool": recovered.tool,
                    "status": recovered.status,
                    "effect": effect_str(recovered.effect),
                    "action": action_tag,
                })),
            )?;
            report.crashed_ops.push(recovered);
        }

        report.state = crash_state;
        report.applied = true;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use kilop_core::cancellation::CancellationToken;
    use kilop_core::event::EventKind;
    use kilop_core::op::OpMeta;
    use kilop_core::time::Deadline;

    struct FakeHasher(FileHash);

    impl FileHasher for FakeHasher {
        fn hash_file(&self, _path: &Path) -> kilop_core::Result<FileHash> {
            Ok(self.0)
        }
    }

    struct NotFoundHasher;

    impl FileHasher for NotFoundHasher {
        fn hash_file(&self, _path: &Path) -> kilop_core::Result<FileHash> {
            Err(SessionError::NotFound("simulated".into()).into())
        }
    }

    fn make_meta(s: &SessionHandle, m: &crate::SessionManager, recovery: RecoveryStrategy) -> (OpMeta, OpId) {
        let op = m.next_op_id();
        let meta = OpMeta::new(
            op,
            s.id(),
            Deadline::at(m.now_ms() + 60_000),
            kilop_core::retry::RetryPolicy::default(),
            CancellationToken::new(),
            recovery,
            m.now_ms(),
        );
        (meta, op)
    }

    fn to_executing(s: &SessionHandle) {
        s.submit_prompt("x", &[]).unwrap();
        s.append_event(EventKind::ContextPrepared, AgentState::BuildingContext, None, None).unwrap();
        s.append_event(EventKind::ModelStarted, AgentState::WaitingForModel, None, None).unwrap();
        s.append_event(EventKind::ModelChunkReceived, AgentState::Streaming, None, None).unwrap();
        // A durable permission request puts the machine at WaitingForPermission
        // and is resumable after a crash.
        let turn_op = s.ops().all()[0];
        s.request_permission(turn_op, &kilop_core::capability::Capability::ReadWorkspace { path: "/w/a".into() })
            .unwrap();
    }

    #[test]
    fn recover_all_verifies_hash_and_completes() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_executing(&s);
        let expected = FileHash::from([7; 32]);
        let (meta, op) = make_meta(
            &s,
            &m,
            RecoveryStrategy::VerifyHash {
                path: "/w/a.txt".into(),
                expected,
            },
        );
        s.start_tool_run(meta, "write_file", serde_json::json!({"path": "/w/a.txt"})).unwrap();
        // "Crash": nothing else happens.
        let report = s.recover_all_with(&FakeHasher(expected)).unwrap();
        assert!(report.applied);
        assert!(!report.contradiction);
        assert!(!report.interrupted_turn);
        assert_eq!(report.crashed_ops.len(), 1);
        assert_eq!(report.crashed_ops[0].op_id, op);
        assert_eq!(report.crashed_ops[0].status, "completed");
        assert_eq!(report.crashed_ops[0].effect, EffectStatus::Verified);
        assert_eq!(
            report.crashed_ops[0].action,
            RecoveryAction::Verified { expected, actual: expected }
        );
        assert_eq!(report.state, AgentState::FailedRecoverable);
        // Journal: CrashDetected + RecoveryApplied, then the state lands.
        let kinds: Vec<_> = s.events_range(1, None).unwrap().into_iter().map(|e| e.kind).collect();
        assert!(kinds.contains(&EventKind::CrashDetected));
        assert!(kinds.contains(&EventKind::RecoveryApplied));
        assert_eq!(s.state().unwrap(), AgentState::FailedRecoverable);
    }

    #[test]
    fn recover_all_hash_mismatch_marks_never_ran() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_executing(&s);
        let expected = FileHash::from([7; 32]);
        let (meta, op) = make_meta(
            &s,
            &m,
            RecoveryStrategy::VerifyHash {
                path: "/w/a.txt".into(),
                expected,
            },
        );
        s.start_tool_run(meta, "write_file", serde_json::json!({"path": "/w/a.txt"})).unwrap();
        let actual = FileHash::from([9; 32]);
        let report = s.recover_all_with(&FakeHasher(actual)).unwrap();
        assert_eq!(report.crashed_ops[0].status, "failed");
        assert_eq!(report.crashed_ops[0].effect, EffectStatus::Failed);
        assert_eq!(
            report.crashed_ops[0].action,
            RecoveryAction::NotApplied { expected, actual: Some(actual) }
        );
    }

    #[test]
    fn recover_all_missing_file_means_never_ran() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_executing(&s);
        let expected = FileHash::from([7; 32]);
        let (meta, _op) = make_meta(
            &s,
            &m,
            RecoveryStrategy::VerifyHash {
                path: "/w/a.txt".into(),
                expected,
            },
        );
        s.start_tool_run(meta, "write_file", serde_json::json!({"path": "/w/a.txt"})).unwrap();
        let report = s.recover_all_with(&NotFoundHasher).unwrap();
        assert_eq!(report.crashed_ops[0].status, "failed");
        assert_eq!(
            report.crashed_ops[0].action,
            RecoveryAction::NotApplied { expected, actual: None }
        );
    }

    #[test]
    fn recover_all_unknown_effect_and_manual_never_rerun() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_executing(&s);
        let (meta, op_unknown) = make_meta(&s, &m, RecoveryStrategy::MarkUnknown);
        s.start_tool_run(meta, "run_test", serde_json::json!({})).unwrap();
        let (meta2, op_manual) = make_meta(&s, &m, RecoveryStrategy::Manual);
        s.start_tool_run(meta2, "deploy", serde_json::json!({})).unwrap();
        let report = s.recover_all_with(&NotFoundHasher).unwrap();
        assert_eq!(report.crashed_ops.len(), 2);
        let by_op = |o: OpId| report.crashed_ops.iter().find(|r| r.op_id == o).unwrap();
        let u = by_op(op_unknown);
        assert_eq!(u.status, "interrupted");
        assert_eq!(u.effect, EffectStatus::Unknown);
        assert_eq!(u.action, RecoveryAction::UnknownEffect);
        let man = by_op(op_manual);
        assert_eq!(man.status, "interrupted");
        assert_eq!(man.action, RecoveryAction::NeedsHuman);
        // Nothing was re-run: no new tool rows, no TurnCompleted.
        assert!(s.pending_tool_runs().unwrap().is_empty());
        assert!(!s
            .events_range(1, None)
            .unwrap()
            .iter()
            .any(|e| e.kind == EventKind::TurnCompleted));
    }

    #[test]
    fn recover_all_idempotent_no_duplicate_events() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_executing(&s);
        let expected = FileHash::from([7; 32]);
        let (meta, _op) = make_meta(
            &s,
            &m,
            RecoveryStrategy::VerifyHash {
                path: "/w/a.txt".into(),
                expected,
            },
        );
        s.start_tool_run(meta, "write_file", serde_json::json!({})).unwrap();
        let first = s.recover_all_with(&FakeHasher(expected)).unwrap();
        assert!(first.applied);
        let events_after_first = s.last_event_seq().unwrap().unwrap().raw();
        // Second sweep: nothing pending, nothing to do, no new events.
        let second = s.recover_all_with(&FakeHasher(expected)).unwrap();
        assert!(!second.applied);
        assert!(second.crashed_ops.is_empty());
        assert_eq!(s.last_event_seq().unwrap().unwrap().raw(), events_after_first);
        // Third sweep still idempotent.
        assert!(!s.recover_all_with(&FakeHasher(expected)).unwrap().applied);
    }

    #[test]
    fn recover_all_detects_interrupted_turn_without_tool_runs() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_executing(&s);
        // No tool rows: the crash hit the model stream itself.
        let report = s.recover_all_with(&NotFoundHasher).unwrap();
        assert!(report.interrupted_turn);
        assert!(report.crashed_ops.is_empty());
        // The crash hit the turn at the permission point; the durable
        // permission request survives, so the session keeps waiting on it.
        assert_eq!(report.state, AgentState::WaitingForPermission);
        assert_eq!(s.state().unwrap(), AgentState::WaitingForPermission);
        // The permission can still be resolved after recovery.
        let (_, op, _) = s.pending_permission(1).unwrap().unwrap();
        let _ = op;
        s.resolve_permission(1, kilop_core::capability::PermissionDecision::Deny).unwrap();
        assert_eq!(s.state().unwrap(), AgentState::ReadyForNextTurn);
        let kinds: Vec<_> = s.events_range(1, None).unwrap().into_iter().map(|e| e.kind).collect();
        assert_eq!(kinds.iter().filter(|k| **k == EventKind::CrashDetected).count(), 1);
        // From FailedRecoverable the user may re-prompt (never blind replay).
        s.submit_prompt("try again", &[]).unwrap();
        assert_eq!(s.state().unwrap(), AgentState::Preparing);
    }

    #[test]
    fn recover_all_terminal_contradiction_flagged_and_rows_fixed() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_executing(&s);
        let (meta, op) = make_meta(&s, &m, RecoveryStrategy::MarkUnknown);
        s.start_tool_run(meta, "run_test", serde_json::json!({})).unwrap();
        // Corrupt the journal: a ToolStarted exists but the session row is
        // forced to Completed (no legal transition does this).
        s.force_append_event(EventKind::TurnCompleted, AgentState::Completed, None, None).unwrap();
        let report = s.recover_all_with(&NotFoundHasher).unwrap();
        assert!(report.contradiction, "journal says Completed, tool row says running");
        assert_eq!(report.state, AgentState::Completed, "state stands; rows are fixed");
        assert_eq!(report.crashed_ops.len(), 1);
        assert_eq!(report.crashed_ops[0].op_id, op);
        assert!(s.pending_tool_runs().unwrap().is_empty(), "rows fixed");
    }

    #[test]
    fn journal_corruption_detected_by_replay() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.submit_prompt("x", &[]).unwrap();
        // Preparing -> Streaming skips the whole chain: corruption.
        s.force_append_event(EventKind::ModelStarted, AgentState::Streaming, None, None).unwrap();
        let err = s.replay_journal().unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Internal);
    }

    #[test]
    fn recover_all_rejects_corrupt_recovery_json() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_executing(&s);
        // Bypass the typed API: store a garbage recovery strategy directly.
        m.store()
            .start_tool_run(
                s.id(),
                m.next_op_id(),
                "run_test",
                serde_json::json!({}),
                serde_json::json!({ "strategy": "delete_everything" }),
                None,
            )
            .unwrap();
        let err = s.recover_all_with(&NotFoundHasher).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Malformed);
    }

    #[test]
    fn recover_all_expected_hash_column_mismatch_is_malformed() {
        let (_d, m) = test_manager();
        let s = session(&m);
        to_executing(&s);
        let expected = FileHash::from([7; 32]);
        // The strategy says one hash; the column says another: tampering.
        m.store()
            .start_tool_run(
                s.id(),
                m.next_op_id(),
                "write_file",
                serde_json::json!({}),
                serde_json::to_value(RecoveryStrategy::VerifyHash {
                    path: "/w/a.txt".into(),
                    expected,
                })
                .unwrap(),
                Some(FileHash::from([9; 32]).to_hex()),
            )
            .unwrap();
        let err = s.recover_all_with(&FakeHasher(expected)).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Malformed);
    }

    #[test]
    fn recover_all_orphan_processes_reported_and_cleared() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let op = s.submit_prompt("x", &[]).unwrap().op_id;
        s.register_process(1234, op).unwrap();
        let report = s.recover_all_with(&NotFoundHasher).unwrap();
        assert_eq!(report.orphans.len(), 1);
        assert_eq!(report.orphans[0].pid, 1234);
        assert!(s.owned_processes().unwrap().is_empty(), "registry cleared");
    }

    #[test]
    fn recover_all_idle_session_is_noop() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let report = s.recover_all_with(&NotFoundHasher).unwrap();
        assert!(!report.applied);
        assert!(!report.interrupted_turn);
        assert_eq!(report.state, AgentState::Idle);
        assert_eq!(s.last_event_seq().unwrap().unwrap().raw(), 1, "no events appended");
    }

    #[test]
    fn system_file_hasher_bounds_and_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let cas = Arc::new(Cas::open(dir.path().join("cas")).unwrap());
        let hasher = SystemFileHasher::with_limit(cas.clone(), 1 << 20);
        let file = dir.path().join("a.txt");
        std::fs::write(&file, b"deterministic content").unwrap();
        let h = hasher.hash_file(&file).unwrap();
        assert_eq!(cas.put(b"deterministic content").unwrap(), h);
        // Missing file -> NotFound.
        let err = hasher.hash_file(&dir.path().join("nope")).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::NotFound);
        // Oversized file -> Oversized before reading.
        let big = dir.path().join("big.bin");
        std::fs::write(&big, vec![0u8; (1 << 20) + 1]).unwrap();
        let err = hasher.hash_file(&big).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Oversized);
        // The verified content is durable in the CAS (audit artifact).
        assert!(cas.has(h));
    }
}
