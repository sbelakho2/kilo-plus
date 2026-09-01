//! Child-process ownership registry (Commandment 8: zero orphans).

use std::collections::HashMap;
use std::sync::Mutex;

use kilop_core::id::OpId;

/// A child process the session owns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedProcess {
    pub pid: u32,
    /// The operation that spawned it; when the op dies, the process must be
    /// killed or transferred deliberately.
    pub op_id: OpId,
    pub started_ms: i64,
}

/// In-memory ownership table, shared per session. Durable crash recovery is
/// the OS's parent-death handling; this registry exists so the runtime can
/// prove ownership transfers before a session ends.
#[derive(Debug, Default)]
pub struct ProcessRegistry {
    inner: Mutex<HashMap<u32, OwnedProcess>>,
}

impl ProcessRegistry {
    pub fn register(&self, proc: OwnedProcess) -> Result<(), crate::SessionError> {
        if proc.pid == 0 {
            return Err(crate::SessionError::Malformed(
                "pid 0 is not a child".into(),
            ));
        }
        let mut map = self.inner.lock().expect("process registry poisoned");
        if map.contains_key(&proc.pid) {
            return Err(crate::SessionError::Conflict(format!(
                "pid {} is already owned by this session",
                proc.pid
            )));
        }
        map.insert(proc.pid, proc);
        Ok(())
    }

    pub fn release(&self, pid: u32) -> Result<OwnedProcess, crate::SessionError> {
        self.inner
            .lock()
            .expect("process registry poisoned")
            .remove(&pid)
            .ok_or_else(|| crate::SessionError::NotFound(format!("pid {pid} is not owned")))
    }

    pub fn all(&self) -> Vec<OwnedProcess> {
        let mut out: Vec<OwnedProcess> = self
            .inner
            .lock()
            .expect("process registry poisoned")
            .values()
            .cloned()
            .collect();
        out.sort_by_key(|p| p.pid);
        out
    }

    /// Take every process (crash recovery: after a restart the children are
    /// presumed dead or re-parented; the runtime must not pretend to own
    /// zombies).
    pub fn drain(&self) -> Vec<OwnedProcess> {
        let mut map = self.inner.lock().expect("process registry poisoned");
        let out: Vec<OwnedProcess> = map.values().cloned().collect();
        map.clear();
        out
    }
}

impl crate::handle::SessionHandle {
    /// Register a child process owned by `op` (Commandment 8). Duplicate pids
    /// conflict; pid 0 is malformed.
    pub fn register_process(&self, pid: u32, op: OpId) -> kilop_core::Result<()> {
        if self.ops().tracked(op).is_none() {
            return Err(crate::SessionError::NotFound(format!(
                "operation {op} is not tracked; cannot own a process"
            ))
            .into());
        }
        self.processes()
            .register(OwnedProcess {
                pid,
                op_id: op,
                started_ms: self.now_ms(),
            })
            .map_err(Into::into)
    }

    /// Release ownership (the process exited or was deliberately transferred).
    pub fn release_process(&self, pid: u32) -> kilop_core::Result<OwnedProcess> {
        self.processes().release(pid).map_err(Into::into)
    }

    pub fn owned_processes(&self) -> kilop_core::Result<Vec<OwnedProcess>> {
        Ok(self.processes().all())
    }
}

#[cfg(test)]
mod tests {
    use crate::handle::tests::{session, test_manager};

    #[test]
    fn zero_orphans_blocks_end_session_until_released() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let r = s.submit_prompt("run tests", &[]).unwrap();
        let op = r.op_id;
        // Move the machine to Validating so end_session is legal once the
        // process is released.
        s.append_event(
            kilop_core::event::EventKind::ContextPrepared,
            kilop_core::state::AgentState::BuildingContext,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::ModelStarted,
            kilop_core::state::AgentState::WaitingForModel,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::ModelChunkReceived,
            kilop_core::state::AgentState::Streaming,
            None,
            None,
        )
        .unwrap();
        s.append_event(
            kilop_core::event::EventKind::ToolCompleted,
            kilop_core::state::AgentState::Validating,
            None,
            None,
        )
        .unwrap();
        s.register_process(4242, op).unwrap();
        // end_session must refuse while a child is owned.
        let err = s.end_session().unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Conflict);
        assert_eq!(
            s.state().unwrap(),
            kilop_core::state::AgentState::Validating
        );
        // Duplicate registration of the same pid conflicts.
        assert!(s.register_process(4242, op).is_err());
        // Releasing an unowned pid is NotFound.
        assert!(s.release_process(9999).is_err());
        // Releasing transfers ownership; now the session may end.
        let released = s.release_process(4242).unwrap();
        assert_eq!(released.pid, 4242);
        assert_eq!(released.op_id, op);
        s.end_session().unwrap();
        assert_eq!(s.state().unwrap(), kilop_core::state::AgentState::Completed);
    }

    #[test]
    fn process_requires_a_tracked_owner_op() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let err = s.register_process(1, m.next_op_id()).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::NotFound);
    }

    #[test]
    fn pid_zero_is_malformed() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let op = s.submit_prompt("x", &[]).unwrap().op_id;
        let err = s.register_process(0, op).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Malformed);
    }

    #[test]
    fn owned_processes_list_is_sorted_and_stable() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let op = s.submit_prompt("x", &[]).unwrap().op_id;
        for pid in [10, 3, 7] {
            s.register_process(pid, op).unwrap();
        }
        let pids: Vec<u32> = s
            .owned_processes()
            .unwrap()
            .into_iter()
            .map(|p| p.pid)
            .collect();
        assert_eq!(pids, vec![3, 7, 10]);
    }
}
