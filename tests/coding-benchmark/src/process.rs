//! Shared child-process lifecycle for the benchmark harness.
//!
//! One [`ManagedChild`] owns spawn, whole-group kill and bounded reap so the
//! daemon and verifier harnesses do not duplicate the helper (and so the
//! Windows signature problem cannot recur: every kill path takes `&mut self`
//! because `std::process::Child::kill` requires mutable access on all
//! platforms).

use std::io;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

/// Poll granularity while waiting for exit.
pub const POLL_GRANULARITY: Duration = Duration::from_millis(10);
/// How long a killed process group may take to reap before giving up.
pub const KILL_REAP_BUDGET: Duration = Duration::from_secs(10);

/// Spawn a command that leads its OWN process group on unix so a deadline
/// kill takes grandchildren too (zero orphans); plain spawn elsewhere.
#[cfg(unix)]
pub fn spawn_killable(command: &mut Command) -> io::Result<Child> {
    use std::os::unix::process::CommandExt;
    command.process_group(0);
    command.spawn()
}

#[cfg(not(unix))]
pub fn spawn_killable(command: &mut Command) -> io::Result<Child> {
    command.spawn()
}

/// One child plus its lifecycle. The child is always spawned through
/// [`spawn_killable`]; killing always takes the whole process group.
pub struct ManagedChild {
    child: Child,
}

impl ManagedChild {
    pub fn spawn(command: &mut Command) -> io::Result<Self> {
        Ok(Self {
            child: spawn_killable(command)?,
        })
    }

    pub fn from_child(child: Child) -> Self {
        Self { child }
    }

    pub fn id(&self) -> u32 {
        self.child.id()
    }

    pub fn try_wait(&mut self) -> io::Result<Option<std::process::ExitStatus>> {
        self.child.try_wait()
    }

    pub fn wait(&mut self) -> io::Result<std::process::ExitStatus> {
        self.child.wait()
    }

    /// Kill the child's whole process group (best effort). On unix this is
    /// killpg on the group we created at spawn; elsewhere it is the direct
    /// child's kill.
    pub fn terminate_tree(&mut self) -> io::Result<()> {
        #[cfg(unix)]
        {
            let pid = self.child.id() as i32;
            // SAFETY: killpg on our own child's group id; the pid is the
            // pgid set with process_group(0) at spawn. ESRCH is fine.
            let r = unsafe { libc::killpg(pid, libc::SIGKILL) };
            if r == 0 {
                return Ok(());
            }
            let err = io::Error::last_os_error();
            if err.raw_os_error() == Some(libc::ESRCH) {
                Ok(())
            } else {
                Err(err)
            }
        }
        #[cfg(not(unix))]
        {
            self.child.kill()
        }
    }

    /// Poll until exit or `deadline`; afterwards the WHOLE group is always
    /// killed (an exited process may still have spawned grandchildren) and
    /// reaped with a bounded budget. Returns the exit code and whether the
    /// deadline fired.
    pub fn wait_or_kill(&mut self, deadline: Instant) -> (Option<i32>, bool) {
        let mut timed_out = false;
        let mut exit = None;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    exit = status.code();
                    break;
                }
                Ok(None) => {
                    if Instant::now() >= deadline {
                        timed_out = true;
                        break;
                    }
                    std::thread::sleep(POLL_GRANULARITY);
                }
                Err(_) => break,
            }
        }
        let _ = self.terminate_tree();
        let reap_deadline = Instant::now() + KILL_REAP_BUDGET;
        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    exit = status.code();
                    break;
                }
                Ok(None) => {
                    if Instant::now() >= reap_deadline {
                        break;
                    }
                    std::thread::sleep(POLL_GRANULARITY);
                }
                Err(_) => break,
            }
        }
        (exit, timed_out)
    }
}
