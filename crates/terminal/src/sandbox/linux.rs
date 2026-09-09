//! Linux network-isolation backend (audit 4/28/35-39), compiled only on
//! `target_os = "linux"`.
//!
//! `NetworkIsolation::DenyAll` spawns install this pre-exec hook: after
//! fork, before exec, the child calls `unshare(CLONE_NEWNET)` and drops
//! into a FRESH network namespace. An empty netns is adequate: the new
//! namespace contains only loopback, and loopback is left DOWN by the
//! kernel — no bring-up is performed, so no TCP or UDP egress can leave
//! the child (a connect() to any address fails at the route/device layer).
//! No network setup code of any kind runs here.
//!
//! Fail-closed contract: if the kernel or the user-namespace policy
//! refuses the unshare (EPERM, EINVAL, ENOSYS, ...), the closure returns
//! the raw OS error, `spawn()` fails, and the supervisor surfaces a typed
//! permission refusal — the child NEVER runs unenforced and nothing is
//! logged as a warning-and-continue. std transports only the raw errno
//! out of the pre-exec child, so the caller classifies any DenyAll spawn
//! failure as the typed refusal (the OS message still names the cause).
//!
//! The closure runs in the single-threaded post-fork child and only calls
//! `unshare` and reads the errno — no allocation, no locks.

use std::io;
use std::os::unix::process::CommandExt;

/// Install the deny-all network isolation pre-exec hook on `cmd`.
///
/// # Safety
///
/// The installed closure runs in the forked child between fork and exec:
/// it must not allocate, lock, or call anything but async-signal-safe
/// operations. It calls `libc::unshare(CLONE_NEWNET)` and reads the
/// errno — nothing else.
pub(crate) unsafe fn apply_deny_all_isolation(cmd: &mut std::process::Command) {
    // SAFETY (of the pre_exec call): std requires the caller to uphold the
    // pre-exec restrictions; the closure below is allocation-free and
    // single-purpose (documented above).
    cmd.pre_exec(unshare_netns_pre_exec);
}

fn unshare_netns_pre_exec() -> io::Result<()> {
    // SAFETY: unshare(2) takes no pointer arguments; the raw errno read
    // after failure is async-signal-safe.
    let ret = unsafe { libc::unshare(libc::CLONE_NEWNET) };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
