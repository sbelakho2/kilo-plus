//! P0-59 ConPTY lifecycle certification (Windows only): exercises the REAL
//! wave-13 ConPTY backend of faktor-pty (`CreatePseudoConsole` +
//! `PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE`) against live processes.
//!
//! Two guarantees under test:
//! 1. `Pty::kill` ends a live sleeper attached to the session: the OS
//!    terminates the attached client when the pseudoconsole closes, with
//!    the bounded `TerminateProcess` fallback.
//! 2. Dropping the `Pty` (the daemon-crash equivalent for a session) kills
//!    every process attached to the session — a `Start-Process
//!    -NoNewWindow` grandchild shares the pseudoconsole console, so
//!    session close must take it too, not just the direct child.
//!
//! Tree shape (deterministic on CI): powershell.exe is the ConPTY client;
//! it starts a ping.exe grandchild (-NoNewWindow => same console session,
//! NOT a detached console) that writes its pid to a file and sleeps
//! ~60 s, then sleeps 60 s itself.
//!
//! Every liveness check is a bounded poll (10 s ceiling, 100 ms steps) so a
//! slow CI reaper can never hang the suite; ping self-terminates after
//! ~60 s, so even a failed assertion cannot leave an eternal sleeper.

#![cfg(windows)]

use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
use windows_sys::Win32::System::Threading::{
    OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE,
};

use faktor_pty::{Pty, PtyConfig};

static PIDFILE_SEQ: AtomicU32 = AtomicU32::new(0);

/// Direct Win32 liveness probe; see crates/winjob/tests/windows_tree.rs for
/// the semantics (an open process handle stays valid after termination).
fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    unsafe {
        let handle = OpenProcess(PROCESS_SYNCHRONIZE, 0, pid);
        if handle.is_null() {
            return false;
        }
        let running = WaitForSingleObject(handle, 0) == WAIT_TIMEOUT;
        CloseHandle(handle);
        running
    }
}

fn wait_until<F: FnMut() -> bool>(what: &str, limit: Duration, mut cond: F) {
    let deadline = Instant::now() + limit;
    while Instant::now() < deadline {
        if cond() {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("timed out after {limit:?} waiting for {what}");
}

fn pid_file_path() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "kp-pty-lifecycle-{}-{}.pid",
        std::process::id(),
        PIDFILE_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// powershell sleeps 60 s; the ping grandchild is created ~1.5 s in (once
/// powershell is a settled ConPTY client), runs -NoNewWindow so it shares
/// the pseudoconsole console, writes its pid, and sleeps ~60 s.
fn sleeper_tree_script(pid_file: &Path) -> String {
    format!(
        "Start-Sleep -Milliseconds 1500; \
         $p = Start-Process -FilePath 'ping.exe' -ArgumentList '-n','60','127.0.0.1' \
             -NoNewWindow -PassThru; \
         [System.IO.File]::WriteAllText('{}', [string]$p.Id); \
         Start-Sleep -Seconds 60",
        pid_file.display()
    )
}

fn pty_config(script: &str) -> PtyConfig {
    PtyConfig {
        command: "powershell.exe".into(),
        args: vec![
            "-NoProfile".into(),
            "-NonInteractive".into(),
            "-Command".into(),
            script.into(),
        ],
        ..Default::default()
    }
}

fn read_grandchild_pid(pid_file: &Path) -> u32 {
    wait_until("grandchild pid file", Duration::from_secs(20), || {
        pid_file.exists()
    });
    let pid: u32 = std::fs::read_to_string(pid_file)
        .expect("grandchild pid file readable")
        .trim()
        .parse()
        .expect("grandchild pid file holds a pid");
    assert_ne!(pid, 0);
    pid
}

/// `Pty::kill` must terminate a live ConPTY child promptly (session close +
/// bounded TerminateProcess fallback), and the pty must report it dead.
#[test]
fn kill_terminates_the_live_conpty_child() {
    let script = "Start-Sleep -Seconds 30".to_string();
    let mut pty = Pty::spawn(&pty_config(&script)).expect("ConPTY spawn on CI");
    let child = pty.pid();
    assert!(
        pid_alive(child),
        "powershell must be alive right after spawn"
    );

    std::thread::sleep(Duration::from_millis(800));
    assert!(pty.is_alive(), "powershell must survive its engine startup");

    pty.kill();

    assert!(!pty.is_alive(), "kill must mark the session closed");
    wait_until("conpty child death", Duration::from_secs(10), || {
        !pid_alive(child)
    });
}

/// Dropping the Pty (session owner gone = the pty-side crash) must kill the
/// direct child AND the -NoNewWindow grandchild sharing the session.
#[test]
fn dropping_the_pty_kills_the_whole_session_tree() {
    let pid_file = pid_file_path();
    let script = sleeper_tree_script(&pid_file);
    let pty = Pty::spawn(&pty_config(&script)).expect("ConPTY spawn on CI");
    let child = pty.pid();
    let grandchild = read_grandchild_pid(&pid_file);

    assert!(
        pid_alive(child) && pid_alive(grandchild),
        "powershell + ping grandchild must be alive pre-drop"
    );

    drop(pty); // kill() path: ClosePseudoConsole -> attached clients die

    wait_until(
        "whole-session death after drop",
        Duration::from_secs(10),
        || !pid_alive(child) && !pid_alive(grandchild),
    );
    let _ = std::fs::remove_file(&pid_file);
}
