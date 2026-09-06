//! P0-59 process-tree certification (Windows only): a `cmd`-spawned
//! parent assigned to a faktor-winjob JobGuard must take its WHOLE tree —
//! child AND grandchild — when the job is terminated explicitly
//! (`JobGuard::terminate`, the OS-enforced `taskkill /T` equivalent) and
//! when the last job handle closes (`JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`,
//! the daemon-crash guarantee: the OS does the killing, never Faktor).
//!
//! Tree shape (deterministic on CI): powershell.exe is the job member; it
//! starts a ping.exe grandchild that sleeps ~60 s and writes its pid to a
//! file, then sleeps 60 s itself. The grandchild's `Start-Process` runs
//! AFTER a 1.5 s delay so the parent is provably assigned to the job before
//! the grandchild exists — job membership for descendants of members is
//! automatic, but a grandchild born BEFORE the assignment would escape.
//!
//! Every liveness check is a bounded poll (10 s ceiling, 100 ms steps) so a
//! slow CI reaper can never hang the suite; ping self-terminates after
//! ~60 s, so even a failed assertion cannot leave an eternal sleeper.

#![cfg(windows)]

use std::path::Path;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{CloseHandle, WAIT_TIMEOUT};
use windows_sys::Win32::System::Threading::{
    OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE,
};

use faktor_winjob::JobGuard;

static PIDFILE_SEQ: AtomicU32 = AtomicU32::new(0);

/// Direct Win32 liveness probe. A dead pid either fails to open
/// (ERROR_INVALID_PARAMETER) or its process object is signaled; an open
/// process handle stays valid after termination (Windows has no zombies),
/// so the WAIT result is authoritative while the pid has not been recycled.
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
        "kp-winjob-tree-{}-{}.pid",
        std::process::id(),
        PIDFILE_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

/// powershell sleeps 60 s; the ping grandchild is created ~1.5 s in (well
/// after the caller assigns the parent), writes its pid, and sleeps ~60 s.
fn sleeper_tree_script(pid_file: &Path) -> String {
    format!(
        "Start-Sleep -Milliseconds 1500; \
         $p = Start-Process -FilePath 'ping.exe' -ArgumentList '-n','60','127.0.0.1' \
             -WindowStyle Hidden -PassThru; \
         [System.IO.File]::WriteAllText('{}', [string]$p.Id); \
         Start-Sleep -Seconds 60",
        pid_file.display()
    )
}

fn spawn_sleeper_tree(script: &str) -> Child {
    Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-Command", script])
        .spawn()
        .expect("spawn powershell sleeper tree")
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

/// TerminateJobObject must take the assigned parent AND its ping grandchild
/// (nested job member), with no taskkill-style /T walk.
#[test]
fn job_terminate_kills_the_whole_assigned_tree() {
    let job = JobGuard::create().expect("CreateJobObject must succeed on CI");
    let pid_file = pid_file_path();
    let mut child = spawn_sleeper_tree(&sleeper_tree_script(&pid_file));
    job.assign(child.id());
    let grandchild = read_grandchild_pid(&pid_file);

    let direct = child.id();
    assert!(
        pid_alive(direct),
        "powershell parent must be alive pre-kill"
    );
    assert!(
        pid_alive(grandchild),
        "ping grandchild must be alive pre-kill"
    );

    job.terminate();

    wait_until("job tree death", Duration::from_secs(10), || {
        !pid_alive(direct) && !pid_alive(grandchild)
    });
    let _ = child.wait(); // reap the exit so std drops no live handle
    let _ = std::fs::remove_file(&pid_file);
}

/// Daemon-crash semantics: dropping the LAST job handle must let the OS
/// kill the tree (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE) — nothing calls
/// TerminateProcess or taskkill in this test.
#[test]
fn closing_the_last_job_handle_kills_the_tree() {
    let pid_file = pid_file_path();
    let mut child = {
        let job = JobGuard::create().expect("CreateJobObject must succeed on CI");
        let mut child = spawn_sleeper_tree(&sleeper_tree_script(&pid_file));
        job.assign(child.id());
        let grandchild = read_grandchild_pid(&pid_file);
        assert!(
            pid_alive(grandchild),
            "ping grandchild must be alive pre-drop"
        );
        child // the job guard drops here: last handle closes -> OS kills
    };
    let direct = child.id();
    let grandchild = read_grandchild_pid(&pid_file);

    wait_until("kill-on-close tree death", Duration::from_secs(10), || {
        !pid_alive(direct) && !pid_alive(grandchild)
    });
    let _ = child.wait();
    let _ = std::fs::remove_file(&pid_file);
}

/// Adversarial: assigning a pid that already exited must be a no-op (never
/// a panic), and terminating an empty job must be harmless.
#[test]
fn assigning_dead_pids_and_terminating_empty_jobs_are_harmless() {
    let job = JobGuard::create().expect("CreateJobObject must succeed on CI");
    job.assign(0);
    job.assign(u32::MAX); // cannot exist; OpenProcess fails inside assign
    job.terminate(); // no members: no-op
}
