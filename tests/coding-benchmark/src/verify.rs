//! `verify.sh` execution: the repository-native verification command.
//!
//! Bounded everything: a wall-time budget (default 180 s) and an output
//! byte cap. The whole process GROUP is killed when the budget expires —
//! `verify.sh` and every child it spawned (cargo/rustc, make/cc, go, node,
//! javac, pytest) die together, so no harness run leaves orphans. The
//! captured output is never retained unboundedly: only a small diagnostic
//! tail survives; the rest is counted and discarded.

use crate::process::spawn_killable;
use std::collections::VecDeque;
use std::io::Read;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct VerifyOptions {
    pub timeout: Duration,
    /// Bytes of child output read into memory at most; beyond this the
    /// output is counted and dropped (never accumulated).
    pub output_cap: usize,
    /// Diagnostic tail retained from the (possibly truncated) output.
    pub tail_bytes: usize,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        VerifyOptions {
            timeout: Duration::from_secs(180),
            output_cap: 1024 * 1024,
            tail_bytes: 8 * 1024,
        }
    }
}

impl VerifyOptions {
    /// Environment overrides for manual/CI benchmark runs.
    pub fn with_env(mut self) -> Self {
        if let Ok(s) = std::env::var("FAKTOR_BENCH_VERIFY_TIMEOUT_S") {
            if let Ok(v) = s.parse::<u64>() {
                if v > 0 {
                    self.timeout = Duration::from_secs(v);
                }
            }
        }
        if let Ok(s) = std::env::var("FAKTOR_BENCH_VERIFY_OUTPUT_CAP") {
            if let Ok(v) = s.parse::<usize>() {
                if v >= 256 {
                    self.output_cap = v;
                }
            }
        }
        self
    }
}

/// The outcome of one `verify.sh` run. `exit == Some(0)` and
/// `!timed_out` = verified success.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerifyOutcome {
    /// Child exit code; `None` when the process died by signal (including
    /// our own kill on timeout).
    pub exit: Option<i32>,
    pub timed_out: bool,
    /// True when the child produced more than `output_cap` bytes.
    pub truncated: bool,
    pub output_cap: usize,
    pub bytes_total: u64,
    /// The last `tail_bytes` bytes of stdout+stderr (diagnostics).
    pub tail: Vec<u8>,
}

impl VerifyOutcome {
    pub fn verified(&self) -> bool {
        !self.timed_out && self.exit == Some(0)
    }
}

/// Bounded sink shared by the two reader threads (stdout + stderr):
/// counts everything, stores only the ring tail, remembers truncation.
#[derive(Debug)]
struct Sink {
    cap: usize,
    total: u64,
    ring: VecDeque<u8>,
    max_ring: usize,
}

impl Sink {
    fn new(cap: usize, max_ring: usize) -> Self {
        Sink {
            cap,
            total: 0,
            ring: VecDeque::with_capacity(max_ring.min(cap)),
            max_ring,
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.total = self.total.saturating_add(bytes.len() as u64);
        for &b in bytes {
            if self.ring.len() == self.max_ring {
                self.ring.pop_front();
            }
            self.ring.push_back(b);
        }
    }

    fn truncated(&self) -> bool {
        self.total > self.cap as u64
    }

    fn tail(&self) -> Vec<u8> {
        self.ring.iter().copied().collect()
    }
}

fn drain<R: Read>(mut reader: R, sink: Arc<Mutex<Sink>>) {
    let mut buf = [0u8; 8192];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => sink.lock().expect("sink poisoned").push(&buf[..n]),
            Err(_) => break,
        }
    }
}

/// Poll until exit or the deadline, then always kill the whole process
/// group and reap it with a bounded budget. Delegates to the shared
/// [`crate::process::ManagedChild`] so daemon and verifier harnesses share
/// one lifecycle implementation (and one Windows-correct signature).
fn wait_or_kill(child: std::process::Child, deadline: Instant) -> (Option<i32>, bool) {
    crate::process::ManagedChild::from_child(child).wait_or_kill(deadline)
}

/// Run `verify.sh` (via `/bin/sh`) with `workspace` as its working
/// directory. stdout and stderr are captured together into the bounded
/// sink. This is a synchronous, blocking helper — call it from a plain
/// thread or a test, never from an async runtime's executor thread.
pub fn run_verify_sh(workspace: &Path, opts: VerifyOptions) -> VerifyOutcome {
    run_with_shell(workspace, "/bin/sh", opts)
}

/// Same as [`run_verify_sh`] over an explicit shell binary (test seam).
pub(crate) fn run_with_shell(workspace: &Path, shell: &str, opts: VerifyOptions) -> VerifyOutcome {
    let sink = Arc::new(Mutex::new(Sink::new(opts.output_cap, opts.tail_bytes)));
    let mut command = std::process::Command::new(shell);
    command.arg("verify.sh");
    command.current_dir(workspace);
    command.stdout(std::process::Stdio::piped());
    command.stderr(std::process::Stdio::piped());
    let mut child = match spawn_killable(&mut command) {
        Ok(c) => c,
        Err(e) => {
            return VerifyOutcome {
                exit: None,
                timed_out: false,
                truncated: false,
                output_cap: opts.output_cap,
                bytes_total: 0,
                tail: format!("failed to spawn {shell} verify.sh: {e}").into_bytes(),
            };
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let mut readers = Vec::new();
    if let Some(out) = stdout {
        let sink = sink.clone();
        readers.push(std::thread::spawn(move || drain(out, sink)));
    }
    if let Some(err) = stderr {
        let sink = sink.clone();
        readers.push(std::thread::spawn(move || drain(err, sink)));
    }

    let deadline = Instant::now() + opts.timeout;
    let (exit, timed_out) = wait_or_kill(child, deadline);
    for reader in readers {
        let _ = reader.join();
    }
    let sink = sink.lock().expect("sink poisoned");
    VerifyOutcome {
        exit,
        timed_out,
        truncated: sink.truncated(),
        output_cap: opts.output_cap,
        bytes_total: sink.total,
        tail: sink.tail(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task_dir(verify_body: &str, files: &[(&str, &[u8])]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, content) in files {
            let p = dir.path().join(name);
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).unwrap();
            }
            std::fs::write(p, content).unwrap();
        }
        std::fs::write(dir.path().join("verify.sh"), verify_body).unwrap();
        dir
    }

    #[test]
    fn pass_fail_and_exit_codes_are_surfaceable() {
        let ok = task_dir("#!/bin/sh\nexit 0\n", &[]);
        let o = run_verify_sh(ok.path(), VerifyOptions::default());
        assert!(o.verified(), "{o:?}");

        let bad = task_dir("#!/bin/sh\necho boom >&2\nexit 7\n", &[]);
        let o = run_verify_sh(bad.path(), VerifyOptions::default());
        assert_eq!(o.exit, Some(7));
        assert!(!o.verified());
        assert!(String::from_utf8_lossy(&o.tail).contains("boom"));
    }

    #[test]
    fn timeout_kills_the_whole_process_group() {
        let hanging = task_dir("#!/bin/sh\nsleep 30\n", &[]);
        let o = run_verify_sh(
            hanging.path(),
            VerifyOptions {
                timeout: Duration::from_millis(300),
                ..VerifyOptions::default()
            },
        );
        assert!(o.timed_out, "{o:?}");
        assert!(!o.verified());
    }

    #[cfg(unix)]
    #[test]
    fn exited_verify_cannot_leave_grandchildren_behind() {
        // verify.sh leaves a long-lived child in ITS process group and
        // exits 0: the group kill must take the child down (zero orphans)
        // even though the verification itself succeeded.
        //
        // The child is spawned with python3 (subprocess.Popen does not
        // wait for its children at exit) because macOS /bin/sh and /bin/dash
        // both wait for background jobs at script end. The duration 31 is
        // the unique pgrep marker (other tests sleep 30).
        if !crate::toolchain::in_path("python3") {
            eprintln!("skip: python3 not on PATH (needed to detach the grandchild)");
            return;
        }
        let spawner = task_dir(
            "#!/bin/sh\npython3 -c 'import subprocess; subprocess.Popen([\"sleep\", \"31\"])'\nexit 0\n",
            &[],
        );
        let o = run_with_shell(
            spawner.path(),
            "/bin/sh",
            VerifyOptions {
                timeout: Duration::from_millis(5000),
                ..VerifyOptions::default()
            },
        );
        assert!(!o.timed_out, "{o:?}");
        assert_eq!(o.exit, Some(0));
        // Give the killed group a moment, then prove the sleep is gone.
        std::thread::sleep(Duration::from_millis(300));
        let alive = std::process::Command::new("pgrep")
            .arg("-f")
            .arg("--")
            .arg("sleep 31")
            .output()
            .map(|o| !o.stdout.is_empty())
            .unwrap_or(false);
        assert!(!alive, "the sleep grandchild must be dead after the run");
    }

    #[test]
    fn output_is_capped_and_tail_retained() {
        let noisy = task_dir(
            "#!/bin/sh\ni=0\nwhile [ $i -lt 500 ]; do echo \"line-$i-0123456789\"; i=$((i+1)); done\nexit 3\n",
            &[],
        );
        let o = run_verify_sh(
            noisy.path(),
            VerifyOptions {
                timeout: Duration::from_secs(30),
                output_cap: 512,
                tail_bytes: 256,
            },
        );
        assert_eq!(o.exit, Some(3));
        assert!(o.truncated, "{o:?}");
        assert!(
            o.bytes_total > 512,
            "bytes_total {} must exceed the cap",
            o.bytes_total
        );
        assert!(o.tail.len() <= 256);
        let tail = String::from_utf8_lossy(&o.tail);
        assert!(
            tail.contains("line-499"),
            "tail must keep the END of the output: {tail}"
        );
        assert!(
            !tail.contains("line-0-"),
            "tail must not keep the head of a huge output"
        );
    }
}
