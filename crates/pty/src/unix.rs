//! Unix interactive-terminal backend.
//!
//! `Pty::spawn` creates a real pseudo-terminal (posix_openpt/grantpt/
//! unlockpt/ptsname), attaches the child's stdio to the slave side with a
//! controlling terminal (setsid + TIOCSCTTY), and exposes the master side:
//! write stdin, resize the window, snapshot/drain output, close.
//!
//! Output capture is a bounded ring (drop-oldest bytes) drained by a
//! dedicated reader thread — the child can never deadlock on a full pipe
//! and memory stays bounded regardless of output volume. The reader thread
//! BLOCKS in read(2) on its own duplicate of the master (no polling): data
//! wakes it, and EOF (every slave fd closed — i.e. the process group died)
//! wakes it at shutdown. The SAME thread is the single process reaper
//! (waitpid), so there is exactly one owner of child reaping; `kill()`
//! signals the group and then joins the reader thread, and `Drop` is the
//! emergency failsafe (immediate SIGKILL + join, no grace sleeps).

use std::fmt;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::sync::{Arc, Condvar, Mutex};

use faktor_core::error::Error;

use crate::ring::Ring;
use crate::validation::validate_spawn_config;
use crate::PtyConfig;

/// One live PTY. Sync API (the master side is O_NONBLOCK, reads are
/// non-blocking snapshots); a background thread owns the child (reads,
/// reaps) — dropping the handle kills the whole process group.
pub struct Pty {
    master: OwnedFd,
    pid: libc::pid_t,
    shared: Arc<(Mutex<Ring>, Condvar)>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    reader: Option<std::thread::JoinHandle<()>>,
}

unsafe impl Send for Pty {}
unsafe impl Sync for Pty {}

impl fmt::Debug for Pty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pty")
            .field("pid", &self.pid)
            .finish_non_exhaustive()
    }
}

impl Pty {
    /// Create the pty and spawn the child with its stdio on the slave.
    pub fn spawn(cfg: &PtyConfig) -> Result<Self, Error> {
        use std::os::unix::process::CommandExt;

        validate_spawn_config(cfg)?;
        // 1. Open the master; grant + unlock + resolve the slave path.
        let master_fd = unsafe { libc::posix_openpt(libc::O_RDWR | libc::O_NOCTTY) };
        if master_fd < 0 {
            return Err(Error::internal("posix_openpt failed"));
        }
        let master = unsafe { OwnedFd::from_raw_fd(master_fd) };
        if unsafe { libc::grantpt(master_fd) } != 0 {
            return Err(Error::internal("grantpt failed"));
        }
        if unsafe { libc::unlockpt(master_fd) } != 0 {
            return Err(Error::internal("unlockpt failed"));
        }
        // The slave path must stay a CStr until libc::open: converting to a
        // Rust String and passing String::as_ptr() to open(2) reads past the
        // logical buffer (the String has no trailing NUL). ptsname's pointer
        // is a libc-owned static buffer valid until the next ptsname call on
        // this thread; we open before any further ptsname call.
        let slave_path = unsafe {
            let p = libc::ptsname(master_fd);
            if p.is_null() {
                return Err(Error::internal("ptsname failed"));
            }
            std::ffi::CStr::from_ptr(p)
        };
        let slave = unsafe { libc::open(slave_path.as_ptr(), libc::O_RDWR | libc::O_NOCTTY) };
        if slave < 0 {
            return Err(Error::internal("open slave failed"));
        }
        let slave_fd = unsafe { OwnedFd::from_raw_fd(slave) };

        // 2. Window size on the master.
        let mut ws = libc::winsize {
            ws_row: cfg.rows,
            ws_col: cfg.cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe {
            libc::ioctl(master_fd, libc::TIOCSWINSZ, &mut ws);
        }
        // Non-blocking master so snapshots never block.
        let flags = unsafe { libc::fcntl(master_fd, libc::F_GETFL) };
        unsafe {
            libc::fcntl(master_fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }

        // 3. Spawn: stdio = slave, setsid + TIOCSCTTY pre-exec.
        let mut cmd = std::process::Command::new(&cfg.command);
        cmd.args(&cfg.args);
        if let Some(cwd) = &cfg.cwd {
            cmd.current_dir(cwd);
        }
        // THE identical environment authority the supervised spawn path
        // uses: env_clear, then the resolved EnvSpec (deny-set filtered,
        // GIT_TERMINAL_PROMPT safety default). PTY children never inherit
        // the daemon environment implicitly.
        cfg.env.apply(&mut cmd);
        let slave_stdio = unsafe { std::process::Stdio::from_raw_fd(slave_fd.into_raw_fd()) };
        cmd.stdin(slave_stdio);
        let dup = |fd: RawFd| unsafe { libc::dup(fd) };
        let err1 = dup(slave);
        let err2 = dup(slave);
        if err1 < 0 || err2 < 0 {
            return Err(Error::internal("dup slave failed"));
        }
        cmd.stdout(unsafe { std::process::Stdio::from_raw_fd(err1) });
        cmd.stderr(unsafe { std::process::Stdio::from_raw_fd(err2) });
        unsafe {
            cmd.pre_exec(move || {
                // New session + controlling terminal on the slave.
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                let r = libc::ioctl(slave, libc::TIOCSCTTY as libc::c_ulong, 0);
                if r != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = cmd
            .spawn()
            .map_err(|e| Error::not_found(format!("spawn {}: {e}", cfg.command)))?;
        let pid = child.id() as libc::pid_t;
        // The Child handle is intentionally dropped after spawn: the reader
        // thread is the SINGLE reaper (waitpid). Keeping a second std Child
        // whose Drop/wait could race the reader's waitpid would create two
        // reapers (audit P0-56).
        drop(child);
        // NOTE: the slave fd was moved into the child's stdio above; it
        // must NOT be closed here (double close aborts under Rust's IO
        // safety checks).

        // 4. Reader + single reaper thread. The thread owns its OWN
        // duplicate of the master fd, made BLOCKING: data wakes read(2),
        // and EOF (the whole process group's slave fds closed) wakes it at
        // shutdown — no EAGAIN polling loop, no periodic sleep while idle.
        let shared = Arc::new((Mutex::new(Ring::new()), Condvar::new()));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let shared = shared.clone();
            let stop = stop.clone();
            let master_fd = unsafe { libc::dup(master.as_raw_fd()) };
            if master_fd < 0 {
                return Err(Error::internal("dup master failed"));
            }
            // The duplicate is blocking; only the thread touches it.
            let f = unsafe { libc::fcntl(master_fd, libc::F_GETFL) };
            unsafe {
                libc::fcntl(master_fd, libc::F_SETFL, f & !libc::O_NONBLOCK);
            }
            std::thread::spawn(move || {
                let mfd = master_fd;
                let mut buf = [0u8; 8192];
                loop {
                    let n = unsafe { libc::read(mfd, buf.as_mut_ptr().cast(), buf.len()) };
                    if n > 0 {
                        let (ring, cv) = &*shared;
                        ring.lock().unwrap().push(&buf[..n as usize]);
                        cv.notify_all();
                    } else if n == 0 {
                        break; // EOF: every slave fd closed
                    } else {
                        let err = std::io::Error::last_os_error();
                        match err.raw_os_error() {
                            Some(libc::EINTR) => {}
                            _ => break, // real error: nothing more to read
                        }
                    }
                }
                // Reap the child — the ONLY waitpid in this crate. Normal
                // children are zombies by now (EOF after group death or
                // natural exit) so WNOHANG succeeds immediately. A child
                // that outlives its stdio (daemonized) is polled at a low
                // cadence until the kill path sets `stop`, then one final
                // blocking wait (the group is being SIGKILLed).
                loop {
                    let mut status = 0;
                    let r = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
                    if r == pid {
                        break;
                    }
                    if r < 0 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD)
                    {
                        break;
                    }
                    if stop.load(std::sync::atomic::Ordering::SeqCst) {
                        let mut status = 0;
                        let _ = unsafe { libc::waitpid(pid, &mut status, 0) };
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                unsafe {
                    libc::close(mfd);
                }
            })
        };

        Ok(Self {
            master,
            pid,
            shared,
            stop,
            reader: Some(reader),
        })
    }

    /// The child pid (0 when unsupported).
    pub fn pid(&self) -> u32 {
        self.pid as u32
    }

    /// Write raw bytes to the pty's stdin (master side).
    pub fn write_all(&self, bytes: &[u8]) -> Result<(), Error> {
        let mut written = 0usize;
        while written < bytes.len() {
            let n = unsafe {
                libc::write(
                    self.master.as_raw_fd(),
                    bytes[written..].as_ptr().cast(),
                    bytes.len() - written,
                )
            };
            if n > 0 {
                written += n as usize;
            } else {
                let err = std::io::Error::last_os_error();
                match err.raw_os_error() {
                    Some(libc::EAGAIN) => {
                        // The tty input buffer is full: wait for POLLOUT
                        // with a bounded timeout instead of a 2 ms busy
                        // sleep. A slow consumer costs one wakeup per poll
                        // interval at most, never a hot loop.
                        let mut pfd = libc::pollfd {
                            fd: self.master.as_raw_fd(),
                            events: libc::POLLOUT,
                            revents: 0,
                        };
                        let r = unsafe { libc::poll(&mut pfd, 1, 100) };
                        if r < 0 {
                            let e2 = std::io::Error::last_os_error();
                            if e2.raw_os_error() == Some(libc::EINTR) {
                                continue;
                            }
                            return Err(Error::internal(format!("pty poll: {e2}")));
                        }
                        if r == 0 {
                            return Err(Error::internal("pty write stalled (POLLOUT timeout)"));
                        }
                    }
                    Some(libc::EINTR) => {}
                    _ => return Err(Error::internal(format!("pty write: {err}"))),
                }
            }
        }
        Ok(())
    }

    /// Write a line (the pty's line discipline handles CR/echo).
    pub fn write_line(&self, line: &str) -> Result<(), Error> {
        let mut b = line.as_bytes().to_vec();
        b.push(b'\n');
        self.write_all(&b)
    }

    /// Resize the terminal window.
    pub fn resize(&self, rows: u16, cols: u16) -> Result<(), Error> {
        if rows == 0 || cols == 0 {
            return Err(Error::malformed("pty size must be non-zero"));
        }
        let mut ws = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        let r = unsafe { libc::ioctl(self.master.as_raw_fd(), libc::TIOCSWINSZ, &mut ws) };
        if r != 0 {
            return Err(Error::internal("TIOCSWINSZ failed"));
        }
        Ok(())
    }

    /// Current window size from the kernel.
    pub fn size(&self) -> (u16, u16) {
        let mut ws = libc::winsize {
            ws_row: 0,
            ws_col: 0,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe {
            libc::ioctl(self.master.as_raw_fd(), libc::TIOCGWINSZ, &mut ws);
        }
        (ws.ws_row, ws.ws_col)
    }

    /// Drain all currently available output.
    pub fn read_available(&self) -> Vec<u8> {
        self.shared.0.lock().unwrap().drain()
    }

    /// Snapshot the current output WITHOUT draining.
    pub fn snapshot(&self) -> Vec<u8> {
        self.shared.0.lock().unwrap().snapshot()
    }

    /// Total bytes ever read from the master.
    pub fn total_bytes(&self) -> u64 {
        self.shared.0.lock().unwrap().total()
    }

    /// Block until `needle` appears in the accumulated output or `timeout`
    /// elapses (test/consumer helper).
    pub fn wait_for_contains(&self, needle: &str, timeout: std::time::Duration) -> bool {
        let (ring, cv) = &*self.shared;
        let mut guard = ring.lock().unwrap();
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let snap = guard.snapshot();
            let text = String::from_utf8_lossy(&snap);
            if text.contains(needle) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            let (g, _t) = cv
                .wait_timeout(guard, std::time::Duration::from_millis(50))
                .unwrap();
            guard = g;
        }
    }

    /// Is the child still running? False once the reader thread has reaped
    /// it (a zombie reports false, matching the single-reaper design).
    pub fn is_alive(&self) -> bool {
        if self.pid <= 0 {
            return false;
        }
        let r = unsafe { libc::kill(self.pid, 0) };
        r == 0
    }

    /// Graceful shutdown: SIGTERM the process group, a short grace period,
    /// SIGKILL, then join the reader/reaper thread (bounded). This is the
    /// normal lifecycle for live objects; [`Drop`] is the emergency path.
    pub fn shutdown(&mut self) {
        if self.pid > 0 {
            unsafe {
                libc::kill(-self.pid, libc::SIGTERM);
            }
            // Give the group a short grace, watching for exit.
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(150);
            loop {
                if !self.is_alive() {
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            if self.is_alive() {
                unsafe {
                    libc::kill(-self.pid, libc::SIGKILL);
                }
            }
        }
        self.join_reader();
    }

    /// Kill the process group (SIGTERM, short grace, SIGKILL) and join the
    /// reader thread. Idempotent; kept for compatibility with `kill()`.
    pub fn kill(&mut self) {
        self.shutdown();
    }

    fn join_reader(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(handle) = self.reader.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Pty {
    /// Emergency failsafe ONLY: immediate SIGKILL of the group (no grace
    /// sleep on the caller's thread) then join the reader thread. Live
    /// objects should use [`Pty::shutdown`] for the graceful SIGTERM path.
    fn drop(&mut self) {
        if self.pid > 0 {
            unsafe {
                libc::kill(-self.pid, libc::SIGKILL);
            }
        }
        self.join_reader();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::RING_MAX_BYTES;
    use crate::EnvSpec;
    use faktor_core::error::ErrorKind;

    fn sh_cfg(script: &str) -> PtyConfig {
        PtyConfig {
            command: "sh".into(),
            args: vec!["-c".into(), script.into()],
            rows: 24,
            cols: 80,
            ..Default::default()
        }
    }

    fn group_alive(pid: libc::pid_t) -> bool {
        (unsafe { libc::kill(pid, 0) }) == 0
    }

    #[test]
    fn interactive_round_trip_through_a_real_tty() {
        // read a line, echo it back; echo disabled so we assert OUR bytes.
        let cfg = sh_cfg("stty -echo; read x; echo out:$x; exit 0");
        let mut pty = Pty::spawn(&cfg).unwrap();
        pty.write_line("hello pty").unwrap();
        assert!(
            pty.wait_for_contains("out:hello pty", std::time::Duration::from_secs(10)),
            "the child must read our line through the pty: {:?}",
            String::from_utf8_lossy(&pty.snapshot())
        );
        pty.kill();
    }

    #[test]
    fn resize_reaches_the_kernel_and_the_shell() {
        // `stty size` prints the live rows/cols AFTER we resize: the child
        // waits on a line of input first, so the test is not a startup race
        // (Linux CI exposed the child winning it).
        let cfg = sh_cfg("read x; stty size");
        let mut pty = Pty::spawn(&cfg).unwrap();
        pty.resize(33, 121).unwrap();
        pty.write_line("go").unwrap();
        assert_eq!(pty.size(), (33, 121));
        assert!(
            pty.wait_for_contains("33 121", std::time::Duration::from_secs(10)),
            "TIOCSWINSZ must reach the child: {:?}",
            String::from_utf8_lossy(&pty.snapshot())
        );
        pty.kill();
    }

    #[test]
    fn huge_output_stays_bounded_and_never_deadlocks() {
        // seq 1..200000 through a pty: the reader drains continuously (the
        // child never blocks) and RAM stays bounded by the ring.
        let cfg = sh_cfg("seq 1 200000; exit 0");
        let mut pty = Pty::spawn(&cfg).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while pty.is_alive() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let _ = pty.read_available();
        }
        assert!(
            std::time::Instant::now() < deadline,
            "pty output must not deadlock the child"
        );
        assert!(pty.total_bytes() >= 200_000, "all output was drained");
        assert!(pty.snapshot().len() <= RING_MAX_BYTES, "ring stays bounded");
        pty.kill();
    }

    #[test]
    fn drop_is_emergency_sigkill_and_fast() {
        let cfg = sh_cfg("sleep 300");
        let (pid, elapsed) = {
            let pty = Pty::spawn(&cfg).unwrap();
            assert!(pty.is_alive());
            let start = std::time::Instant::now();
            drop(pty);
            (0, start.elapsed())
        };
        let _ = pid;
        assert!(
            elapsed < std::time::Duration::from_millis(200),
            "Drop must not block the caller with grace sleeps: {elapsed:?}"
        );
    }

    #[test]
    fn drop_kills_the_process_group() {
        let cfg = sh_cfg("sleep 300");
        let pid = {
            let pty = Pty::spawn(&cfg).unwrap();
            assert!(pty.is_alive());
            pty.pid()
        };
        // Dropped: the child group must be dead shortly after.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if !group_alive(pid as libc::pid_t) {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "dropped pty must kill its child group"
            );
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    #[test]
    fn naturally_exited_child_is_reaped_by_the_reader_thread() {
        // The reader thread is the single reaper: after a natural exit the
        // child must be reaped (no zombie) — is_alive() turns false.
        let cfg = sh_cfg("echo done; exit 0");
        let mut pty = Pty::spawn(&cfg).unwrap();
        assert!(
            pty.wait_for_contains("done", std::time::Duration::from_secs(10)),
            "child output arrives"
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while pty.is_alive() {
            assert!(
                std::time::Instant::now() < deadline,
                "the reader thread must reap the exited child"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        pty.kill();
    }

    #[test]
    fn sigterm_resisting_child_is_escalated_and_reaped() {
        // The child traps SIGTERM: shutdown() must escalate to SIGKILL and
        // the single reaper must reap it (no zombie, join succeeds).
        let cfg = sh_cfg("trap '' TERM; echo armed; sleep 30");
        let mut pty = Pty::spawn(&cfg).unwrap();
        assert!(
            pty.wait_for_contains("armed", std::time::Duration::from_secs(10)),
            "child armed"
        );
        let pid = pty.pid();
        let start = std::time::Instant::now();
        pty.shutdown();
        assert!(
            start.elapsed() < std::time::Duration::from_secs(5),
            "shutdown must escalate within the bound"
        );
        assert!(
            !group_alive(pid as libc::pid_t),
            "SIGKILL must have taken the group"
        );
    }

    #[test]
    fn idle_pty_reader_does_not_busy_poll_and_wakes_on_output() {
        // The reader blocks in read(2): an idle pty performs no periodic
        // wakeups. The child sleeps 1s then writes; the output must arrive
        // promptly after the write (a 2 ms-polling reader would also pass,
        // so the structural guarantee is asserted via shutdown promptness:
        // a blocking reader cannot be interrupted by polling, yet kill()
        // returns quickly because EOF wakes it).
        let cfg = sh_cfg("sleep 1; echo late; sleep 30");
        let mut pty = Pty::spawn(&cfg).unwrap();
        let start = std::time::Instant::now();
        // Idle for 600 ms (no reads from our side): nothing should stall.
        std::thread::sleep(std::time::Duration::from_millis(600));
        assert!(start.elapsed() < std::time::Duration::from_secs(2));
        assert!(
            pty.wait_for_contains("late", std::time::Duration::from_secs(10)),
            "blocking reader wakes on data"
        );
        let t0 = std::time::Instant::now();
        pty.shutdown();
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(2),
            "kill path must wake the blocking reader via group EOF"
        );
    }

    #[test]
    fn spawn_errors_are_loud() {
        let mut cfg = sh_cfg("true");
        cfg.command = "/nonexistent-binary".into();
        assert!(Pty::spawn(&cfg).is_err());
        let err = Pty::spawn(&PtyConfig::default()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
    }

    #[test]
    fn pty_env_uses_the_identical_authority_and_no_daemon_var_leaks() {
        // The exact terminal-side assertion, repeated through a REAL PTY:
        // PATH + the approved toolchain vars arrive; configured secret
        // names set in the parent never cross (even allowlisted explicitly);
        // an undeclared daemon var never arrives.
        std::env::set_var("CARGO_HOME", "/tmp/kp-pty-cargo-home");
        std::env::set_var("RUSTUP_HOME", "/tmp/kp-pty-rustup-home");
        std::env::set_var("FAKTOR_SERVER_PASSWORD", "hunter2");
        std::env::set_var("OPENAI_API_KEY", "sk-pty-secret");
        std::env::set_var("TEST_PRIVATE_SECRET", "private");
        std::env::set_var("KP_PTY_UNDECLARED", "must-not-arrive");
        // The child PRINTS its env through the PTY: the assertions run on
        // the bytes the terminal actually delivered.
        let mut cfg = sh_cfg("env");
        cfg.env = EnvSpec::toolchain();
        let mut pty = Pty::spawn(&cfg).unwrap();
        assert!(
            pty.wait_for_contains(
                "CARGO_HOME=/tmp/kp-pty-cargo-home",
                std::time::Duration::from_secs(10)
            ),
            "PATH/toolchain vars must arrive: {:?}",
            String::from_utf8_lossy(&pty.snapshot())
        );
        let printed = String::from_utf8_lossy(&pty.snapshot()).into_owned();
        assert!(
            printed
                .lines()
                .any(|l| l.trim_end().starts_with("PATH=") && l.len() > "PATH=".len()),
            "PATH must be present and non-empty: {printed:?}"
        );
        assert!(printed.contains("RUSTUP_HOME=/tmp/kp-pty-rustup-home"));
        for secret in [
            "FAKTOR_SERVER_PASSWORD",
            "OPENAI_API_KEY",
            "TEST_PRIVATE_SECRET",
            "KP_PTY_UNDECLARED",
        ] {
            assert!(!printed.contains(secret), "{secret} leaked through the pty");
        }
        pty.kill();
        // Explicit entries cannot smuggle denied names either.
        let mut cfg = sh_cfg(
            "test -z \"$OPENAI_API_KEY\" && test -z \"$TEST_PRIVATE_SECRET\" \
             && echo pty-explicit-exact",
        );
        cfg.env = EnvSpec::Explicit(vec![
            ("OPENAI_API_KEY".into(), "leak".into()),
            ("TEST_PRIVATE_SECRET".into(), "leak".into()),
        ]);
        let mut pty = Pty::spawn(&cfg).unwrap();
        assert!(
            pty.wait_for_contains("pty-explicit-exact", std::time::Duration::from_secs(10)),
            "{:?}",
            String::from_utf8_lossy(&pty.snapshot())
        );
        pty.kill();
        std::env::remove_var("CARGO_HOME");
        std::env::remove_var("RUSTUP_HOME");
        std::env::remove_var("FAKTOR_SERVER_PASSWORD");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("TEST_PRIVATE_SECRET");
        std::env::remove_var("KP_PTY_UNDECLARED");
    }

    #[test]
    fn hostile_environment_variables_do_not_break_spawn() {
        // Hostile env specs never panic the spawn path: NUL-bearing explicit
        // entries are rejected pre-spawn as Malformed; a huge allowlist name
        // is dropped by resolve (no daemon value exists).
        let mut cfg = sh_cfg("echo ok");
        cfg.env = EnvSpec::Explicit(vec![("K\0EY".into(), "v".into())]);
        let err = Pty::spawn(&cfg).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
        let mut cfg = sh_cfg("echo ok");
        cfg.env = EnvSpec::Allowlisted(vec!["K\0EY".into()]);
        let err = Pty::spawn(&cfg).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
        let mut cfg = sh_cfg("echo ok");
        cfg.env = EnvSpec::Allowlisted(vec!["NOPE_NOT_SET".into()]);
        let mut pty = Pty::spawn(&cfg).unwrap();
        assert!(
            pty.wait_for_contains("ok", std::time::Duration::from_secs(10)),
            "child runs with a custom env"
        );
        pty.kill();
    }

    #[test]
    fn resize_with_zero_dimensions_is_rejected_before_any_ioctl() {
        let cfg = sh_cfg("true");
        let mut pty = Pty::spawn(&cfg).unwrap();
        assert_eq!(
            Pty::resize(&pty, 0, 80).unwrap_err().kind,
            ErrorKind::Malformed
        );
        assert_eq!(
            Pty::resize(&pty, 24, 0).unwrap_err().kind,
            ErrorKind::Malformed
        );
        assert_eq!(pty.size(), (24, 80));
        pty.kill();
    }

    #[test]
    fn nul_bytes_in_args_fail_validation_before_spawn() {
        // Previously this surfaced as a late io::Error mapped to NotFound;
        // pre-spawn validation must reject it as Malformed.
        let mut cfg = sh_cfg("true");
        cfg.args = vec!["-c".into(), "echo\0owned".into()];
        let err = Pty::spawn(&cfg).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
    }
}
