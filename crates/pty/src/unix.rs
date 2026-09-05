//! Unix interactive-terminal backend.
//!
//! `Pty::spawn` creates a real pseudo-terminal (posix_openpt/grantpt/
//! unlockpt/ptsname), attaches the child's stdio to the slave side with a
//! controlling terminal (setsid + TIOCSCTTY), and exposes the master side:
//! write stdin, resize the window, snapshot/drain output, close.
//!
//! Output capture is a bounded ring (drop-oldest bytes) drained by a
//! dedicated reader thread — the child can never deadlock on a full pipe
//! and memory stays bounded regardless of output volume. The child is
//! reaped by the same thread; `Drop` terminates the whole process group
//! (SIGTERM → SIGKILL) so a dropped pty can never leak children.

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
    child: Option<std::process::Child>,
    shared: Arc<(Mutex<Ring>, Condvar)>,
    reader_stop: Arc<std::sync::atomic::AtomicBool>,
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
        let slave_path = unsafe {
            let p = libc::ptsname(master_fd);
            if p.is_null() {
                return Err(Error::internal("ptsname failed"));
            }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        };

        // 2. Slave fd for the child's stdio + controlling terminal.
        let slave =
            unsafe { libc::open(slave_path.as_ptr().cast(), libc::O_RDWR | libc::O_NOCTTY) };
        if slave < 0 {
            return Err(Error::internal("open slave failed"));
        }
        let slave_fd = unsafe { OwnedFd::from_raw_fd(slave) };

        // 3. Window size on the master.
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

        // 4. Spawn: stdio = slave, setsid + TIOCSCTTY pre-exec.
        let mut cmd = std::process::Command::new(&cfg.command);
        cmd.args(&cfg.args);
        if let Some(cwd) = &cfg.cwd {
            cmd.current_dir(cwd);
        }
        for (k, v) in &cfg.env {
            cmd.env(k, v);
        }
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
        // NOTE: the slave fd was moved into the child's stdio above; it
        // must NOT be closed here (double close aborts under Rust's IO
        // safety checks).

        // 5. Reader + reaper thread.
        let shared = Arc::new((Mutex::new(Ring::new()), Condvar::new()));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let shared = shared.clone();
            let stop = stop.clone();
            // The reader thread owns its OWN duplicate of the master fd so
            // it can close it at EOF without racing the Pty handle's drop
            // (the handle's OwnedFd is the only closer of the original).
            let master_fd = unsafe { libc::dup(master.as_raw_fd()) };
            if master_fd < 0 {
                return Err(Error::internal("dup master failed"));
            }
            std::thread::spawn(move || {
                let mut buf = [0u8; 8192];
                let mfd = master_fd;
                loop {
                    if stop.load(std::sync::atomic::Ordering::SeqCst) {
                        break;
                    }
                    let n = unsafe { libc::read(mfd, buf.as_mut_ptr().cast(), buf.len()) };
                    if n > 0 {
                        let (ring, cv) = &*shared;
                        ring.lock().unwrap().push(&buf[..n as usize]);
                        cv.notify_all();
                    } else if n == 0 {
                        break; // EOF (slave closed)
                    } else {
                        let err = std::io::Error::last_os_error();
                        match err.raw_os_error() {
                            Some(libc::EAGAIN) => {
                                std::thread::sleep(std::time::Duration::from_millis(2));
                            }
                            Some(libc::EINTR) => {}
                            _ => break,
                        }
                    }
                }
                // Reap the child (this thread owns the wait).
                let mut status = 0;
                loop {
                    let r = unsafe { libc::waitpid(pid, &mut status, 0) };
                    if r == pid
                        || (r < 0
                            && std::io::Error::last_os_error().raw_os_error() == Some(libc::ECHILD))
                    {
                        break;
                    }
                    if r < 0 {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                }
                unsafe {
                    libc::close(mfd);
                }
            });
        }

        Ok(Self {
            master,
            pid,
            child: Some(child),
            shared,
            reader_stop: stop,
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
                    Some(libc::EAGAIN) | Some(libc::EINTR) => {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
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

    /// Is the child still running?
    pub fn is_alive(&self) -> bool {
        if self.pid <= 0 {
            return false;
        }
        let r = unsafe { libc::kill(self.pid, 0) };
        r == 0
    }

    /// Terminate the child process group (SIGTERM, then SIGKILL after a
    /// short grace) and reap. Idempotent.
    pub fn kill(&mut self) {
        if self.pid > 0 {
            unsafe {
                libc::kill(-self.pid, libc::SIGTERM);
            }
            // Give the child a moment to exit, then SIGKILL the group.
            std::thread::sleep(std::time::Duration::from_millis(200));
            unsafe {
                libc::kill(-self.pid, libc::SIGKILL);
            }
        }
        self.reap();
    }

    fn reap(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.wait();
        }
    }
}

impl Drop for Pty {
    fn drop(&mut self) {
        self.reader_stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if self.pid > 0 {
            unsafe {
                libc::kill(-self.pid, libc::SIGTERM);
            }
            std::thread::sleep(std::time::Duration::from_millis(150));
            unsafe {
                libc::kill(-self.pid, libc::SIGKILL);
            }
        }
        self.reap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::RING_MAX_BYTES;
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
        // `stty size` prints the live rows/cols after resize.
        let cfg = sh_cfg("stty size");
        let mut pty = Pty::spawn(&cfg).unwrap();
        pty.resize(33, 121).unwrap();
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
            let alive = unsafe { libc::kill(pid as libc::pid_t, 0) } == 0;
            if !alive {
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
    fn spawn_errors_are_loud() {
        let mut cfg = sh_cfg("true");
        cfg.command = "/nonexistent-binary".into();
        assert!(Pty::spawn(&cfg).is_err());
        let err = Pty::spawn(&PtyConfig::default()).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
    }

    #[test]
    fn hostile_environment_variables_do_not_break_spawn() {
        // Env entries with NULs etc. must not panic the spawn path.
        let mut cfg = sh_cfg("echo ok");
        cfg.env
            .push(("PATH".into(), std::env::var("PATH").unwrap_or_default()));
        let mut pty = Pty::spawn(&cfg).unwrap();
        assert!(
            pty.wait_for_contains("ok", std::time::Duration::from_secs(10)),
            "child runs with custom env"
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
