//! Interactive terminal support for Faktor, on both unix and Windows.
//!
//! Unix: `Pty::spawn` creates a real pseudo-terminal (posix_openpt/
//! grantpt/unlockpt/ptsname), attaches the child's stdio to the slave side
//! with a controlling terminal (setsid + TIOCSCTTY), and exposes the master
//! side: write stdin, resize the window, snapshot/drain output, close.
//!
//! Windows: the same API is backed by a real ConPTY session
//! (`CreatePseudoConsole` + `PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE`); see
//! [`windows`] for the exact construction. (Audit round 53: ConPTY was the
//! last declared platform blocker for terminal parity — crates/winjob
//! already covers OS-enforced Job-Object kill-on-close for process trees,
//! which is deliberately NOT duplicated in this crate.)
//!
//! Both backends share: a bounded output ring (drop-oldest bytes; a child
//! can never deadlock on a full pipe and memory stays bounded regardless of
//! output volume), a background reader thread, and pre-spawn config
//! validation ([`validation`]) that rejects hostile config (NUL bytes,
//! oversized fields) identically on every platform before any OS call.

mod ring;

mod validation;

/// THE child-environment authority shared with the supervised spawn paths
/// ([`faktor_core::command::EnvSpec`]): [`PtyConfig::env`] always clears the
/// inherited environment and applies the resolved spec — no PTY child ever
/// inherits the daemon's full environment, and the deny-set removes
/// configured secret names on every backend.
pub use faktor_core::command::EnvSpec;

/// Pure Win32 mapping helpers (error tables, COORD geometry bounds, command
/// line quoting, env block layout). Windows-only by nature, but compiled on
/// unix in the test build so the adversarial tests in it run everywhere.
#[cfg(any(windows, test))]
mod win_common;

#[cfg(unix)]
mod unix;

#[cfg(windows)]
mod windows;

#[cfg(unix)]
pub use unix::Pty;

#[cfg(windows)]
pub use windows::Pty;

/// Spawn configuration for [`Pty`].
///
/// Bounds (enforced pre-spawn on every platform by `validation`): the
/// command must be non-empty and free of NUL bytes; individual fields and
/// the assembled command line are capped; env entries must be NUL-free.
/// `rows`/`cols` are the initial terminal size — on Windows they must fit
/// the ConPTY `COORD` range (1..=32767) and are validated before any Win32
/// call, so a config that cannot be honored errors at spawn instead of
/// silently truncating.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct PtyConfig {
    pub command: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    /// THE environment authority ([`EnvSpec`]): every backend env-clears and
    /// applies the resolved spec — the identical builder the supervisor
    /// uses. The default is the safe platform baseline (PATH/HOME/platform
    /// bits), never the daemon's full environment.
    pub env: EnvSpec,
    pub rows: u16,
    pub cols: u16,
}

impl Default for PtyConfig {
    fn default() -> Self {
        Self {
            command: String::new(),
            args: vec![],
            cwd: None,
            env: EnvSpec::default_baseline(),
            rows: 24,
            cols: 80,
        }
    }
}
