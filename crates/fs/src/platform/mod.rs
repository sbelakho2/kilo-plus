//! Platform split for fd-relative, symlink-bounded traversal (P0-49
//! wave-11 hardening).
//!
//! The unix implementation is the real deliverable: a component-by-component
//! `openat(2)` walk anchored on the workspace root's directory fd, which
//! never re-resolves a path string after the walk starts. Symlink components
//! are followed only by explicit, bounded `readlink` re-anchoring, so a
//! hostile concurrent process that swaps an intermediate directory for a
//! symlink can no longer redirect the resolution (the canonicalize-then-open
//! window is gone).
//!
//! Non-unix platforms (Windows) get an honest [`unsupported`] module: there
//! is no `openat(2)` equivalent in the Win32 API surface used here, and a
//! reparse-point-aware `CreateFileW` walk is future work. Those platforms
//! keep the canonicalize-then-open flow with the post-open identity net.

#[cfg(unix)]
mod unix;
#[cfg(unix)]
pub(crate) use unix::*;

#[cfg(not(unix))]
mod unsupported;
