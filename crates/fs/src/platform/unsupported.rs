//! Non-unix platforms (Windows): no fd-relative walk.
//!
//! The P0-49 hardening walk needs per-component, no-follow opens anchored on
//! a directory handle. The unix `openat(2)` family provides this directly;
//! Windows would need a per-component `CreateFileW` walk that opens every
//! intermediate with `FILE_FLAG_OPEN_REPARSE_POINT` and manually re-anchors
//! bounded reparse points (a "delete-on-close + O_NOFOLLOW" equivalent does
//! not exist). That is deliberately NOT implemented blind: it cannot be
//! exercised on this machine, and an honest unsupported surface beats a
//! subtly wrong handle walk.
//!
//! Windows therefore keeps the canonicalize-then-open flow with the
//! post-open identity net compiled in (`WorkspaceHandle::resolve_fd` exists
//! on non-unix targets only to document the absence: it always fails
//! typed). Reparse-point-based redirection of workspace content is out of
//! scope for this wave on Windows and is documented as such.
