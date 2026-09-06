//! fd-relative anchored traversal on unix (P0-49, wave-11 final hardening).
//!
//! [`open_no_follow_walk`] resolves `rel` under the already-canonicalized
//! workspace `root` WITHOUT ever re-resolving a path string: every step is an
//! `openat(2)` on the directory fd produced by the previous step, so a
//! hostile process that swaps a directory entry between two of our steps can
//! only redirect steps that have NOT happened yet — and every such redirect
//! attempt is itself checked at the moment it happens.
//!
//! Walk rules
//! ----------
//! * The root is anchored by `open(root, O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW)`:
//!   the workspace root is durable configuration, but if the root entry
//!   itself is ever swapped for a symlink the walk fails loudly instead of
//!   following it.
//! * Each remaining component is opened relative to the CURRENT directory fd
//!   with `O_NOFOLLOW`: a plain entry is descended (intermediates with
//!   `O_DIRECTORY`, the final component with the caller's `final_flags`);
//!   a symlink entry yields `ELOOP`, never a kernel follow.
//! * On `ELOOP` the component is followed MANUALLY and with bounds: the link
//!   target is read with `readlinkat`, then the walk is rebased onto the
//!   target. Absolute targets must stay under `root` (component-wise prefix
//!   check against the canonical root) or they are denied; relative targets
//!   are normalized against the current position and denied when they would
//!   climb above `root`. Each rebase re-walks from the trusted root fd, so
//!   every path component is re-checked at the moment it is really opened.
//! * At most [`MAX_SYMLINK_HOPS`] symlink hops are charged per walk; a loop
//!   or an attacker that keeps swapping links in front of the walk fails
//!   with a typed [`ErrorKind::Permission`] error — never a hang.
//! * `..`, absolute paths outside the root, and NUL-containing components
//!   are denied before any `openat` runs.
//!
//! Honest limits
//! -------------
//! A directory entry that is swapped for a REAL (non-symlink) directory by a
//! concurrent rename is indistinguishable from legitimate content and is
//! walked; the file read is then whatever is genuinely inside the workspace
//! tree at that instant. Directories are pinned the moment their fd is
//! opened: a rename-away after the fd exists cannot redirect the walk (it
//! continues inside the pinned directory), which the post-open identity net
//! in `lib.rs` then cross-checks against the path name.

use std::collections::VecDeque;
use std::ffi::{CString, OsStr, OsString};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Component, Path};
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

use faktor_core::error::Error;

/// Hard bound on symlink hops per walk: at most 8 links may be followed
/// while resolving ONE relative path. A symlink loop therefore fails after
/// 9 `ELOOP`s with a typed error instead of hanging (Linux charges the same
/// idea per resolution; we charge per walk for determinism).
pub const MAX_SYMLINK_HOPS: usize = 8;

/// Hard bound on the number of components one walk may consume (hostile
/// `rel` of unbounded length must fail loudly, never exhaust the syscall
/// budget).
const MAX_COMPONENTS: usize = 4096;

const O_DIR: i32 = libc::O_RDONLY | libc::O_DIRECTORY;

/// Open the entry named by `rel` under the canonical directory `root`.
///
/// `final_flags` are OR-ed with `O_NOFOLLOW | O_CLOEXEC` for the LAST
/// component only; intermediate components always get
/// `O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC`. Returns the fd of the final
/// entry, or of the root itself when `rel` is empty/`.`.
///
/// Error kinds: lexical traversal and symlink escapes are
/// [`ErrorKind::Permission`]; an absent final component is
/// [`ErrorKind::NotFound`]; an absent/unusable intermediate component is a
/// `Permission` (mirrors the historical `resolve_within` parent-resolution
/// failure); a path beyond [`MAX_COMPONENTS`] is [`ErrorKind::Oversized`].
pub(crate) fn open_no_follow_walk(
    root: &Path,
    rel: &Path,
    final_flags: i32,
) -> Result<OwnedFd, Error> {
    let mut pending = lexical_components(root, rel)?;
    if pending.len() > MAX_COMPONENTS {
        return Err(Error::oversized(format!(
            "{rel:?} exceeds the {MAX_COMPONENTS}-component walk bound"
        )));
    }
    let mut dir = open_root(root)?;
    // Root-relative position of `dir` as pure Normal components. Used ONLY
    // for arithmetic on relative symlink targets (never for a syscall): a
    // relative target containing `..` is normalized against it and the walk
    // is rebased onto the trusted root fd afterwards, so a stale position
    // string can only cause an over-strict denial, never an escape.
    let mut pos: Vec<OsString> = Vec::new();
    let mut hops = 0usize;
    loop {
        let Some(comp) = pending.pop_front() else {
            // Exhausted: the resolved entry IS the current directory (root,
            // or a final symlink whose empty target names its own parent).
            return Ok(dir);
        };
        walk_seam(&comp);
        let last = pending.is_empty();
        let flags = if last {
            final_flags | libc::O_NOFOLLOW | libc::O_CLOEXEC
        } else {
            O_DIR | libc::O_NOFOLLOW | libc::O_CLOEXEC
        };
        match openat_comp(&dir, &comp, flags) {
            Ok(fd) => {
                if last {
                    return Ok(fd);
                }
                pos.push(comp);
                dir = fd;
            }
            Err(e) => {
                // O_NOFOLLOW on a symlink yields ELOOP on Linux; macOS
                // reports ENOTDIR instead whenever O_DIRECTORY is in the
                // flags. Distinguish the two real cases with one lstat:
                // a genuine symlink goes through the bounded follow; a real
                // non-directory stays an error.
                let is_symlink = e.raw_os_error() == Some(libc::ELOOP)
                    || (e.raw_os_error() == Some(libc::ENOTDIR) && entry_is_symlink(&dir, &comp)?);
                if !is_symlink {
                    return Err(map_open_error(rel, last, e));
                }
                hops += 1;
                if hops > MAX_SYMLINK_HOPS {
                    return Err(Error::permission(format!(
                        "{rel:?}: symlink resolution exceeded the {MAX_SYMLINK_HOPS}-hop bound (possible symlink loop)"
                    )));
                }
                let target = readlinkat_comp(&dir, &comp).map_err(|e| map_io(rel, e))?;
                match rebase_target(root, rel, &pos, &target)? {
                    Rebase::Stay(tc) => {
                        for c in tc.into_iter().rev() {
                            pending.push_front(c);
                        }
                    }
                    Rebase::Relocate(full) => {
                        for c in full.into_iter().rev() {
                            pending.push_front(c);
                        }
                        pos.clear();
                        dir = open_root(root)?;
                    }
                }
            }
        }
    }
}

/// Where a symlink hop sends the walk.
enum Rebase {
    /// Relative target without `..`: kernel semantics say it resolves
    /// against the directory containing the link, so the walk simply
    /// continues from the SAME fd with the target's components in front.
    Stay(VecDeque<OsString>),
    /// Absolute target (already verified under `root`) or a relative target
    /// containing `..`: the walk restarts from the trusted root fd with the
    /// fully normalized target components in front (every component is then
    /// re-checked at the moment it is really opened).
    Relocate(VecDeque<OsString>),
}

/// Lexically split + normalize a symlink target against the walk position.
fn rebase_target(
    root: &Path,
    rel: &Path,
    pos: &[OsString],
    target: &OsStr,
) -> Result<Rebase, Error> {
    let t = Path::new(target);
    let escape =
        |what: &str| Error::permission(format!("{rel:?}: symlink target {target:?} {what}"));
    if t.is_absolute() {
        // Absolute target: must name a location under the canonical root.
        let suffix = t
            .strip_prefix(root)
            .map_err(|_| escape("leaves the workspace root"))?;
        let mut acc: Vec<OsString> = Vec::new();
        for c in suffix.components() {
            match c {
                Component::Normal(n) => acc.push(n.to_os_string()),
                Component::CurDir => {}
                Component::ParentDir => {
                    if acc.pop().is_none() {
                        return Err(escape("climbs above the workspace root"));
                    }
                }
                _ => return Err(escape("is malformed")),
            }
        }
        return Ok(Rebase::Relocate(acc.into()));
    }
    let mut acc: Vec<OsString> = pos.to_vec();
    let mut pure: Vec<OsString> = Vec::new();
    let mut saw_parent = false;
    for c in t.components() {
        match c {
            Component::Normal(n) => {
                let n = n.to_os_string();
                pure.push(n.clone());
                acc.push(n);
            }
            Component::CurDir => {}
            Component::ParentDir => {
                saw_parent = true;
                if acc.pop().is_none() {
                    return Err(escape("climbs above the workspace root"));
                }
            }
            _ => return Err(escape("is malformed")),
        }
    }
    if saw_parent {
        Ok(Rebase::Relocate(acc.into()))
    } else {
        Ok(Rebase::Stay(pure.into()))
    }
}

/// Validate the caller's path lexically and split it into Normal components.
/// `..`, absolute escapes and NUL components are denied BEFORE any open.
fn lexical_components(root: &Path, rel: &Path) -> Result<VecDeque<OsString>, Error> {
    let mut out = VecDeque::new();
    let mut path = rel;
    if rel.is_absolute() {
        let rest = rel
            .strip_prefix(root)
            .map_err(|_| Error::permission(format!("path escapes workspace: {rel:?}")))?;
        if rest.as_os_str().is_empty() {
            return Ok(out);
        }
        path = rest;
    }
    for c in path.components() {
        match c {
            Component::Normal(n) => out.push_back(n.to_os_string()),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(Error::permission(format!(
                    "path traversal rejected: {rel:?}"
                )));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(Error::permission(format!(
                    "path traversal rejected: {rel:?}"
                )));
            }
        }
    }
    Ok(out)
}

fn open_root(root: &Path) -> Result<OwnedFd, Error> {
    let c = CString::new(root.as_os_str().as_bytes())
        .map_err(|_| Error::malformed(format!("workspace root {:?} contains a NUL byte", root)))?;
    let flags = O_DIR | libc::O_NOFOLLOW | libc::O_CLOEXEC;
    let fd = unsafe { libc::open(c.as_ptr(), flags) };
    if fd < 0 {
        let e = io::Error::last_os_error();
        // macOS reports ENOTDIR (not ELOOP) when O_NOFOLLOW|O_DIRECTORY hits
        // a symlink: either way the durable root entry was swapped.
        if matches!(e.raw_os_error(), Some(libc::ELOOP) | Some(libc::ENOTDIR)) {
            return Err(Error::permission(format!(
                "workspace root {} was swapped for a symlink or non-directory",
                root.display()
            )));
        }
        return Err(if e.kind() == io::ErrorKind::NotFound {
            Error::not_found(format!("workspace root {}: {e}", root.display()))
        } else {
            Error::internal(format!("workspace root {}: {e}", root.display()))
        });
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn openat_comp(dir: &OwnedFd, name: &OsStr, flags: i32) -> Result<OwnedFd, io::Error> {
    let c = CString::new(name.as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("path component {name:?} contains a NUL byte"),
        )
    })?;
    // SAFETY: `c` is a NUL-terminated name relative to the open directory
    // fd `dir`; flags do not include O_CREAT, so no mode argument is needed.
    let fd = unsafe { libc::openat(dir.as_raw_fd(), c.as_ptr(), flags) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Is the entry named by `comp` in `dir` a symlink? (lstat, no follow.)
/// macOS reports ENOTDIR — not ELOOP — when `openat(O_DIRECTORY|O_NOFOLLOW)`
/// hits a symlink, so the walk needs one lstat to classify the error.
/// Vanished entries are treated as "not a symlink" (the original error then
/// maps to its normal meaning).
fn entry_is_symlink(dir: &OwnedFd, name: &OsStr) -> Result<bool, io::Error> {
    let c = match CString::new(name.as_bytes()) {
        Ok(c) => c,
        Err(_) => return Ok(false),
    };
    // SAFETY: fstatat does not modify the NUL-terminated `c`; `st` is a
    // valid output buffer for the stat result.
    let mut st: libc::stat = unsafe { std::mem::zeroed() };
    let r = unsafe {
        libc::fstatat(
            dir.as_raw_fd(),
            c.as_ptr(),
            &mut st,
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    if r < 0 {
        return Ok(false);
    }
    Ok((st.st_mode & libc::S_IFMT) == libc::S_IFLNK)
}

/// Read a symlink target relative to `dir`. Symlink content is bounded by
/// the filesystem (link(2) refuses targets beyond PATH_MAX); the buffer
/// still grows defensively in the impossible case of a full read.
fn readlinkat_comp(dir: &OwnedFd, name: &OsStr) -> Result<OsString, io::Error> {
    let c = CString::new(name.as_bytes()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("path component {name:?} contains a NUL byte"),
        )
    })?;
    let mut buf = vec![0u8; 4096];
    loop {
        // SAFETY: readlinkat does not modify the NUL-terminated `c`, and
        // `buf` is a writable byte buffer of the reported size.
        let n = unsafe {
            libc::readlinkat(
                dir.as_raw_fd(),
                c.as_ptr(),
                buf.as_mut_ptr().cast::<libc::c_char>(),
                buf.len(),
            )
        };
        if n < 0 {
            return Err(io::Error::last_os_error());
        }
        let n = n as usize;
        if n < buf.len() {
            buf.truncate(n);
            return Ok(OsString::from_vec(buf));
        }
        if buf.len() >= 1 << 20 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "symlink target exceeds 1 MiB",
            ));
        }
        buf.resize(buf.len() * 2, 0);
    }
}

/// Error mapping for a component `openat` failure (P0-49 parity with the
/// historical string-based `resolve_within`):
/// * absent FINAL component -> NotFound (readers surface not-found);
/// * absent/blocked/not-a-directory INTERMEDIATE component -> Permission
///   ("parent resolution failed", as before);
/// * anything else on the final component -> Internal, matching the
///   historical `err_not_found` classification.
fn map_open_error(rel: &Path, last: bool, e: io::Error) -> Error {
    match e.raw_os_error() {
        Some(libc::ENOENT) if last => Error::not_found(format!("{}", rel.display())),
        Some(libc::ENOENT) => Error::permission(format!("parent resolution failed: {rel:?}")),
        Some(libc::ENOTDIR) => Error::permission(format!("parent resolution failed: {rel:?}")),
        Some(libc::EACCES) if !last => {
            Error::permission(format!("parent resolution failed: {rel:?}"))
        }
        _ if last => Error::internal(format!("{}: {e}", rel.display())),
        _ => Error::permission(format!("parent resolution failed: {rel:?}")),
    }
}

fn map_io(rel: &Path, e: io::Error) -> Error {
    Error::internal(format!("{}: {e}", rel.display()))
}

/// Test seam: fired with the next component name BEFORE it is opened.
/// Deterministic TOCTOU tests swap a directory entry to a symlink inside
/// this window (between two walk steps).
#[cfg(test)]
type WalkSeam = Box<dyn Fn(&OsStr) + Send>;
#[cfg(test)]
static WALK_SEAM: OnceLock<Mutex<Option<WalkSeam>>> = OnceLock::new();
#[cfg(test)]
pub(crate) fn install_walk_seam(hook: WalkSeam) {
    let m = WALK_SEAM.get_or_init(|| Mutex::new(None));
    *m.lock().expect("walk seam poisoned") = Some(hook);
}
#[cfg(test)]
pub(crate) fn clear_walk_seam() {
    if let Some(lock) = WALK_SEAM.get() {
        *lock.lock().expect("walk seam poisoned") = None;
    }
}
#[cfg(test)]
fn walk_seam(comp: &OsStr) {
    if let Some(lock) = WALK_SEAM.get() {
        if let Some(hook) = lock.lock().expect("walk seam poisoned").as_ref() {
            hook(comp);
        }
    }
}
#[cfg(not(test))]
fn walk_seam(_comp: &OsStr) {}
