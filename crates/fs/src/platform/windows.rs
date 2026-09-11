//! Windows reparse-safe, handle-relative workspace traversal (audits
//! 30/53/54).
//!
//! This is the Windows counterpart of `super::unix`: instead of a
//! component-by-component `openat(2)` walk, it opens the workspace root as a
//! directory HANDLE (`CreateFileW` + `FILE_FLAG_BACKUP_SEMANTICS` +
//! `FILE_FLAG_OPEN_REPARSE_POINT`), then opens every path component with
//! `NtCreateFile` relative to the previous component's HANDLE
//! (`OBJECT_ATTRIBUTES.RootDirectory`). A validated path is never reopened
//! by string: the handle walk IS the resolution.
//!
//! The same walk backs `lib.rs::resolve_within` on Windows
//! ([`canonicalize_within`]): `std::fs::canonicalize` cannot be used there
//! because it fails for a path whose tail does not exist yet (the atomic
//! writers create the parent directories AFTER resolving the destination)
//! and it cannot follow a relative reparse target whose stored substitute
//! name contains forward slashes (the Win32 parser keeps that spelling
//! verbatim). The walk reads substitute names itself, normalizes `/` and
//! `\`, follows only verified in-root targets, canonicalizes the deepest
//! existing ancestor by handle, and appends the not-yet-existing
//! components. The final component is
//! always opened with `FILE_OPEN_REPARSE_POINT`, so a reparse point is
//! handed to the walker, never silently followed by the kernel.
//!
//! Reparse policy
//! --------------
//! `FILE_ATTRIBUTE_REPARSE_POINT` on any walked component is inspected
//! explicitly. Only two tags are permitted, and only when their target can
//! be verified and normalized like unix in-root symlinks:
//!
//! * `IO_REPARSE_TAG_SYMLINK` (file or directory symlinks), and
//! * `IO_REPARSE_TAG_MOUNT_POINT` (NTFS junctions / mount points),
//!
//! and only when the target is a path (never a native `\Device\...`
//! object), normalizes to a location *inside* the canonical workspace root,
//! and is reached with at most [`MAX_SYMLINK_HOPS`] reparse hops. Absolute
//! targets must share the root's volume form (drive or UNC) and prefix;
//! relative targets are normalized against the current walk position and
//! denied when they would climb above the root. Every other tag
//! (`IO_REPARSE_TAG_APPEXECLINK`, cloud placeholders, WCI, ...) and every
//! target that cannot be verified is a typed `Permission` denial.
//!
//! Path hazards
//! ------------
//! Before any open, caller-supplied relative paths are validated by the
//! platform-independent [`split_relative`] / [`lexical_components`] pair:
//! `\\?\`, `\??\` (extended prefixes), `\\.\` (device namespace), UNC
//! roots, drive-relative (`C:foo`) and rooted forms, `..`, alternate data
//! stream colons (`file:stream`), NULs and embedded separators are all
//! rejected. Absolute paths are accepted only when they name a location
//! under the canonical root (same discipline as unix); the root itself and
//! reparse targets are compared component-wise with the case policy below.
//!
//! Case-fold policy (documented ambiguity)
//! ---------------------------------------
//! NTFS is case-insensitive by default but can be marked case-sensitive
//! per-directory. This walk opens each requested spelling exactly
//! (`OBJ_CASE_INSENSITIVE` makes a missing spelling fall back to a case
//! variant). Identity and containment comparisons fold ASCII case only;
//! non-ASCII case variants are conservatively *denied*, never silently
//! accepted. Callers must therefore treat two names that differ only by
//! case as one file: on a case-insensitive volume they ARE one file, and
//! this API cannot detect a case-sensitive directory, so a case-fold
//! collision is ambiguous by construction and resolves to whichever
//! spelling exists on disk.
//!
//! Identity
//! --------
//! Identity is volume serial + 128-bit file id (`FILE_ID_INFO`): the final
//! entry of a walk must live on the root's volume (a mount point to another
//! volume is denied even if its path string looks in-root), and the
//! post-open net in `lib.rs` ([`opened_is_path`]) compares the opened
//! handle's identity with a no-follow open of the canonical path, the
//! Windows equivalent of the unix `(dev, ino)` check. Filesystems without
//! stable file ids fail loudly instead of silently skipping the net.
//!
//! Honest limits
//! -------------
//! `fs::rename`/`fs::hard_link` have no compare-and-swap form on Windows
//! either, so the guarded writers keep the same recheck-to-rename window
//! unix documents; the parent-directory handle walk immediately before the
//! rename removes the swap-the-parent window. This module is compiled under
//! `cfg(test)` on unix hosts too, but only its platform-independent
//! validators/parser run there; the Win32 walk itself requires a Windows
//! runner.

use std::collections::VecDeque;
use std::path::Path;

use faktor_core::error::Error;

/// Hard bound on reparse hops per walk (parity with `super::unix`): at most
/// 8 symlink/junction hops may be followed while resolving ONE path, so a
/// reparse loop fails loudly instead of hanging.
pub(crate) const MAX_SYMLINK_HOPS: usize = 8;

/// Hard bound on the number of components one walk may consume (parity with
/// unix): a hostile `rel` of unbounded length fails loudly.
const MAX_COMPONENTS: usize = 4096;

/// Bound on a reparse target extracted from `FSCTL_GET_REPARSE_POINT`.
const MAX_REPARSE_TARGET_UNITS: usize = 32 * 1024;

const IO_REPARSE_TAG_MOUNT_POINT: u32 = 0xA000_0003;
const IO_REPARSE_TAG_SYMLINK: u32 = 0xA000_000C;
const SYMLINK_FLAG_RELATIVE: u32 = 1;

/// One rejected path hazard. The variants are the platform-independent
/// policy surface: they are exercised by unit tests on every host (the
/// windows walk is only one consumer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PathHazard {
    /// `\\?\` or `\??\` extended-length/NT prefix.
    ExtendedPrefix,
    /// `\\.\` device namespace.
    DeviceNamespace,
    /// `\\server\share` UNC root.
    UncRoot,
    /// `C:foo` (drive without a following separator).
    DriveRelative,
    /// A rooted path (leading separator or `C:\...`) outside the workspace.
    Rooted,
    /// `..` component.
    ParentDir,
    /// `:` inside a component (alternate data stream).
    AlternateDataStream,
    /// NUL inside a component.
    Nul,
    /// Separator inside a single component (defensive; splitting already
    /// removed them).
    Separator,
    /// The path normalizes outside the canonical workspace root.
    OutsideWorkspace,
    /// Beyond [`MAX_COMPONENTS`] components.
    TooManyComponents,
}

impl PathHazard {
    /// Typed error for a hazard (caller-facing kind + message).
    pub(crate) fn into_error(self, rel: &Path) -> Error {
        match self {
            PathHazard::ExtendedPrefix => Error::permission(format!(
                "path traversal rejected: {rel:?}: extended-length prefix (\\\\?\\ or \\??\\) is not accepted"
            )),
            PathHazard::DeviceNamespace => Error::permission(format!(
                "path traversal rejected: {rel:?}: device namespace (\\\\.\\) is denied"
            )),
            PathHazard::UncRoot => Error::permission(format!(
                "path traversal rejected: {rel:?}: UNC rooted paths are denied"
            )),
            PathHazard::DriveRelative => Error::permission(format!(
                "path traversal rejected: {rel:?}: drive-relative path is denied"
            )),
            PathHazard::Rooted => Error::permission(format!(
                "path traversal rejected: {rel:?}: rooted path is denied"
            )),
            PathHazard::ParentDir => Error::permission(format!(
                "path traversal rejected: {rel:?}"
            )),
            PathHazard::AlternateDataStream => Error::permission(format!(
                "alternate data stream rejected: {rel:?}"
            )),
            PathHazard::Nul => Error::malformed(format!(
                "path component contains a NUL byte: {rel:?}"
            )),
            PathHazard::Separator => Error::permission(format!(
                "path component contains a separator: {rel:?}"
            )),
            PathHazard::OutsideWorkspace => Error::permission(format!(
                "path escapes workspace: {rel:?}"
            )),
            PathHazard::TooManyComponents => Error::oversized(format!(
                "{rel:?} exceeds the {MAX_COMPONENTS}-component walk bound"
            )),
        }
    }
}

fn is_sep(u: u16) -> bool {
    u == u16::from(b'\\') || u == u16::from(b'/')
}

fn is_dot(comp: &[u16]) -> bool {
    comp == [u16::from(b'.')]
}

fn is_dotdot(comp: &[u16]) -> bool {
    comp == [u16::from(b'.'), u16::from(b'.')]
}

fn ascii_lower(u: u16) -> u16 {
    if (u16::from(b'A')..=u16::from(b'Z')).contains(&u) {
        u + 32
    } else {
        u
    }
}

fn eq_ascii_ci(a: u16, b: u16) -> bool {
    ascii_lower(a) == ascii_lower(b)
}

fn is_ascii_alpha(u: u16) -> bool {
    (u16::from(b'A')..=u16::from(b'Z')).contains(&u)
        || (u16::from(b'a')..=u16::from(b'z')).contains(&u)
}

fn starts_with_ci(hay: &[u16], needle: &str) -> bool {
    let mut expected = needle.encode_utf16();
    let mut actual = hay.iter().copied();
    loop {
        match (expected.next(), actual.next()) {
            (Some(e), Some(a)) if eq_ascii_ci(a, e) => {}
            (None, _) => return true,
            _ => return false,
        }
    }
}

fn units_eq_ci(a: &[u16], b: &[u16]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| eq_ascii_ci(*x, *y))
}

fn lower_units(units: &[u16]) -> Vec<u16> {
    units.iter().map(|u| ascii_lower(*u)).collect()
}

/// Split raw units on both separators, dropping empty segments (Win32
/// collapses repeated separators and a trailing separator).
fn split_raw(units: &[u16]) -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for &u in units {
        if is_sep(u) {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(u);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Validate one already-split component against the Windows path hazards.
fn validate_component(comp: &[u16]) -> Result<(), PathHazard> {
    if comp.is_empty() {
        return Err(PathHazard::Rooted);
    }
    if comp.contains(&0) {
        return Err(PathHazard::Nul);
    }
    if comp.contains(&u16::from(b':')) {
        return Err(PathHazard::AlternateDataStream);
    }
    if comp.iter().any(|u| is_sep(*u)) {
        return Err(PathHazard::Separator);
    }
    Ok(())
}

fn is_drive_spec(units: &[u16]) -> bool {
    units.len() >= 2 && units[1] == u16::from(b':') && is_ascii_alpha(units[0])
}

fn looks_rooted(units: &[u16]) -> bool {
    units.first().is_some_and(|u| is_sep(*u)) || is_drive_spec(units)
}

fn validate_caller_components(comps: Vec<Vec<u16>>) -> Result<Vec<Vec<u16>>, PathHazard> {
    let mut out = Vec::with_capacity(comps.len());
    for comp in comps {
        if is_dot(&comp) {
            continue;
        }
        if is_dotdot(&comp) {
            return Err(PathHazard::ParentDir);
        }
        validate_component(&comp)?;
        out.push(comp);
    }
    Ok(out)
}

/// Split and validate a caller-supplied *relative* path. Absolute/rooted
/// forms are hazards here; [`lexical_components`] handles absolute paths
/// that name a location under the canonical root.
pub(crate) fn split_relative(rel: &[u16]) -> Result<Vec<Vec<u16>>, PathHazard> {
    if starts_with_ci(rel, r"\\?\") || starts_with_ci(rel, r"\??\") {
        return Err(PathHazard::ExtendedPrefix);
    }
    if starts_with_ci(rel, r"\\.\") {
        return Err(PathHazard::DeviceNamespace);
    }
    if rel.first().is_some_and(|u| is_sep(*u)) {
        return Err(if rel.get(1).is_some_and(|u| is_sep(*u)) {
            PathHazard::UncRoot
        } else {
            PathHazard::Rooted
        });
    }
    if is_drive_spec(rel) {
        return Err(if rel.len() == 2 || !is_sep(rel[2]) {
            PathHazard::DriveRelative
        } else {
            PathHazard::Rooted
        });
    }
    validate_caller_components(split_raw(rel))
}

/// Lexically split + validate `rel` against the canonical `root`, never
/// touching the filesystem. `..`, absolute escapes, rooted/device forms and
/// ADS/NUL components are denied before any open.
pub(crate) fn lexical_components(root: &[u16], rel: &[u16]) -> Result<Vec<Vec<u16>>, PathHazard> {
    if rel.is_empty() {
        return Ok(Vec::new());
    }
    // Rejected even when the prefix names the root itself.
    if starts_with_ci(rel, r"\\?\") || starts_with_ci(rel, r"\??\") {
        return Err(PathHazard::ExtendedPrefix);
    }
    if starts_with_ci(rel, r"\\.\") {
        return Err(PathHazard::DeviceNamespace);
    }
    let comps = if looks_rooted(rel) {
        validate_caller_components(strip_root_prefix(root, rel)?)?
    } else {
        split_relative(rel)?
    };
    if comps.len() > MAX_COMPONENTS {
        return Err(PathHazard::TooManyComponents);
    }
    Ok(comps)
}

/// A parsed Windows root marker, comparable across spelling variants
/// (`C:`, `\??\C:`, `\\?\C:`, `\\server\share`, `\??\UNC\server\share`).
#[derive(Debug, Clone, PartialEq, Eq)]
enum RootForm {
    Drive(Vec<u16>),
    Unc(Vec<u16>, Vec<u16>),
}

fn parse_rooted(units: &[u16]) -> Option<(RootForm, Vec<Vec<u16>>)> {
    let mut rest = units;
    if starts_with_ci(rest, r"\\?\") || starts_with_ci(rest, r"\??\") {
        rest = &rest[4..];
    }
    if starts_with_ci(rest, r"UNC\") {
        rest = &rest[4..];
        let comps = split_raw(rest);
        if comps.len() < 2 {
            return None;
        }
        return Some((
            RootForm::Unc(lower_units(&comps[0]), lower_units(&comps[1])),
            comps[2..].to_vec(),
        ));
    }
    if rest.len() >= 2 && is_sep(rest[0]) && is_sep(rest[1]) {
        let comps = split_raw(rest);
        if comps.len() < 2 {
            return None;
        }
        return Some((
            RootForm::Unc(lower_units(&comps[0]), lower_units(&comps[1])),
            comps[2..].to_vec(),
        ));
    }
    if is_drive_spec(rest) {
        return Some((
            RootForm::Drive(lower_units(&rest[..2])),
            split_raw(&rest[2..]),
        ));
    }
    // Single-rooted NT-native paths (`\Device\...`) have no comparable
    // Win32 root: they can never be verified against a workspace root.
    None
}

fn root_form_eq(a: &RootForm, b: &RootForm) -> bool {
    match (a, b) {
        (RootForm::Drive(a), RootForm::Drive(b)) => units_eq_ci(a, b),
        (RootForm::Unc(sa, ha), RootForm::Unc(sb, hb)) => {
            units_eq_ci(sa, sb) && units_eq_ci(ha, hb)
        }
        _ => false,
    }
}

/// Strip the canonical `root` from an absolute `path`, component-wise and
/// case-insensitively (ASCII case policy, documented above). Returns the
/// remaining components relative to the root, or a hazard when the path is
/// not verifiably inside it.
fn strip_root_prefix(root: &[u16], path: &[u16]) -> Result<Vec<Vec<u16>>, PathHazard> {
    let (root_form, root_comps) = parse_rooted(root).ok_or(PathHazard::OutsideWorkspace)?;
    let (path_form, path_comps) = parse_rooted(path).ok_or(PathHazard::OutsideWorkspace)?;
    if !root_form_eq(&root_form, &path_form) {
        return Err(PathHazard::OutsideWorkspace);
    }
    if path_comps.len() < root_comps.len()
        || !path_comps[..root_comps.len()]
            .iter()
            .zip(&root_comps)
            .all(|(a, b)| units_eq_ci(a, b))
    {
        return Err(PathHazard::OutsideWorkspace);
    }
    Ok(path_comps[root_comps.len()..].to_vec())
}

/// Where a reparse hop sends the walk (same shape as the unix `Rebase`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Rebase {
    /// Relative target without `..`: the walk continues from the SAME
    /// directory handle with the target components in front.
    Stay(VecDeque<Vec<u16>>),
    /// Absolute target (verified in-root) or a relative target containing
    /// `..`: the walk restarts from the trusted root handle.
    Relocate(VecDeque<Vec<u16>>),
}

/// Normalize a reparse target against the walk position. `relative` is the
/// symlink `SYMLINK_FLAG_RELATIVE` flag (junctions are always absolute).
/// Only targets that stay inside the canonical root are accepted.
pub(crate) fn rebase_target(
    root: &[u16],
    pos: &[Vec<u16>],
    target: &[u16],
    relative: bool,
) -> Result<Rebase, PathHazard> {
    let rooted = looks_rooted(target);
    if relative == rooted {
        // (relative && rooted) or (!relative && !rooted): the flag and the
        // target syntax disagree — a malformed/junction-relative target.
        return Err(PathHazard::OutsideWorkspace);
    }
    if relative {
        let mut acc = pos.to_vec();
        let mut pure = Vec::new();
        let mut saw_parent = false;
        for comp in split_raw(target) {
            if is_dot(&comp) {
                continue;
            }
            if is_dotdot(&comp) {
                saw_parent = true;
                if acc.pop().is_none() {
                    return Err(PathHazard::OutsideWorkspace);
                }
                continue;
            }
            validate_component(&comp)?;
            pure.push(comp.clone());
            acc.push(comp);
        }
        return Ok(if saw_parent {
            Rebase::Relocate(acc.into())
        } else {
            Rebase::Stay(pure.into())
        });
    }
    let mut acc: Vec<Vec<u16>> = Vec::new();
    for comp in strip_root_prefix(root, target)? {
        if is_dot(&comp) {
            continue;
        }
        if is_dotdot(&comp) {
            if acc.pop().is_none() {
                return Err(PathHazard::OutsideWorkspace);
            }
            continue;
        }
        validate_component(&comp)?;
        acc.push(comp);
    }
    Ok(Rebase::Relocate(acc.into()))
}

/// A parsed reparse point: permitted tags only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedReparse {
    pub(crate) tag: u32,
    pub(crate) flags: u32,
    pub(crate) substitute: Vec<u16>,
    pub(crate) print: Vec<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReparseError {
    TooShort,
    Truncated,
    OddLength,
    TargetTooLong,
    NulInTarget,
    UnsupportedTag,
}

fn read_u16(data: &[u8], at: usize) -> Result<u16, ReparseError> {
    let bytes = data.get(at..at + 2).ok_or(ReparseError::Truncated)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(data: &[u8], at: usize) -> Result<u32, ReparseError> {
    let bytes = data.get(at..at + 4).ok_or(ReparseError::Truncated)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn utf16_field(
    body: &[u8],
    base: usize,
    offset: usize,
    len: usize,
) -> Result<Vec<u16>, ReparseError> {
    if !len.is_multiple_of(2) {
        return Err(ReparseError::OddLength);
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    let start = base.checked_add(offset).ok_or(ReparseError::Truncated)?;
    let end = start.checked_add(len).ok_or(ReparseError::Truncated)?;
    let bytes = body.get(start..end).ok_or(ReparseError::Truncated)?;
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    if units.len() > MAX_REPARSE_TARGET_UNITS {
        return Err(ReparseError::TargetTooLong);
    }
    if units.contains(&0) {
        return Err(ReparseError::NulInTarget);
    }
    Ok(units)
}

/// Parse a `REPARSE_DATA_BUFFER` as returned by
/// `FSCTL_GET_REPARSE_POINT`. Only the symlink and mount-point layouts are
/// whitelisted; everything else (including unknown Microsoft/third-party
/// tags) is [`ReparseError::UnsupportedTag`].
pub(crate) fn parse_reparse_data(data: &[u8]) -> Result<ParsedReparse, ReparseError> {
    if data.len() < 8 {
        return Err(ReparseError::TooShort);
    }
    let tag = read_u32(data, 0)?;
    let data_len = read_u16(data, 4)? as usize;
    let end = 8usize
        .checked_add(data_len)
        .ok_or(ReparseError::Truncated)?;
    if data.len() < end {
        return Err(ReparseError::Truncated);
    }
    let body = &data[8..end];
    match tag {
        IO_REPARSE_TAG_SYMLINK => {
            if body.len() < 12 {
                return Err(ReparseError::Truncated);
            }
            let sub_off = read_u16(body, 0)? as usize;
            let sub_len = read_u16(body, 2)? as usize;
            let print_off = read_u16(body, 4)? as usize;
            let print_len = read_u16(body, 6)? as usize;
            let flags = read_u32(body, 8)?;
            let substitute = utf16_field(body, 12, sub_off, sub_len)?;
            let print = utf16_field(body, 12, print_off, print_len)?;
            Ok(ParsedReparse {
                tag,
                flags,
                substitute,
                print,
            })
        }
        IO_REPARSE_TAG_MOUNT_POINT => {
            if body.len() < 8 {
                return Err(ReparseError::Truncated);
            }
            let sub_off = read_u16(body, 0)? as usize;
            let sub_len = read_u16(body, 2)? as usize;
            let print_off = read_u16(body, 4)? as usize;
            let print_len = read_u16(body, 6)? as usize;
            let substitute = utf16_field(body, 8, sub_off, sub_len)?;
            let print = utf16_field(body, 8, print_off, print_len)?;
            Ok(ParsedReparse {
                tag,
                flags: 0,
                substitute,
                print,
            })
        }
        _ => Err(ReparseError::UnsupportedTag),
    }
}

#[cfg(windows)]
mod nt {
    use std::collections::VecDeque;
    use std::ffi::{c_void, OsString};
    use std::fs::File;
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use std::path::{Path, PathBuf};
    use std::ptr;

    use faktor_core::error::Error;
    use windows_sys::Wdk::Foundation::OBJECT_ATTRIBUTES;
    use windows_sys::Wdk::Storage::FileSystem::{
        NtCreateFile, FILE_DIRECTORY_FILE, FILE_OPEN, FILE_OPEN_FOR_BACKUP_INTENT,
        FILE_OPEN_REPARSE_POINT, FILE_SYNCHRONOUS_IO_NONALERT,
    };
    use windows_sys::Win32::Foundation::{
        GetLastError, RtlNtStatusToDosError, ERROR_ACCESS_DENIED, ERROR_FILE_NOT_FOUND,
        ERROR_INSUFFICIENT_BUFFER, ERROR_MORE_DATA, ERROR_PATH_NOT_FOUND, HANDLE,
        INVALID_HANDLE_VALUE, NTSTATUS, OBJ_CASE_INSENSITIVE, STATUS_ACCESS_DENIED,
        STATUS_NOT_A_DIRECTORY, STATUS_OBJECT_NAME_NOT_FOUND, STATUS_OBJECT_PATH_NOT_FOUND,
        UNICODE_STRING,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FileAttributeTagInfo, FileIdInfo, GetFileInformationByHandleEx,
        GetFinalPathNameByHandleW, FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_REPARSE_POINT,
        FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_ID_INFO,
        FILE_LIST_DIRECTORY, FILE_READ_ATTRIBUTES, FILE_READ_DATA, FILE_READ_EA, FILE_SHARE_DELETE,
        FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING, SYNCHRONIZE, VOLUME_NAME_DOS,
    };
    use windows_sys::Win32::System::Ioctl::FSCTL_GET_REPARSE_POINT;
    use windows_sys::Win32::System::IO::{DeviceIoControl, IO_STATUS_BLOCK};

    use super::{
        lexical_components, parse_reparse_data, rebase_target, strip_root_prefix, ParsedReparse,
        Rebase, IO_REPARSE_TAG_MOUNT_POINT, IO_REPARSE_TAG_SYMLINK, MAX_COMPONENTS,
        MAX_SYMLINK_HOPS, SYMLINK_FLAG_RELATIVE,
    };
    use crate::platform::OpenKind;

    const INITIAL_REPARSE_BUF: usize = 16 * 1024;
    const MAX_REPARSE_BUF: usize = 128 * 1024;

    /// Volume serial + 128-bit file id: the Windows identity pair.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct FileIdentity {
        volume_serial: u64,
        file_id: [u8; 16],
    }

    /// Outcome of one walk. On success `missing` is empty and `handle` is
    /// the requested entry; in `tolerate_missing` mode `handle` is the
    /// deepest existing directory and `missing` holds the components (in
    /// order) that do not exist yet.
    struct WalkOutcome {
        handle: OwnedHandle,
        missing: VecDeque<Vec<u16>>,
    }

    fn raw(h: &OwnedHandle) -> HANDLE {
        h.as_raw_handle() as HANDLE
    }

    fn identity_from_handle(handle: HANDLE) -> Option<FileIdentity> {
        let mut info = FILE_ID_INFO::default();
        let ok = unsafe {
            GetFileInformationByHandleEx(
                handle,
                FileIdInfo,
                (&mut info as *mut FILE_ID_INFO).cast::<c_void>(),
                std::mem::size_of::<FILE_ID_INFO>() as u32,
            )
        };
        (ok != 0).then_some(FileIdentity {
            volume_serial: info.VolumeSerialNumber,
            file_id: info.FileId.Identifier,
        })
    }

    fn attribute_tag(
        handle: HANDLE,
    ) -> Option<windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_TAG_INFO> {
        let mut info = windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_TAG_INFO::default();
        let ok = unsafe {
            GetFileInformationByHandleEx(
                handle,
                FileAttributeTagInfo,
                (&mut info
                    as *mut windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_TAG_INFO)
                    .cast::<c_void>(),
                std::mem::size_of::<windows_sys::Win32::Storage::FileSystem::FILE_ATTRIBUTE_TAG_INFO>(
                ) as u32,
            )
        };
        (ok != 0).then_some(info)
    }

    fn require_identity(handle: HANDLE, rel: &Path) -> Result<FileIdentity, Error> {
        identity_from_handle(handle).ok_or_else(|| {
            Error::internal(format!(
                "{}: file identity (volume serial + file id) is unavailable; \
                 the workspace volume does not report stable file ids",
                rel.display()
            ))
        })
    }

    fn open_root(root: &Path) -> Result<OwnedHandle, Error> {
        let wide: Vec<u16> = root
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `wide` is a NUL-terminated UTF-16 string; the remaining
        // pointers are null and the flags ask for a directory handle whose
        // final entry is not followed. Desired access follows the canonical
        // directory recipe (`FILE_LIST_DIRECTORY | SYNCHRONIZE`) plus
        // `FILE_READ_ATTRIBUTES`, which this module needs for the reparse
        // attribute/identity queries.
        let handle = unsafe {
            CreateFileW(
                wide.as_ptr(),
                FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(map_win32(root, unsafe { GetLastError() }));
        }
        // SAFETY: the handle came from CreateFileW with a non-invalid value.
        let owned = unsafe { OwnedHandle::from_raw_handle(handle) };
        let info = attribute_tag(raw(&owned)).ok_or_else(|| {
            Error::internal(format!(
                "workspace root {}: attribute query failed",
                root.display()
            ))
        })?;
        if info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(Error::permission(format!(
                "workspace root {} was swapped for a reparse point",
                root.display()
            )));
        }
        if info.FileAttributes & FILE_ATTRIBUTE_DIRECTORY == 0 {
            return Err(Error::not_found(format!(
                "workspace root {} is not a directory",
                root.display()
            )));
        }
        Ok(owned)
    }

    fn map_win32(what: &Path, code: u32) -> Error {
        match code {
            ERROR_FILE_NOT_FOUND | ERROR_PATH_NOT_FOUND => {
                Error::not_found(format!("{}", what.display()))
            }
            ERROR_ACCESS_DENIED => Error::permission(format!("{}: access denied", what.display())),
            _ => Error::internal(format!("{}: win32 error {code}", what.display())),
        }
    }

    enum NtOpenError {
        Status(NTSTATUS),
        NameTooLong,
        EmptyName,
    }

    /// Render the accumulated walk position (pure components of the last
    /// pinned directory) for diagnostics.
    fn render_pos(pos: &[Vec<u16>]) -> String {
        if pos.is_empty() {
            return String::from("\\");
        }
        let mut out = String::new();
        for comp in pos {
            out.push('\\');
            out.push_str(&String::from_utf16_lossy(comp));
        }
        out
    }

    /// Position diagnostics for a walk failure that has no NTSTATUS of its
    /// own (policy denials): component index, name and resolved position.
    fn walk_diag(index: usize, comp: &[u16], pos: &[Vec<u16>]) -> String {
        format!(
            "at component {index} ({:?}), resolved {:?}",
            String::from_utf16_lossy(comp),
            render_pos(pos),
        )
    }

    /// Map a component open failure to a typed error, carrying the component
    /// index, the accumulated resolved position and the raw NTSTATUS so a
    /// Windows runner failure is diagnosable from the panic alone.
    fn map_nt_error(
        rel: &Path,
        index: usize,
        comp: &[u16],
        pos: &[Vec<u16>],
        last: bool,
        err: NtOpenError,
    ) -> Error {
        let status = match err {
            NtOpenError::EmptyName => {
                return Error::malformed(format!(
                    "{rel:?}: empty path component {}",
                    walk_diag(index, comp, pos)
                ))
            }
            NtOpenError::NameTooLong => {
                return Error::oversized(format!(
                    "{rel:?}: path component {:?} exceeds the NT name bound ({})",
                    String::from_utf16_lossy(comp),
                    walk_diag(index, comp, pos),
                ))
            }
            NtOpenError::Status(status) => status,
        };
        let diag = format!(
            "component {index} {:?}, resolved {:?}, NTSTATUS {:#010X}",
            String::from_utf16_lossy(comp),
            render_pos(pos),
            status as u32,
        );
        match status {
            STATUS_OBJECT_NAME_NOT_FOUND | STATUS_OBJECT_PATH_NOT_FOUND if last => {
                Error::not_found(format!("{} ({diag})", rel.display()))
            }
            STATUS_OBJECT_NAME_NOT_FOUND | STATUS_OBJECT_PATH_NOT_FOUND => {
                Error::permission(format!("parent resolution failed: {rel:?} ({diag})"))
            }
            STATUS_NOT_A_DIRECTORY => Error::permission(format!(
                "parent resolution failed: {rel:?}: component is not a directory ({diag})"
            )),
            STATUS_ACCESS_DENIED if !last => {
                Error::permission(format!("parent resolution failed: {rel:?} ({diag})"))
            }
            _ if last => Error::internal(format!("{} ({diag}; win32 {})", rel.display(), unsafe {
                RtlNtStatusToDosError(status)
            })),
            _ => Error::permission(format!("parent resolution failed: {rel:?} ({diag})")),
        }
    }

    /// STATUS codes that mean "this component does not exist" rather than
    /// "it exists but the open failed".
    fn is_missing_status(status: NTSTATUS) -> bool {
        status == STATUS_OBJECT_NAME_NOT_FOUND || status == STATUS_OBJECT_PATH_NOT_FOUND
    }

    /// Open `name` relative to the parent directory handle, never following
    /// the final reparse point. `directory` requests a directory handle
    /// (`FILE_DIRECTORY_FILE`); intermediates are always directories.
    ///
    /// Canonical NT recipe (MSDN `NtCreateFile` / `OBJECT_ATTRIBUTES`):
    /// length-delimited `UNICODE_STRING` (no NUL; a Windows file name may
    /// legally contain none, and the string is not NUL-terminated),
    /// `OBJ_CASE_INSENSITIVE`, RWD share access, `FILE_OPEN` disposition,
    /// `FILE_SYNCHRONOUS_IO_NONALERT`, and `FILE_OPEN_REPARSE_POINT` so the
    /// walker — not the kernel — decides whether a reparse point is
    /// followed. `FILE_OPEN_FOR_BACKUP_INTENT` keeps traversal working
    /// through entries whose ACL denies ordinary access (the walk still
    /// re-validates every target itself).
    fn nt_open_relative(
        parent: HANDLE,
        name: &[u16],
        directory: bool,
    ) -> Result<OwnedHandle, NtOpenError> {
        if name.is_empty() {
            return Err(NtOpenError::EmptyName);
        }
        if name.len() > (u16::MAX as usize) / 2 {
            return Err(NtOpenError::NameTooLong);
        }
        let unicode = UNICODE_STRING {
            Length: (name.len() * 2) as u16,
            MaximumLength: (name.len() * 2) as u16,
            Buffer: name.as_ptr() as *mut u16,
        };
        let attrs = OBJECT_ATTRIBUTES {
            Length: std::mem::size_of::<OBJECT_ATTRIBUTES>() as u32,
            RootDirectory: parent,
            ObjectName: &unicode,
            Attributes: OBJ_CASE_INSENSITIVE,
            SecurityDescriptor: ptr::null(),
            SecurityQualityOfService: ptr::null(),
        };
        // Intermediates need only list the directory plus the attribute
        // query the reparse check performs; a final file needs read data
        // (`FILE_LIST_DIRECTORY` is the directory form of `FILE_READ_DATA`).
        let access = if directory {
            FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES | SYNCHRONIZE
        } else {
            FILE_READ_DATA | FILE_READ_ATTRIBUTES | FILE_READ_EA | SYNCHRONIZE
        };
        let mut options =
            FILE_OPEN_REPARSE_POINT | FILE_OPEN_FOR_BACKUP_INTENT | FILE_SYNCHRONOUS_IO_NONALERT;
        if directory {
            options |= FILE_DIRECTORY_FILE;
        }
        let mut handle: HANDLE = ptr::null_mut();
        let mut iosb = IO_STATUS_BLOCK::default();
        // SAFETY: the OBJECT_ATTRIBUTES point at a live UNICODE_STRING whose
        // buffer is `name`; `handle`/`iosb` are valid out-parameters; no
        // allocation/EAs are requested.
        let status = unsafe {
            NtCreateFile(
                &mut handle,
                access,
                &attrs,
                &mut iosb,
                ptr::null(),
                0,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                FILE_OPEN,
                options,
                ptr::null(),
                0,
            )
        };
        if status < 0 {
            return Err(NtOpenError::Status(status));
        }
        // SAFETY: NtCreateFile succeeded and the handle is owned by us.
        Ok(unsafe { OwnedHandle::from_raw_handle(handle) })
    }

    fn read_reparse(handle: HANDLE) -> Result<ParsedReparse, Error> {
        let mut buf = vec![0u8; INITIAL_REPARSE_BUF];
        loop {
            let mut returned = 0u32;
            // SAFETY: `buf` is a writable output buffer of `buf.len()` bytes
            // and the input buffer is empty (FSCTL_GET_REPARSE_POINT takes
            // none).
            let ok = unsafe {
                DeviceIoControl(
                    handle,
                    FSCTL_GET_REPARSE_POINT,
                    ptr::null(),
                    0,
                    buf.as_mut_ptr().cast::<c_void>(),
                    buf.len() as u32,
                    &mut returned,
                    ptr::null_mut(),
                )
            };
            if ok != 0 {
                let n = (returned as usize).min(buf.len());
                return parse_reparse_data(&buf[..n]).map_err(|e| {
                    Error::permission(format!("reparse point data is not a permitted link: {e:?}"))
                });
            }
            let code = unsafe { GetLastError() };
            if code == ERROR_MORE_DATA || code == ERROR_INSUFFICIENT_BUFFER {
                if buf.len() >= MAX_REPARSE_BUF {
                    return Err(Error::oversized(format!(
                        "reparse point data exceeds the {MAX_REPARSE_BUF}-byte bound"
                    )));
                }
                buf.resize(buf.len() * 2, 0);
                continue;
            }
            return Err(Error::internal(format!(
                "FSCTL_GET_REPARSE_POINT failed: win32 error {code}"
            )));
        }
    }

    /// One walk over `rel` under the canonical directory `root`.
    ///
    /// `seam` fires the deterministic test hook before each component open;
    /// `tolerate_missing` returns the deepest existing directory plus the
    /// components that do not exist yet instead of failing when a component
    /// is absent (used by [`canonicalize_within`], which must resolve paths
    /// a writer is about to create).
    fn walk(
        root: &Path,
        rel: &Path,
        kind: OpenKind,
        seam: bool,
        tolerate_missing: bool,
    ) -> Result<WalkOutcome, Error> {
        let root_units: Vec<u16> = root.as_os_str().encode_wide().collect();
        let rel_units: Vec<u16> = rel.as_os_str().encode_wide().collect();
        let mut pending: VecDeque<Vec<u16>> = lexical_components(&root_units, &rel_units)
            .map_err(|h| h.into_error(rel))?
            .into();
        if pending.len() > MAX_COMPONENTS {
            return Err(Error::oversized(format!(
                "{rel:?} exceeds the {MAX_COMPONENTS}-component walk bound"
            )));
        }
        let root_handle = open_root(root)?;
        let root_id = require_identity(raw(&root_handle), rel)?;
        let mut dir = root_handle;
        let mut pos: Vec<Vec<u16>> = Vec::new();
        let mut hops = 0usize;
        let mut index = 0usize;
        loop {
            let Some(comp) = pending.pop_front() else {
                let id = require_identity(raw(&dir), rel)?;
                if id.volume_serial != root_id.volume_serial {
                    return Err(Error::permission(format!(
                        "{rel:?}: resolved entry is on volume {:#x}, not the workspace volume {:#x}",
                        id.volume_serial, root_id.volume_serial
                    )));
                }
                return Ok(WalkOutcome {
                    handle: dir,
                    missing: VecDeque::new(),
                });
            };
            if seam {
                walk_seam(&comp);
            }
            let last = pending.is_empty();
            let directory = !last || matches!(kind, OpenKind::Directory);
            let handle = match nt_open_relative(raw(&dir), &comp, directory) {
                Ok(handle) => handle,
                Err(NtOpenError::Status(status))
                    if tolerate_missing && is_missing_status(status) =>
                {
                    // `dir` is the deepest existing directory; this component
                    // and everything after it do not exist yet.
                    let mut missing = pending;
                    missing.push_front(comp);
                    return Ok(WalkOutcome {
                        handle: dir,
                        missing,
                    });
                }
                Err(e) => return Err(map_nt_error(rel, index, &comp, &pos, last, e)),
            };
            let comp_index = index;
            index += 1;
            let info = attribute_tag(raw(&handle)).ok_or_else(|| {
                Error::internal(format!(
                    "{rel:?}: attribute query failed at component {comp_index} ({:?}, resolved {:?})",
                    String::from_utf16_lossy(&comp),
                    render_pos(&pos)
                ))
            })?;
            if info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
                hops += 1;
                if hops > MAX_SYMLINK_HOPS {
                    return Err(Error::permission(format!(
                        "{rel:?}: reparse point resolution exceeded the {MAX_SYMLINK_HOPS}-hop bound (possible loop) {}",
                        walk_diag(comp_index, &comp, &pos),
                    )));
                }
                let parsed = read_reparse(raw(&handle)).map_err(|mut e| {
                    e.message
                        .push_str(&format!(" ({})", walk_diag(comp_index, &comp, &pos)));
                    e
                })?;
                let (target, relative_target) = match parsed.tag {
                    IO_REPARSE_TAG_SYMLINK => (
                        parsed.substitute.as_slice(),
                        parsed.flags & SYMLINK_FLAG_RELATIVE != 0,
                    ),
                    IO_REPARSE_TAG_MOUNT_POINT => (parsed.substitute.as_slice(), false),
                    other => {
                        return Err(Error::permission(format!(
                            "{rel:?}: reparse tag {other:#010X} is not permitted in a workspace path {}",
                            walk_diag(comp_index, &comp, &pos),
                        )));
                    }
                };
                match rebase_target(&root_units, &pos, target, relative_target) {
                    Ok(Rebase::Stay(comps)) => {
                        // Relative target without `..`: continue from the
                        // link's parent directory handle (the current `dir`),
                        // with the target's components in front.
                        for c in comps.into_iter().rev() {
                            pending.push_front(c);
                        }
                    }
                    Ok(Rebase::Relocate(comps)) => {
                        // Absolute in-root target or a relative target with
                        // `..`: restart from the trusted root handle so every
                        // component is re-validated as it is really opened.
                        for c in comps.into_iter().rev() {
                            pending.push_front(c);
                        }
                        pos.clear();
                        dir = open_root(root)?;
                    }
                    Err(hazard) => {
                        let print = String::from_utf16_lossy(&parsed.print);
                        let mut err = hazard.into_error(rel);
                        err.message.push_str(&format!(
                            " ({}; reparse target {:?}, print name {:?})",
                            walk_diag(comp_index, &comp, &pos),
                            String::from_utf16_lossy(target),
                            print
                        ));
                        return Err(err);
                    }
                }
                if pending.len() > MAX_COMPONENTS {
                    return Err(Error::oversized(format!(
                        "{rel:?} exceeds the {MAX_COMPONENTS}-component walk bound {}",
                        walk_diag(comp_index, &comp, &pos),
                    )));
                }
                continue;
            }
            if last {
                let id = require_identity(raw(&handle), rel)?;
                if id.volume_serial != root_id.volume_serial {
                    return Err(Error::permission(format!(
                        "{rel:?}: resolved entry is on volume {:#x}, not the workspace volume {:#x}",
                        id.volume_serial, root_id.volume_serial
                    )));
                }
                return Ok(WalkOutcome {
                    handle,
                    missing: VecDeque::new(),
                });
            }
            pos.push(comp);
            dir = handle;
        }
    }

    /// Open the entry named by `rel` under the canonical directory `root`.
    ///
    /// Identical contract to `super::unix::open_no_follow_walk`: the walk is
    /// the resolution, no path string is re-resolved after it starts,
    /// permitted reparse points are followed only by explicit bounded
    /// re-anchoring, and the final entry is opened without following.
    pub(crate) fn open_no_follow_walk(
        root: &Path,
        rel: &Path,
        kind: OpenKind,
    ) -> Result<OwnedHandle, Error> {
        walk(root, rel, kind, true, false).map(|outcome| outcome.handle)
    }

    /// Resolve `rel` under the canonical `root` to an absolute canonical
    /// path via the handle walk — never `std::fs::canonicalize`.
    ///
    /// Two Windows realities make the Win32 canonicalize unusable here:
    ///
    /// 1. `std::fs::canonicalize` fails outright for a path whose tail does
    ///    not exist yet, but `write_atomic`/`write_atomic_cas` resolve the
    ///    destination and only then create its parent directories.
    /// 2. A relative symlink target containing forward slashes is stored
    ///    verbatim by `CreateSymbolicLinkW` and cannot be followed by the
    ///    Win32 parser, while this walker reads the substitute name out of
    ///    the reparse buffer and normalizes both separators itself.
    ///
    /// The deepest existing ancestor is canonicalized from its handle
    /// (`GetFinalPathNameByHandleW`, `VOLUME_NAME_DOS`) and the missing,
    /// already lexically validated components are appended; the result is
    /// re-checked component-wise against `root` before it is returned.
    pub(crate) fn canonicalize_within(root: &Path, rel: &Path) -> Result<PathBuf, Error> {
        let outcome = match walk(root, rel, OpenKind::Read, false, true) {
            Ok(outcome) => outcome,
            Err(first) => match walk(root, rel, OpenKind::Directory, false, true) {
                Ok(outcome) => outcome,
                Err(_) => return Err(first),
            },
        };
        let mut path = final_path_of(raw(&outcome.handle), rel)?;
        for comp in &outcome.missing {
            path.push(OsString::from_wide(comp));
        }
        // Defense in depth: the walk only follows in-root reparse targets
        // and checks the resolved volume, but the OS-reported name must
        // also still name a location under the canonical root.
        let root_units: Vec<u16> = root.as_os_str().encode_wide().collect();
        let path_units: Vec<u16> = path.as_os_str().encode_wide().collect();
        strip_root_prefix(&root_units, &path_units).map_err(|_| {
            Error::permission(format!(
                "path escapes workspace: {rel:?} (resolved {})",
                path.display()
            ))
        })?;
        Ok(path)
    }

    /// Canonical Win32 path of an open handle
    /// (`GetFinalPathNameByHandleW` with `VOLUME_NAME_DOS`, which yields the
    /// `\\?\` form). The handle is the authority; no path string is ever
    /// re-resolved.
    fn final_path_of(handle: HANDLE, rel: &Path) -> Result<PathBuf, Error> {
        let mut buf = vec![0u16; 260];
        loop {
            // SAFETY: `buf` is a writable output buffer of `buf.len()`
            // UTF-16 units and `handle` is a live file/directory handle.
            let n = unsafe {
                GetFinalPathNameByHandleW(
                    handle,
                    buf.as_mut_ptr(),
                    buf.len() as u32,
                    VOLUME_NAME_DOS,
                )
            };
            if n == 0 {
                let code = unsafe { GetLastError() };
                return Err(Error::internal(format!(
                    "{}: GetFinalPathNameByHandleW failed: win32 error {code}",
                    rel.display()
                )));
            }
            let n = n as usize;
            if n < buf.len() {
                buf.truncate(n);
                return Ok(PathBuf::from(OsString::from_wide(&buf)));
            }
            // The return value then includes the terminating NUL.
            buf.resize(n + 1, 0);
        }
    }

    /// Pure lexical pre-screen used by `lib.rs::resolve_within` on Windows:
    /// a caller-supplied path is rejected before any string canonicalization
    /// when it carries a hazard (extended/device prefixes, UNC escapes,
    /// drive-relative forms, `..`, ADS, NULs).
    pub(crate) fn lexical_check(root: &Path, rel: &Path) -> Result<(), Error> {
        let root_units: Vec<u16> = root.as_os_str().encode_wide().collect();
        let rel_units: Vec<u16> = rel.as_os_str().encode_wide().collect();
        lexical_components(&root_units, &rel_units)
            .map(|_| ())
            .map_err(|h| h.into_error(rel))
    }

    fn open_path_no_follow(path: &Path) -> Option<OwnedHandle> {
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `wide` is NUL-terminated; null pointers/flags are valid.
        let handle = unsafe {
            CreateFileW(
                wide.as_ptr(),
                FILE_READ_ATTRIBUTES | SYNCHRONIZE,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return None;
        }
        // SAFETY: the handle came from CreateFileW with a non-invalid value.
        Some(unsafe { OwnedHandle::from_raw_handle(handle) })
    }

    /// Post-open identity net (audit 47, Windows): the opened handle's
    /// (volume serial, file id) must equal a no-follow open of the canonical
    /// path. A directory entry swapped between the walk and this check — or
    /// an intermediate swapped after the walk pinned it — changes the
    /// identity and is rejected loudly.
    pub(crate) fn opened_is_path(f: &File, path: &Path) -> bool {
        let Some(opened) = identity_from_handle(f.as_raw_handle() as HANDLE) else {
            return false;
        };
        let Some(at_path) = open_path_no_follow(path) else {
            return false;
        };
        identity_from_handle(raw(&at_path)) == Some(opened)
    }

    #[cfg(all(test, windows))]
    type WalkSeam = Box<dyn Fn(&[u16]) + Send + 'static>;
    #[cfg(all(test, windows))]
    static WALK_SEAM: std::sync::OnceLock<std::sync::Mutex<Option<WalkSeam>>> =
        std::sync::OnceLock::new();

    #[cfg(all(test, windows))]
    pub(crate) fn install_walk_seam(hook: WalkSeam) {
        let m = WALK_SEAM.get_or_init(|| std::sync::Mutex::new(None));
        *m.lock().expect("walk seam poisoned") = Some(hook);
    }

    #[cfg(all(test, windows))]
    pub(crate) fn clear_walk_seam() {
        if let Some(lock) = WALK_SEAM.get() {
            *lock.lock().expect("walk seam poisoned") = None;
        }
    }

    #[cfg(all(test, windows))]
    fn walk_seam(comp: &[u16]) {
        if let Some(lock) = WALK_SEAM.get() {
            if let Some(hook) = lock.lock().expect("walk seam poisoned").as_ref() {
                hook(comp);
            }
        }
    }

    #[cfg(not(all(test, windows)))]
    fn walk_seam(_comp: &[u16]) {}
}

#[cfg(windows)]
pub(crate) use nt::{canonicalize_within, lexical_check, open_no_follow_walk, opened_is_path};

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::error::ErrorKind;

    fn u(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    fn comps(s: &str) -> Vec<Vec<u16>> {
        split_relative(&u(s)).expect("relative path must split")
    }

    fn text(c: &[u16]) -> String {
        String::from_utf16_lossy(c)
    }

    #[test]
    fn hop_bound_is_eight_like_unix() {
        assert_eq!(MAX_SYMLINK_HOPS, 8);
    }

    #[test]
    fn rejects_extended_prefix_and_device_namespace() {
        for evil in [r"\\?\C:\x", r"\\?\C:\ws\x", r"\??\C:\x"] {
            assert_eq!(
                split_relative(&u(evil)),
                Err(PathHazard::ExtendedPrefix),
                "{evil}"
            );
            assert_eq!(
                lexical_components(&u(r"C:\ws"), &u(evil)),
                Err(PathHazard::ExtendedPrefix),
                "{evil} must stay rejected even when in-root"
            );
        }
        for evil in [r"\\.\C:", r"\\.\PhysicalDrive0"] {
            assert_eq!(
                split_relative(&u(evil)),
                Err(PathHazard::DeviceNamespace),
                "{evil}"
            );
        }
    }

    #[test]
    fn rejects_unc_root_rooted_and_drive_relative() {
        assert_eq!(
            split_relative(&u(r"\\server\share\x")),
            Err(PathHazard::UncRoot)
        );
        assert_eq!(split_relative(&u(r"\x")), Err(PathHazard::Rooted));
        assert_eq!(split_relative(&u("/x")), Err(PathHazard::Rooted));
        assert_eq!(split_relative(&u("C:foo")), Err(PathHazard::DriveRelative));
        assert_eq!(split_relative(&u("C:")), Err(PathHazard::DriveRelative));
        assert_eq!(split_relative(&u(r"C:\x")), Err(PathHazard::Rooted));
        // An ESCAPE attempt via an absolute path outside the root.
        assert_eq!(
            lexical_components(&u(r"C:\ws"), &u(r"C:\Windows\system.ini")),
            Err(PathHazard::OutsideWorkspace)
        );
        assert_eq!(
            lexical_components(&u(r"C:\ws"), &u(r"\\evil\share\x")),
            Err(PathHazard::OutsideWorkspace)
        );
        // NT-native single-rooted paths are never verifiable.
        assert_eq!(
            lexical_components(&u(r"C:\ws"), &u(r"\Device\HarddiskVolume1\x")),
            Err(PathHazard::OutsideWorkspace)
        );
    }

    #[test]
    fn rejects_ads_nul_parent_and_separator() {
        assert_eq!(
            split_relative(&u("file.txt:stream")),
            Err(PathHazard::AlternateDataStream)
        );
        // `a:b` parses as a drive-relative path (drive A:, path b) on
        // Windows; either classification must deny.
        assert_eq!(split_relative(&u("a:b:c")), Err(PathHazard::DriveRelative));
        assert_eq!(
            split_relative(&u("dir/name:stream")),
            Err(PathHazard::AlternateDataStream)
        );
        assert_eq!(split_relative(&u("..")), Err(PathHazard::ParentDir));
        assert_eq!(split_relative(&u("a/../b")), Err(PathHazard::ParentDir));
        assert_eq!(split_relative(&u(r"a\..\b")), Err(PathHazard::ParentDir));
        let mut nul = u("a");
        nul.push(0);
        assert_eq!(split_relative(&nul), Err(PathHazard::Nul));
        assert_eq!(
            validate_component(&[u16::from(b'a'), u16::from(b'\\')]),
            Err(PathHazard::Separator)
        );
    }

    #[test]
    fn accepts_nested_paths_and_preserves_case() {
        let c = comps(r"a\b/c");
        assert_eq!(c.len(), 3);
        assert_eq!(text(&c[0]), "a");
        assert_eq!(text(&c[1]), "b");
        assert_eq!(text(&c[2]), "c");
        // Case is preserved verbatim (the walk opens the requested spelling).
        let c = comps("File.TXT");
        assert_eq!(c.len(), 1);
        assert_eq!(text(&c[0]), "File.TXT");
        // `.` segments vanish; empty rel resolves to the root.
        assert_eq!(comps(".").len(), 0);
        assert_eq!(comps("").len(), 0);
        assert_eq!(comps("./a/./b/").len(), 2);
        // Unicode names are fine.
        assert_eq!(text(&comps("rép😀/x")[0]), "rép😀");
        // An absolute path under the canonical root is accepted (unix parity).
        let c = lexical_components(&u(r"C:\ws"), &u(r"C:\ws\src\main.rs")).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(text(&c[1]), "main.rs");
        // A sibling directory whose name merely starts with the root name is
        // NOT under it (component-boundary check, not string prefix).
        assert_eq!(
            lexical_components(&u(r"C:\ws"), &u(r"C:\wsx\f.txt")),
            Err(PathHazard::OutsideWorkspace)
        );
        // Empty relative path resolves to the root itself.
        assert!(lexical_components(&u(r"C:\ws"), &[]).unwrap().is_empty());
    }

    #[test]
    fn case_policy_folds_ascii_only_and_is_component_wise() {
        assert!(units_eq_ci(&u(r"c:\WS"), &u(r"C:\ws")));
        assert!(units_eq_ci(&u("A/B"), &u("a/b")));
        // Non-ASCII case is conservatively NOT folded: over-strict denial,
        // never a silent cross-path match.
        assert!(!units_eq_ci(&u("straße"), &u("STRASSE")));
        // Absolute paths under the root match case-insensitively...
        assert_eq!(
            lexical_components(&u(r"C:\ws"), &u(r"c:\WS\SRC\x.rs"))
                .map(|c| c.len())
                .unwrap(),
            2
        );
        // ...but a drive change never matches.
        assert_eq!(
            lexical_components(&u(r"C:\ws"), &u(r"D:\ws\x")),
            Err(PathHazard::OutsideWorkspace)
        );
        // UNC roots compare server+share component-wise.
        assert_eq!(
            lexical_components(&u(r"\\?\UNC\Server\Share\ws"), &u(r"\\server\share\ws\f"))
                .map(|c| c.len())
                .unwrap(),
            1
        );
        assert_eq!(
            lexical_components(&u(r"\\?\UNC\Server\Share\ws"), &u(r"\\server\other\ws\f")),
            Err(PathHazard::OutsideWorkspace)
        );
    }

    #[test]
    fn hazard_error_mapping_is_typed() {
        let p = Path::new("x");
        assert_eq!(
            PathHazard::ExtendedPrefix.into_error(p).kind,
            ErrorKind::Permission
        );
        assert_eq!(
            PathHazard::DeviceNamespace.into_error(p).kind,
            ErrorKind::Permission
        );
        assert_eq!(
            PathHazard::UncRoot.into_error(p).kind,
            ErrorKind::Permission
        );
        assert_eq!(
            PathHazard::DriveRelative.into_error(p).kind,
            ErrorKind::Permission
        );
        assert_eq!(PathHazard::Rooted.into_error(p).kind, ErrorKind::Permission);
        assert_eq!(
            PathHazard::ParentDir.into_error(p).kind,
            ErrorKind::Permission
        );
        assert_eq!(
            PathHazard::AlternateDataStream.into_error(p).kind,
            ErrorKind::Permission
        );
        assert_eq!(PathHazard::Nul.into_error(p).kind, ErrorKind::Malformed);
        assert_eq!(
            PathHazard::Separator.into_error(p).kind,
            ErrorKind::Permission
        );
        assert_eq!(
            PathHazard::OutsideWorkspace.into_error(p).kind,
            ErrorKind::Permission
        );
        assert_eq!(
            PathHazard::TooManyComponents.into_error(p).kind,
            ErrorKind::Oversized
        );
    }

    // ------------------------------------------------- reparse parser

    fn reparse(tag: u32, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&(body.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        out.extend_from_slice(body);
        out
    }

    fn utf16_bytes(s: &str) -> Vec<u8> {
        s.encode_utf16().flat_map(|c| c.to_le_bytes()).collect()
    }

    fn symlink_body(sub: &str, print: &str, flags: u32) -> Vec<u8> {
        let sub_bytes = utf16_bytes(sub);
        let print_bytes = utf16_bytes(print);
        let mut b = Vec::new();
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(&(sub_bytes.len() as u16).to_le_bytes());
        b.extend_from_slice(&(sub_bytes.len() as u16).to_le_bytes());
        b.extend_from_slice(&(print_bytes.len() as u16).to_le_bytes());
        b.extend_from_slice(&flags.to_le_bytes());
        b.extend_from_slice(&sub_bytes);
        b.extend_from_slice(&print_bytes);
        b
    }

    fn mount_body(sub: &str, print: &str) -> Vec<u8> {
        let sub_bytes = utf16_bytes(sub);
        let print_bytes = utf16_bytes(print);
        let mut b = Vec::new();
        b.extend_from_slice(&0u16.to_le_bytes());
        b.extend_from_slice(&(sub_bytes.len() as u16).to_le_bytes());
        b.extend_from_slice(&(sub_bytes.len() as u16).to_le_bytes());
        b.extend_from_slice(&(print_bytes.len() as u16).to_le_bytes());
        b.extend_from_slice(&sub_bytes);
        b.extend_from_slice(&print_bytes);
        b
    }

    #[test]
    fn reparse_parser_extracts_symlink_and_mount_point_targets() {
        let data = reparse(
            IO_REPARSE_TAG_SYMLINK,
            &symlink_body(r"\??\C:\ws\real.txt", r"C:\ws\real.txt", 0),
        );
        let parsed = parse_reparse_data(&data).unwrap();
        assert_eq!(parsed.tag, IO_REPARSE_TAG_SYMLINK);
        assert_eq!(parsed.flags, 0);
        assert_eq!(text(&parsed.substitute), r"\??\C:\ws\real.txt");
        assert_eq!(text(&parsed.print), r"C:\ws\real.txt");

        let data = reparse(
            IO_REPARSE_TAG_SYMLINK,
            &symlink_body("other.txt", "other.txt", SYMLINK_FLAG_RELATIVE),
        );
        let parsed = parse_reparse_data(&data).unwrap();
        assert_eq!(parsed.flags & SYMLINK_FLAG_RELATIVE, 1);
        assert_eq!(text(&parsed.substitute), "other.txt");

        let data = reparse(
            IO_REPARSE_TAG_MOUNT_POINT,
            &mount_body(r"\??\C:\ws\target", r"C:\ws\target"),
        );
        let parsed = parse_reparse_data(&data).unwrap();
        assert_eq!(parsed.tag, IO_REPARSE_TAG_MOUNT_POINT);
        assert_eq!(parsed.flags, 0);
        assert_eq!(text(&parsed.substitute), r"\??\C:\ws\target");
    }

    #[test]
    fn reparse_parser_rejects_malformed_and_unknown() {
        assert_eq!(parse_reparse_data(&[]), Err(ReparseError::TooShort));
        assert_eq!(parse_reparse_data(&[0u8; 4]), Err(ReparseError::TooShort));
        // Declared body longer than the buffer.
        let mut truncated = reparse(IO_REPARSE_TAG_SYMLINK, &symlink_body("x", "x", 0));
        truncated.truncate(truncated.len() - 1);
        assert_eq!(parse_reparse_data(&truncated), Err(ReparseError::Truncated));
        // Odd substitute length.
        let mut body = symlink_body("ab", "", 0);
        body[2] = 3; // substitute length = 3 bytes
        assert_eq!(
            parse_reparse_data(&reparse(IO_REPARSE_TAG_SYMLINK, &body)),
            Err(ReparseError::OddLength)
        );
        // Out-of-bounds offset.
        let mut body = symlink_body("ab", "", 0);
        body[0] = 0xFF; // substitute offset far past the path buffer
        assert_eq!(
            parse_reparse_data(&reparse(IO_REPARSE_TAG_SYMLINK, &body)),
            Err(ReparseError::Truncated)
        );
        // Embedded NUL in the target.
        let mut body = symlink_body("a", "", 0);
        body[13] = 0; // second byte of the single UTF-16 unit
        body[12] = 0;
        assert_eq!(
            parse_reparse_data(&reparse(IO_REPARSE_TAG_SYMLINK, &body)),
            Err(ReparseError::NulInTarget)
        );
        // A cloud/placeholder tag is not whitelisted.
        let data = reparse(0x9000_001A, &[0u8; 12]);
        assert_eq!(parse_reparse_data(&data), Err(ReparseError::UnsupportedTag));
    }

    #[test]
    fn rebase_follows_only_verified_in_root_targets() {
        let root = u(r"C:\ws");
        let pos = vec![u("sub")];

        // Absolute target under the root (NT DOS namespace spelling).
        let rebase = rebase_target(&root, &pos, &u(r"\??\C:\ws\real"), false).unwrap();
        match rebase {
            Rebase::Relocate(c) => assert_eq!(c.into_iter().collect::<Vec<_>>(), vec![u("real")]),
            Rebase::Stay(_) => panic!("absolute target must relocate"),
        }
        // Absolute 8.3-ish path with `.` / `..` that stays in root.
        let rebase = rebase_target(&root, &pos, &u(r"C:\ws\a\..\b"), false).unwrap();
        match rebase {
            Rebase::Relocate(c) => assert_eq!(c.into_iter().collect::<Vec<_>>(), vec![u("b")]),
            Rebase::Stay(_) => panic!(),
        }
        // Absolute target climbing above the root: denied.
        assert_eq!(
            rebase_target(&root, &pos, &u(r"\??\C:\ws\..\outside"), false),
            Err(PathHazard::OutsideWorkspace)
        );
        assert_eq!(
            rebase_target(&root, &pos, &u(r"\??\C:\other\ws"), false),
            Err(PathHazard::OutsideWorkspace)
        );
        // NT-native absolute target: not verifiable.
        assert_eq!(
            rebase_target(&root, &pos, &u(r"\Device\HarddiskVolume1\x"), false),
            Err(PathHazard::OutsideWorkspace)
        );
        // Relative without `..`: continue from the link's parent.
        match rebase_target(&root, &pos, &u("real"), true).unwrap() {
            Rebase::Stay(c) => assert_eq!(c.into_iter().collect::<Vec<_>>(), vec![u("real")]),
            Rebase::Relocate(_) => panic!("plain relative target must stay"),
        }
        // Relative with `..` that stays in root: relocate from the root.
        match rebase_target(&root, &pos, &u(r"..\other"), true).unwrap() {
            Rebase::Relocate(c) => assert_eq!(c.into_iter().collect::<Vec<_>>(), vec![u("other")]),
            Rebase::Stay(_) => panic!(),
        }
        // Relative climbing above the root: denied.
        assert_eq!(
            rebase_target(&root, &pos, &u(r"..\..\escape"), true),
            Err(PathHazard::OutsideWorkspace)
        );
        // Flag/syntax disagreement is malformed on both sides.
        assert_eq!(
            rebase_target(&root, &pos, &u(r"\??\C:\ws\real"), true),
            Err(PathHazard::OutsideWorkspace)
        );
        assert_eq!(
            rebase_target(&root, &pos, &u("relative"), false),
            Err(PathHazard::OutsideWorkspace)
        );
        // Relative target with ADS is rejected component-wise.
        assert_eq!(
            rebase_target(&root, &pos, &u("dir/name:stream"), true),
            Err(PathHazard::AlternateDataStream)
        );
    }
}

// ======================================================================
// Windows runtime tests (cfg(windows)) — these require a Windows runner.
//
// Every adversarial scenario mirrors the unix wave-11 suite: a swap between
// two walk steps, a swap after a directory was pinned, an outside reparse
// target, a bounded loop, parent re-verification before a rename, and a
// destination reparse swap. Assertions are always "the original content, a
// safe replacement, or a loud typed rejection — never the attacker's file".
// ======================================================================

#[cfg(all(test, windows))]
mod win_tests {
    use super::nt::{clear_walk_seam, install_walk_seam};
    use crate::{WorkspaceFileService, WorkspaceHandle};
    use faktor_core::error::ErrorKind;
    use faktor_core::id::WorkspaceId;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, MutexGuard};

    fn fixture() -> (
        tempfile::TempDir,
        Arc<WorkspaceFileService>,
        WorkspaceHandle,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("ws");
        std::fs::create_dir_all(&root).unwrap();
        let service = WorkspaceFileService::new();
        let handle = service.open(WorkspaceId::new(1), root).unwrap();
        (dir, service, handle)
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    /// Seam tests share ONE global walk seam: run serially and always clear
    /// it (also on panic).
    struct SeamTest {
        _serial: MutexGuard<'static, ()>,
    }

    impl SeamTest {
        fn new() -> Self {
            static SERIAL: Mutex<()> = Mutex::new(());
            let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
            Self { _serial }
        }
    }

    impl Drop for SeamTest {
        fn drop(&mut self) {
            clear_walk_seam();
        }
    }

    /// Install a walk seam firing at most once, on the installing thread,
    /// only for the given component.
    fn swap_seam(root: &Path, target: &str, hook: impl Fn(&Path) + Send + 'static) {
        let me = std::thread::current().id();
        let fired = Arc::new(AtomicBool::new(false));
        let root = root.to_path_buf();
        let target = wide(target);
        install_walk_seam(Box::new(move |comp: &[u16]| {
            if std::thread::current().id() == me
                && comp == target.as_slice()
                && !fired.swap(true, Ordering::SeqCst)
            {
                hook(&root);
            }
        }));
    }

    /// Create a directory junction (IO_REPARSE_TAG_MOUNT_POINT). Junctions
    /// need no SeCreateSymbolicLinkPrivilege, so these tests always run.
    fn make_junction(link: &Path, target: &Path) {
        let out = std::process::Command::new("cmd")
            .arg("/C")
            .arg("mklink")
            .arg("/J")
            .arg(link)
            .arg(target)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "mklink /J {} -> {} failed: {}",
            link.display(),
            target.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// File symlink (IO_REPARSE_TAG_SYMLINK). Requires
    /// SeCreateSymbolicLinkPrivilege (admin or Developer Mode); returns false
    /// when the OS refuses so a non-admin dev box skips only this scenario.
    fn try_file_symlink(link: &Path, target: &Path) -> bool {
        match std::os::windows::fs::symlink_file(target, link) {
            Ok(()) => true,
            Err(e) if e.raw_os_error() == Some(1314) => {
                eprintln!("skipping: SeCreateSymbolicLinkPrivilege unavailable");
                false
            }
            Err(e) => panic!("symlink_file {}: {e}", link.display()),
        }
    }

    #[test]
    fn normal_nested_file_reads_writes_and_stats() {
        let (_d, _s, h) = fixture();
        let hash = h.write_atomic(Path::new("a/b/c.txt"), b"NESTED").unwrap();
        assert_eq!(
            hash,
            faktor_core::hash::FileHash::from(blake3::hash(b"NESTED").into())
        );
        let data = h.read(Path::new("a/b/c.txt"), 100).unwrap();
        assert_eq!(data.bytes, b"NESTED");
        let meta = h.stat(Path::new("a/b/c.txt")).unwrap();
        assert_eq!(meta.size, 6);
        // resolve_fd returns a real handle that reads the content.
        let fd = h.resolve_fd(Path::new("a/b/c.txt")).unwrap();
        let mut f = std::fs::File::from(fd);
        use std::io::Read;
        let mut buf = String::new();
        f.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "NESTED");
    }

    /// The failing Windows case in isolation: a plain nested path whose
    /// parent directories do not exist yet. `write_atomic` creates them, and
    /// both the resolution of the not-yet-existing tail and the handle walk
    /// over the created directories must round-trip.
    #[test]
    fn plain_nested_directory_round_trip() {
        let (_d, _s, h) = fixture();
        let root = h.root().to_path_buf();
        // No pre-created parents: resolve + write + read + stat + fd.
        let hash = h.write_atomic(Path::new("a/b/c.txt"), b"NESTED").unwrap();
        assert_eq!(
            hash,
            faktor_core::hash::FileHash::from(blake3::hash(b"NESTED").into())
        );
        assert_eq!(
            h.read(Path::new("a/b/c.txt"), 100).unwrap().bytes,
            b"NESTED"
        );
        assert_eq!(h.stat(Path::new("a/b/c.txt")).unwrap().size, 6);
        let fd = h.resolve_fd(Path::new("a/b/c.txt")).unwrap();
        let mut f = std::fs::File::from(fd);
        use std::io::Read;
        let mut buf = String::new();
        f.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "NESTED");
        // The nested directory itself resolves through the walk.
        let dir = h.resolve(Path::new("a/b")).unwrap();
        assert!(dir.starts_with(&root), "{dir:?}");
        assert!(dir.ends_with(Path::new("a").join("b")), "{dir:?}");
        // A missing nested destination still resolves under the root with
        // its tail preserved: the writers create it afterwards.
        let missing = h.resolve(Path::new("x/y/z.txt")).unwrap();
        assert!(missing.starts_with(&root), "{missing:?}");
        assert_eq!(
            missing.strip_prefix(&root).unwrap(),
            Path::new("x").join("y").join("z.txt")
        );
    }

    /// An in-root SYMLINK inside a subdirectory with a relative target.
    /// `std::os::windows::fs::symlink_file` stores a relative target
    /// verbatim — forward slashes included — and the Win32 parser cannot
    /// follow that substitute name; the handle walk reads it from the
    /// reparse buffer and normalizes both separators, so `read`/`resolve`
    /// must reach the in-root target.
    #[test]
    fn in_root_relative_symlink_in_subdirectory_is_followed() {
        let (_d, _s, h) = fixture();
        let root: PathBuf = h.root().to_path_buf();
        std::fs::create_dir_all(root.join("sub/deep")).unwrap();
        std::fs::write(root.join("sub/real.txt"), b"RELATIVE-TARGET").unwrap();
        std::fs::write(root.join("sub/deep/here.txt"), b"STAY-TARGET").unwrap();
        // Plain relative target (no `..`): resolves against the link's
        // parent directory handle.
        if try_file_symlink(&root.join("sub/deep/local-link.txt"), Path::new("here.txt")) {
            assert_eq!(
                h.read(Path::new("sub/deep/local-link.txt"), 100)
                    .unwrap()
                    .bytes,
                b"STAY-TARGET"
            );
        }
        // Relative target with `..` AND forward slashes: the
        // broken-for-Win32 spelling the Windows lane exposed.
        if try_file_symlink(&root.join("sub/deep/up-link.txt"), Path::new("../real.txt")) {
            let data = h.read(Path::new("sub/deep/up-link.txt"), 100).unwrap();
            assert_eq!(data.bytes, b"RELATIVE-TARGET");
            let resolved = h.resolve(Path::new("sub/deep/up-link.txt")).unwrap();
            assert_eq!(resolved, root.join("sub").join("real.txt"));
            let fd = h.resolve_fd(Path::new("sub/deep/up-link.txt")).unwrap();
            let mut f = std::fs::File::from(fd);
            use std::io::Read;
            let mut buf = String::new();
            f.read_to_string(&mut buf).unwrap();
            assert_eq!(buf, "RELATIVE-TARGET");
        }
        // Backslash spelling of the same relative target.
        if try_file_symlink(
            &root.join("sub/deep/bs-link.txt"),
            Path::new(r"..\real.txt"),
        ) {
            assert_eq!(
                h.read(Path::new("sub/deep/bs-link.txt"), 100)
                    .unwrap()
                    .bytes,
                b"RELATIVE-TARGET"
            );
        }
    }

    /// Walk failures carry the failing component index, the raw NTSTATUS and
    /// the accumulated resolved position so the Windows lane is diagnosable
    /// from the panic alone.
    #[test]
    fn walk_failure_diagnostics_name_component_and_status() {
        let (_d, _s, h) = fixture();
        std::fs::create_dir_all(h.root().join("a")).unwrap();
        let err = h.read(Path::new("a/absent/f.txt"), 100).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission, "{err:?}");
        assert!(err.message.contains("component 1"), "{err:?}");
        assert!(err.message.contains("NTSTATUS 0xC000003"), "{err:?}");
        assert!(err.message.contains(r#"resolved "\\a""#), "{err:?}");
        // A fully absent parent chain reports the first component under the
        // workspace root.
        let err = h.read(Path::new("gone/f.txt"), 100).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission, "{err:?}");
        assert!(err.message.contains("component 0"), "{err:?}");
        assert!(err.message.contains(r#"resolved "\\""#), "{err:?}");
    }

    #[test]
    fn in_root_junction_and_symlink_are_followed() {
        let (_d, _s, h) = fixture();
        let root: PathBuf = h.root().to_path_buf();
        std::fs::create_dir_all(root.join("real")).unwrap();
        std::fs::write(root.join("real/f.txt"), b"VIA-JUNCTION").unwrap();
        // Junction dir link (mount point tag).
        make_junction(&root.join("jlink"), &root.join("real"));
        assert_eq!(
            h.read(Path::new("jlink/f.txt"), 100).unwrap().bytes,
            b"VIA-JUNCTION"
        );
        // File symlink (symlink tag).
        if try_file_symlink(&root.join("slink.txt"), &root.join("real/f.txt")) {
            assert_eq!(
                h.read(Path::new("slink.txt"), 100).unwrap().bytes,
                b"VIA-JUNCTION"
            );
        }
        // Relative file symlink with a `..` that stays inside the root.
        if try_file_symlink(
            &root.join("real/nested-link.txt"),
            Path::new("../real/f.txt"),
        ) {
            assert_eq!(
                h.read(Path::new("real/nested-link.txt"), 100)
                    .unwrap()
                    .bytes,
                b"VIA-JUNCTION"
            );
        }
    }

    #[test]
    fn outside_junction_is_denied_and_never_read() {
        let (_d, _s, h) = fixture();
        let root: PathBuf = h.root().to_path_buf();
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("secret.txt"), b"OUTSIDE-SECRET").unwrap();
        make_junction(&root.join("evil"), outside.path());
        let err = h.read(Path::new("evil/secret.txt"), 100).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission, "{err:?}");
        assert_eq!(
            std::fs::read(outside.path().join("secret.txt")).unwrap(),
            b"OUTSIDE-SECRET",
            "the outside file must never be read through the workspace"
        );
    }

    #[test]
    fn escape_hazards_are_denied_before_any_open() {
        let (_d, _s, h) = fixture();
        std::fs::write(h.root().join("f.txt"), b"x").unwrap();
        let evil: &[&str] = &[
            "..",
            "../out.txt",
            r"\\?\C:\Windows\win.ini",
            r"\\.\C:",
            r"C:\Windows\win.ini",
            "f.txt:stream",
            "/rooted",
        ];
        for e in evil {
            let r = h.resolve_fd(Path::new(e));
            assert!(r.is_err(), "{e} must be denied");
        }
        // A plain relative nested path is not a hazard.
        std::fs::create_dir_all(h.root().join("a")).unwrap();
        std::fs::write(h.root().join("a/b.txt"), b"ok").unwrap();
        let ok = h.resolve_fd(Path::new(r"a\b.txt")).unwrap();
        drop(ok);
    }

    #[test]
    fn reparse_loop_fails_bounded() {
        let (_d, _s, h) = fixture();
        let root: PathBuf = h.root().to_path_buf();
        if !try_file_symlink(&root.join("loop.txt"), &root.join("loop.txt")) {
            return;
        }
        let err = h.resolve_fd(Path::new("loop.txt")).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission, "{err:?}");
        assert!(err.message.contains("hop bound"), "{err:?}");
    }

    /// (a1) Intermediate swapped for an OUTSIDE junction before the walk
    /// opens it: the walk sees the reparse point with
    /// FILE_OPEN_REPARSE_POINT, parses the target and denies the escape.
    #[test]
    fn walk_rejects_intermediate_swap_to_outside_reparse_before_open() {
        let _seam = SeamTest::new();
        let (_d, _s, h) = fixture();
        let root: PathBuf = h.root().to_path_buf();
        std::fs::create_dir_all(root.join("w/dir")).unwrap();
        std::fs::write(root.join("w/dir/f.txt"), b"INSIDE-ORIGINAL").unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(outside.path().join("dir")).unwrap();
        std::fs::write(outside.path().join("dir/f.txt"), b"OUTSIDE-SECRET").unwrap();
        let out = outside.path().to_path_buf();
        swap_seam(&root, "w", move |root| {
            std::fs::rename(root.join("w"), root.join("w-moved")).unwrap();
            make_junction(&root.join("w"), &out);
        });
        let err = h.read(Path::new("w/dir/f.txt"), 100).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission, "{err:?}");
        assert!(
            err.message.contains("workspace") || err.message.contains("reparse"),
            "{err:?}"
        );
        assert_eq!(
            std::fs::read(outside.path().join("dir/f.txt")).unwrap(),
            b"OUTSIDE-SECRET"
        );
        std::fs::remove_dir(root.join("w")).unwrap();
        std::fs::rename(root.join("w-moved"), root.join("w")).unwrap();
        assert_eq!(
            h.read(Path::new("w/dir/f.txt"), 100).unwrap().bytes,
            b"INSIDE-ORIGINAL"
        );
    }

    /// (a2) Swap AFTER the directory handle was pinned: the walk continues
    /// inside the ORIGINAL directory and the post-open identity net
    /// (volume serial + file id) rejects the read.
    #[test]
    fn walk_after_intermediate_swap_reads_pinned_dir_and_net_rejects() {
        let _seam = SeamTest::new();
        let (_d, _s, h) = fixture();
        let root: PathBuf = h.root().to_path_buf();
        std::fs::create_dir_all(root.join("w/dir")).unwrap();
        std::fs::write(root.join("w/dir/f.txt"), b"INSIDE-ORIGINAL").unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(outside.path().join("dir")).unwrap();
        std::fs::write(outside.path().join("dir/f.txt"), b"OUTSIDE-SECRET").unwrap();
        let out = outside.path().to_path_buf();
        swap_seam(&root, "dir", move |root| {
            std::fs::rename(root.join("w"), root.join("w-moved")).unwrap();
            make_junction(&root.join("w"), &out);
        });
        let err = h.read(Path::new("w/dir/f.txt"), 100).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission, "{err:?}");
        assert!(err.message.contains("TOCTOU"), "{err:?}");
        assert_eq!(
            std::fs::read(outside.path().join("dir/f.txt")).unwrap(),
            b"OUTSIDE-SECRET"
        );
        std::fs::remove_dir(root.join("w")).unwrap();
        std::fs::rename(root.join("w-moved"), root.join("w")).unwrap();
    }

    /// (a3) Same mid-walk swap, but the replacement junction points INSIDE
    /// the root: the handle walk still reads the pinned original.
    #[test]
    fn walk_after_inroot_swap_completes_with_original_content() {
        let _seam = SeamTest::new();
        let (_d, _s, h) = fixture();
        let root: PathBuf = h.root().to_path_buf();
        std::fs::create_dir_all(root.join("w/dir")).unwrap();
        std::fs::write(root.join("w/dir/f.txt"), b"INSIDE-ORIGINAL").unwrap();
        swap_seam(&root, "dir", move |root| {
            std::fs::rename(root.join("w"), root.join("w-moved")).unwrap();
            make_junction(&root.join("w"), &root.join("w-moved"));
        });
        let ok = h
            .read(Path::new("w/dir/f.txt"), 100)
            .expect("in-root swap after anchoring must not redirect");
        assert_eq!(ok.bytes, b"INSIDE-ORIGINAL");
        std::fs::remove_dir(root.join("w")).unwrap();
        std::fs::rename(root.join("w-moved"), root.join("w")).unwrap();
    }

    /// (c-write) Parent directory swapped for an outside junction between
    /// resolution and the rename: the pre-rename handle walk fails the write
    /// and nothing lands outside.
    #[test]
    fn write_refuses_when_parent_replaced_by_outside_junction() {
        let _seam = SeamTest::new();
        let (_d, _s, h) = fixture();
        let root: PathBuf = h.root().to_path_buf();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/f.txt"), b"BASE-CONTENT").unwrap();
        let base_hash = h
            .read(Path::new("sub/f.txt"), 100)
            .unwrap()
            .full_hash()
            .expect("small file read whole");
        let outside = tempfile::tempdir().unwrap();
        std::fs::write(outside.path().join("marker"), b"M").unwrap();
        let out = outside.path().to_path_buf();
        swap_seam(&root, "sub", move |root| {
            std::fs::rename(root.join("sub"), root.join("sub-moved")).unwrap();
            make_junction(&root.join("sub"), &out);
        });
        let err = h
            .write_atomic_cas(Path::new("sub/f.txt"), base_hash, b"EDITED")
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission, "{err:?}");
        let names: Vec<String> = std::fs::read_dir(outside.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(
            names,
            vec!["marker"],
            "temp/rename leaked outside: {names:?}"
        );
        std::fs::remove_dir(root.join("sub")).unwrap();
        std::fs::rename(root.join("sub-moved"), root.join("sub")).unwrap();
        // After restoring, the same write succeeds.
        let h2 = h
            .write_atomic_cas(Path::new("sub/f.txt"), base_hash, b"EDITED")
            .unwrap();
        let after = h.read(Path::new("sub/f.txt"), 100).unwrap();
        assert_eq!(after.bytes, b"EDITED");
        assert_eq!(after.full_hash(), Some(h2));
    }

    /// Destination entry swapped for an outside file symlink before the
    /// rename: the rename replaces the link itself; the outside victim is
    /// never written through.
    #[test]
    fn destination_reparse_swap_before_write_never_touches_target() {
        let _seam = SeamTest::new();
        let (_d, _s, h) = fixture();
        let root: PathBuf = h.root().to_path_buf();
        std::fs::create_dir_all(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/f.txt"), b"ORIGINAL").unwrap();
        let outside = tempfile::tempdir().unwrap();
        let victim = outside.path().join("victim.txt");
        std::fs::write(&victim, b"VICTIM").unwrap();
        let victim2 = victim.clone();
        swap_seam(&root, "sub", move |root| {
            std::fs::remove_file(root.join("sub/f.txt")).unwrap();
            if !try_file_symlink(&root.join("sub/f.txt"), &victim2) {
                // Without the symlink privilege the swap cannot be staged;
                // fall back to a rename-over (still a hostile swap).
                std::fs::write(root.join("sub/f.txt"), b"ATTACKER").unwrap();
            }
        });
        let result = h.write_atomic(Path::new("sub/f.txt"), b"REPLACED");
        // The victim is NEVER modified through the link either way.
        assert_eq!(std::fs::read(&victim).unwrap(), b"VICTIM");
        if let Ok(hash) = result {
            let after = h.read(Path::new("sub/f.txt"), 100).unwrap();
            assert_eq!(after.bytes, b"REPLACED");
            assert_eq!(after.full_hash(), Some(hash));
        } else {
            // A loud rejection is equally acceptable (link left in place).
            let meta = std::fs::symlink_metadata(root.join("sub/f.txt")).unwrap();
            assert!(meta.file_type().is_symlink() || meta.is_file());
        }
    }

    /// Case-fold policy, on disk: NTFS is case-insensitive by default, so a
    /// differently-cased spelling resolves to the same file (documented
    /// ambiguity — callers must not treat the spellings as distinct).
    #[test]
    fn case_fold_policy_resolves_to_the_same_file_on_default_volumes() {
        let (_d, _s, h) = fixture();
        h.write_atomic(Path::new("CaseFile.TXT"), b"CASE").unwrap();
        let data = h.read(Path::new("casefile.txt"), 100).unwrap();
        assert_eq!(data.bytes, b"CASE");
        assert_eq!(h.stat(Path::new("CASEFILE.txt")).unwrap().size, 4);
    }
}
