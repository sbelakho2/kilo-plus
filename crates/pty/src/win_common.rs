//! Pure, platform-independent helpers for the Windows ConPTY backend.
//!
//! This module intentionally depends on NOTHING windows-specific (no
//! windows-sys): every function takes plain values, so the whole module can
//! be compiled and adversarially tested on unix hosts. It is only included
//! in builds where something uses it — the Windows backend — plus the test
//! build, which is how the darwin (and any) host runs its tests.
//!
//! The win32 error codes below are frozen ABI values from winerror.h;
//! numeric literals keep the mapping testable without a windows crate.

use faktor_core::error::{Error, ErrorKind};

/// COORD dimensions are `i16` — a ConPTY size above this cannot be honored.
pub(crate) const COORD_MAX: u16 = i16::MAX as u16;

// winerror.h codes used for honest classification (frozen ABI values).
const ERROR_FILE_NOT_FOUND: u32 = 2;
const ERROR_PATH_NOT_FOUND: u32 = 3;
const ERROR_ACCESS_DENIED: u32 = 5;
const ERROR_INVALID_HANDLE: u32 = 6;
const ERROR_BAD_EXE_FORMAT: u32 = 193;
const ERROR_BROKEN_PIPE: u32 = 109;
const ERROR_NO_DATA: u32 = 232;
const ERROR_PIPE_NOT_CONNECTED: u32 = 233;
const ERROR_DIRECTORY: u32 = 267;
const ERROR_OPERATION_ABORTED: u32 = 995;

/// Map a raw `GetLastError()` code onto the honest error kind. Never
/// panics: any bogus or unknown code (0, 0xFFFFFFFF, fabricated values)
/// falls back to `Internal`.
pub(crate) fn classify_win32(code: u32) -> ErrorKind {
    match code {
        ERROR_FILE_NOT_FOUND | ERROR_PATH_NOT_FOUND | ERROR_DIRECTORY => ErrorKind::NotFound,
        ERROR_ACCESS_DENIED => ErrorKind::Permission,
        ERROR_BAD_EXE_FORMAT => ErrorKind::Malformed,
        _ => ErrorKind::Internal,
    }
}

/// Map a failed HRESULT onto the honest error kind. Win32-facility HRESULTs
/// (`HRESULT_FROM_WIN32`) are decoded into their win32 code first. Never
/// panics on any value, including bogus or even successful HRESULTs.
pub(crate) fn classify_hresult(hr: u32) -> ErrorKind {
    if hr >> 31 == 0 {
        // Success HRESULT reaching an error path is an internal bug, not a
        // config error — but it must never panic.
        return ErrorKind::Internal;
    }
    // FACILITY_WIN32 == 7: low 16 bits carry the win32 error code.
    if (hr >> 16) & 0x7fff == 7 {
        return classify_win32(hr & 0xffff);
    }
    match hr {
        0x80070057 => ErrorKind::Malformed,  // E_INVALIDARG
        0x80070005 => ErrorKind::Permission, // E_ACCESSDENIED
        _ => ErrorKind::Internal,            // E_FAIL, E_OUTOFMEMORY, ...
    }
}

/// Win32 codes that mean "the channel the ConPTY used is gone": a write or
/// read against a closed pseudoconsole session must surface as a typed
/// error, never a panic.
pub(crate) fn channel_closed(code: u32) -> bool {
    matches!(
        code,
        ERROR_BROKEN_PIPE
            | ERROR_NO_DATA
            | ERROR_PIPE_NOT_CONNECTED
            | ERROR_INVALID_HANDLE
            | ERROR_OPERATION_ABORTED
    )
}

/// Geometry bounds for anything that must fit a ConPTY `COORD`
/// (rows/cols are `i16`). Zero and >32767 are rejected: zero produces a
/// degenerate console and larger values would silently truncate.
pub(crate) fn validate_geometry(rows: u16, cols: u16) -> Result<(), Error> {
    if rows == 0 || cols == 0 {
        return Err(Error::malformed("pty size must be non-zero"));
    }
    if rows > COORD_MAX || cols > COORD_MAX {
        return Err(Error::oversized(format!(
            "pty size exceeds the ConPTY COORD range (max {COORD_MAX})"
        )));
    }
    Ok(())
}

/// Quote one argument so that `CommandLineToArgvW` (which both the CRT and
/// CreateProcessW children use to rebuild argv) returns it verbatim. Rules
/// are the documented Win32 convention: a literal quote preceded by `n`
/// backslashes is emitted as `2n+1` backslashes + quote, and a trailing run
/// of `n` backslashes (before the added closing quote) is doubled.
pub(crate) fn quote_cmdline_arg(arg: &str) -> String {
    let needs_quoting = arg.is_empty() || arg.chars().any(|c| matches!(c, ' ' | '\t' | '"'));
    if !needs_quoting {
        return arg.to_string();
    }
    let mut out = String::with_capacity(arg.len() + 2);
    out.push('"');
    let mut chars = arg.chars().peekable();
    loop {
        let mut backslashes = 0usize;
        while chars.peek() == Some(&'\\') {
            backslashes += 1;
            chars.next();
        }
        match chars.next() {
            None => {
                out.extend(std::iter::repeat_n('\\', backslashes * 2));
                break;
            }
            Some('"') => {
                out.extend(std::iter::repeat_n('\\', backslashes * 2 + 1));
                out.push('"');
            }
            Some(c) => {
                out.extend(std::iter::repeat_n('\\', backslashes));
                out.push(c);
            }
        }
    }
    out.push('"');
    out
}

/// Assemble the mutable command line passed to CreateProcessW with
/// `lpApplicationName = NULL`. The first token (the executable) is quoted
/// with the same rules so paths containing spaces resolve as one module
/// name.
pub(crate) fn build_command_line(command: &str, args: &[String]) -> String {
    let mut line = quote_cmdline_arg(command);
    for arg in args {
        line.push(' ');
        line.push_str(&quote_cmdline_arg(arg));
    }
    line
}

/// Merge explicit env overrides on top of the parent environment. Windows
/// environment names are matched case-insensitively, so an override
/// replaces the first entry that matches ignoring ASCII case, otherwise it
/// is appended. Pure and deterministic.
pub(crate) fn merge_env(
    parent: &[(String, String)],
    overrides: &[(String, String)],
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = parent.to_vec();
    for (key, value) in overrides {
        match out
            .iter_mut()
            .find(|(existing, _)| existing.eq_ignore_ascii_case(key))
        {
            Some(slot) => slot.1 = value.clone(),
            None => out.push((key.clone(), value.clone())),
        }
    }
    out
}

/// Build the double-NUL-terminated UTF-16 environment block CreateProcessW
/// expects: `KEY=VALUE\0` entries, sorted case-insensitively by key (the
/// documented block layout). Pure: works on any host, so it is tested here.
pub(crate) fn build_env_block(entries: &[(String, String)]) -> Vec<u16> {
    let mut sorted: Vec<&(String, String)> = entries.iter().collect();
    sorted.sort_by_key(|a| a.0.to_lowercase());
    let mut block = Vec::new();
    for (key, value) in sorted {
        block.extend(key.encode_utf16());
        block.push('=' as u16);
        block.extend(value.encode_utf16());
        block.push(0);
    }
    block.push(0);
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(r: Result<(), Error>) -> ErrorKind {
        r.unwrap_err().kind
    }

    #[test]
    fn win32_error_mapping_is_total_and_never_panics() {
        // Table-driven with fabricated codes: the mapping must be total.
        let cases = [
            (0u32, ErrorKind::Internal),
            (1, ErrorKind::Internal), // ERROR_INVALID_FUNCTION
            (2, ErrorKind::NotFound),
            (3, ErrorKind::NotFound),
            (5, ErrorKind::Permission),
            (6, ErrorKind::Internal),
            (7, ErrorKind::Internal),
            (87, ErrorKind::Internal),
            (109, ErrorKind::Internal),
            (193, ErrorKind::Malformed),
            (267, ErrorKind::NotFound),
            (0xDEAD_BEEF, ErrorKind::Internal),
            (0xFFFF_FFFF, ErrorKind::Internal),
        ];
        for (code, expected) in cases {
            assert_eq!(classify_win32(code), expected, "win32 code {code:#x}");
        }
    }

    #[test]
    fn hresult_mapping_decodes_win32_facility_and_never_panics() {
        let cases = [
            (0x80070002u32, ErrorKind::NotFound), // HRESULT_FROM_WIN32(2)
            (0x80070005, ErrorKind::Permission),  // HRESULT_FROM_WIN32(5)
            (0x80070057, ErrorKind::Internal),    // E_INVALIDARG (win32 87)
            (0x80000005, ErrorKind::Internal),    // E_FAIL
            (0x80004005, ErrorKind::Internal),    // E_FAIL
            (0x8007000E, ErrorKind::Internal),    // E_OUTOFMEMORY (win32 14)
            (0, ErrorKind::Internal),             // S_OK must never panic
            (1, ErrorKind::Internal),             // S_FALSE must never panic
            (0xDEAD_BEEF, ErrorKind::Internal),
            (0xFFFF_FFFF, ErrorKind::Internal),
            (0x800700C1, ErrorKind::Malformed), // HRESULT_FROM_WIN32(193)
        ];
        for (hr, expected) in cases {
            assert_eq!(classify_hresult(hr), expected, "hr {hr:#x}");
        }
    }

    #[test]
    fn closed_channel_codes_are_recognized_and_others_are_not() {
        for code in [6u32, 109, 232, 233, 995] {
            assert!(channel_closed(code), "{code} must classify as closed");
        }
        for code in [0u32, 1, 2, 5, 87, 0xFFFF_FFFF] {
            assert!(!channel_closed(code), "{code} must not classify as closed");
        }
    }

    #[test]
    fn geometry_validation_rejects_zero_and_over_coord_max() {
        assert_eq!(kind(validate_geometry(0, 80)), ErrorKind::Malformed);
        assert_eq!(kind(validate_geometry(24, 0)), ErrorKind::Malformed);
        assert_eq!(kind(validate_geometry(0, 0)), ErrorKind::Malformed);
        assert_eq!(kind(validate_geometry(40_000, 80)), ErrorKind::Oversized);
        assert_eq!(kind(validate_geometry(24, 40_000)), ErrorKind::Oversized);
        assert_eq!(kind(validate_geometry(65_535, 80)), ErrorKind::Oversized);
        assert!(validate_geometry(1, 1).is_ok());
        assert!(validate_geometry(COORD_MAX, COORD_MAX).is_ok());
        assert!(validate_geometry(24, 80).is_ok());
    }

    #[test]
    fn cmdline_quoting_matches_win32_conventions() {
        // Table of (arg, expected quoted form) derived from the documented
        // CommandLineToArgvW algorithm.
        let cases = [
            ("", "\"\""),
            ("plain", "plain"),
            ("has space", "\"has space\""),
            ("\ttabbed", "\"\ttabbed\""),
            ("quote\"inside", "\"quote\\\"inside\""),
            ("\"", "\"\\\"\""),
            // bare backslashes are literal when unquoted (no space/tab/quote)
            ("trailing\\", "trailing\\"),
            ("trailing\\\\", "trailing\\\\"),
            ("back\\slash", "back\\slash"),
            // once quoting is needed, trailing runs double before the close
            ("a b\\", "\"a b\\\\\""),
            ("a b\\\\", "\"a b\\\\\\\\\""),
            ("both\\\"quoted", "\"both\\\\\\\"quoted\""),
            ("C:\\Program Files\\", "\"C:\\Program Files\\\\\""),
            ("日本 語", "\"日本 語\""),
        ];
        for (arg, expected) in cases {
            assert_eq!(quote_cmdline_arg(arg), expected, "arg {arg:?}");
        }
    }

    #[test]
    fn command_line_joins_and_quotes_every_token() {
        let line = build_command_line("C:\\Program Files\\app.exe", &[]);
        assert_eq!(line, "\"C:\\Program Files\\app.exe\"");
        let line = build_command_line("sh", &["-c".into(), "echo \"hi\"".into()]);
        assert_eq!(line, "sh -c \"echo \\\"hi\\\"\"");
        let line = build_command_line("cmd.exe", &[String::new()]);
        assert_eq!(line, "cmd.exe \"\"");
    }

    #[test]
    fn env_merge_replaces_case_insensitively_and_appends_new_keys() {
        let parent = vec![
            ("PATH".to_string(), "/bin".to_string()),
            ("HOME".to_string(), "/root".to_string()),
        ];
        let merged = merge_env(&parent, &[("path".into(), "/usr/bin".into())]);
        assert_eq!(merged.len(), 2);
        assert!(merged.contains(&("PATH".to_string(), "/usr/bin".to_string())));
        let merged = merge_env(&parent, &[("NEW".into(), "1".into())]);
        assert_eq!(merged.len(), 3);
        assert!(merged.contains(&("NEW".to_string(), "1".to_string())));
        let merged = merge_env(&[], &[("A".into(), "1".into())]);
        assert_eq!(merged, vec![("A".to_string(), "1".to_string())]);
    }

    #[test]
    fn env_block_layout_is_sorted_and_double_nul_terminated() {
        let block = build_env_block(&[
            ("Z".to_string(), "1".to_string()),
            ("a".to_string(), "2".to_string()),
            ("middle".to_string(), "with=equals".to_string()),
        ]);
        // entries sorted case-insensitively: a, middle, Z
        let text = String::from_utf16(&block).unwrap();
        assert!(text.starts_with("a=2\u{0}middle=with=equals\u{0}Z=1\u{0}\u{0}"));
        assert!(block.ends_with(&[0, 0]));
        assert_eq!(block.len() % 2, 0);
    }

    #[test]
    fn fabricated_bogus_inputs_never_panic() {
        let _ = classify_win32(u32::MAX);
        let _ = classify_win32(0);
        let _ = classify_hresult(u32::MAX);
        let _ = classify_hresult(0x0000_0002);
        let _ = validate_geometry(0, 0);
        let _ = validate_geometry(COORD_MAX + 1, COORD_MAX + 1);
        let _ = quote_cmdline_arg(&"\\\"".repeat(10_000));
        let _ = build_env_block(&[("k".repeat(200_000), "v".repeat(200_000))]);
    }
}
