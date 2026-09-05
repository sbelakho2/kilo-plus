//! Platform-independent pre-spawn validation shared by the unix and Windows
//! `Pty::spawn` paths. Config is hostile input (it can arrive over the
//! wire): it is fully checked BEFORE any OS call, so misconfiguration errors
//! identically and loudly on every platform and never half-creates a pty.
//!
//! All bounds are deliberately shared: a config Windows could never honor
//! (NUL bytes, or a command line over the Win32 32,767-UTF-16-unit limit)
//! must not silently behave differently on unix.

use crate::PtyConfig;
use faktor_core::error::Error;

pub(crate) const MAX_COMMAND_LEN: usize = 32 * 1024;
pub(crate) const MAX_ARG_LEN: usize = 32 * 1024;
pub(crate) const MAX_ARGS: usize = 4096;
pub(crate) const MAX_ENV_ENTRIES: usize = 4096;
pub(crate) const MAX_ENV_ENTRY_CHARS: usize = 32 * 1024;
pub(crate) const MAX_CWD_CHARS: usize = 32 * 1024;
/// CreateProcessW refuses command lines longer than 32,767 UTF-16 units;
/// both platforms pay the same (generous) bound for config parity.
pub(crate) const MAX_COMMAND_LINE_UNITS: usize = 32_000;

/// Rejects configs that could never spawn on any backend. Pure: no I/O, no
/// panics on any input.
pub(crate) fn validate_spawn_config(cfg: &PtyConfig) -> Result<(), Error> {
    if cfg.command.is_empty() {
        return Err(Error::malformed("pty command must not be empty"));
    }
    if cfg.command.len() > MAX_COMMAND_LEN {
        return Err(Error::oversized(format!(
            "pty command exceeds {MAX_COMMAND_LEN} bytes"
        )));
    }
    if cfg.command.contains('\0') {
        return Err(Error::malformed("pty command contains a NUL byte"));
    }
    if cfg.args.len() > MAX_ARGS {
        return Err(Error::oversized(format!(
            "pty args exceed {MAX_ARGS} entries"
        )));
    }
    for arg in &cfg.args {
        if arg.len() > MAX_ARG_LEN {
            return Err(Error::oversized(format!(
                "pty arg exceeds {MAX_ARG_LEN} bytes"
            )));
        }
        if arg.contains('\0') {
            return Err(Error::malformed("pty arg contains a NUL byte"));
        }
    }
    if cfg.env.len() > MAX_ENV_ENTRIES {
        return Err(Error::oversized(format!(
            "pty env exceeds {MAX_ENV_ENTRIES} entries"
        )));
    }
    for (key, value) in &cfg.env {
        if key.contains('\0') || value.contains('\0') {
            return Err(Error::malformed(
                "pty env entry contains a NUL byte".to_string(),
            ));
        }
        if key.chars().count() + 1 + value.chars().count() > MAX_ENV_ENTRY_CHARS {
            return Err(Error::oversized(format!(
                "pty env entry exceeds {MAX_ENV_ENTRY_CHARS} chars"
            )));
        }
    }
    if let Some(cwd) = &cfg.cwd {
        if cwd.contains('\0') {
            return Err(Error::malformed("pty cwd contains a NUL byte"));
        }
        if cwd.chars().count() > MAX_CWD_CHARS {
            return Err(Error::oversized(format!(
                "pty cwd exceeds {MAX_CWD_CHARS} chars"
            )));
        }
    }
    let mut command_line_units = cfg.command.chars().count();
    for arg in &cfg.args {
        command_line_units += 1 + arg.chars().count();
    }
    if command_line_units > MAX_COMMAND_LINE_UNITS {
        return Err(Error::oversized(format!(
            "pty command line exceeds the {MAX_COMMAND_LINE_UNITS}-unit limit"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_core::error::ErrorKind;

    fn cfg() -> PtyConfig {
        PtyConfig {
            command: "sh".into(),
            ..Default::default()
        }
    }

    fn kind(r: Result<(), Error>) -> ErrorKind {
        r.unwrap_err().kind
    }

    #[test]
    fn empty_command_is_rejected() {
        assert_eq!(
            kind(validate_spawn_config(&PtyConfig::default())),
            ErrorKind::Malformed
        );
    }

    #[test]
    fn nul_bytes_anywhere_are_rejected() {
        let mut c = cfg();
        c.command = "sh\0x".into();
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Malformed);
        let mut c = cfg();
        c.args = vec!["ok".into(), "nul\0arg".into()];
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Malformed);
        let mut c = cfg();
        c.cwd = Some("/tmp/\0owned".into());
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Malformed);
        let mut c = cfg();
        c.env = vec![("K\0EY".into(), "v".into())];
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Malformed);
        let mut c = cfg();
        c.env = vec![("KEY".into(), "v\0alue".into())];
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Malformed);
    }

    #[test]
    fn oversized_fields_are_rejected_as_oversized() {
        let mut c = cfg();
        c.command = "x".repeat(MAX_COMMAND_LEN + 1);
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Oversized);
        let mut c = cfg();
        c.args = vec!["x".repeat(MAX_ARG_LEN + 1)];
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Oversized);
        let mut c = cfg();
        c.args = (0..=MAX_ARGS).map(|i| i.to_string()).collect();
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Oversized);
        let mut c = cfg();
        c.env = (0..=MAX_ENV_ENTRIES)
            .map(|i| (format!("K{i}"), "v".into()))
            .collect();
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Oversized);
        let mut c = cfg();
        c.env = vec![("K".into(), "v".repeat(MAX_ENV_ENTRY_CHARS + 1))];
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Oversized);
        let mut c = cfg();
        c.cwd = Some("d".repeat(MAX_CWD_CHARS + 1));
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Oversized);
        // command line that would overflow the Win32 32,767-unit budget
        let mut c = cfg();
        c.args = vec![
            "arg".repeat(MAX_COMMAND_LINE_UNITS / 2 + 1),
            "arg".repeat(MAX_COMMAND_LINE_UNITS / 2 + 1),
        ];
        assert_eq!(kind(validate_spawn_config(&c)), ErrorKind::Oversized);
    }

    #[test]
    fn sane_configs_pass() {
        let mut c = cfg();
        c.args = vec!["-c".into(), "echo ok".into()];
        c.cwd = Some("/tmp".into());
        c.env = vec![("PATH".into(), "/usr/bin".into())];
        assert!(validate_spawn_config(&c).is_ok());
        // max geometry is a windows concern (COORD), not a shared one
        let mut c = cfg();
        c.rows = 65_535;
        c.cols = 65_535;
        assert!(validate_spawn_config(&c).is_ok());
    }

    #[test]
    fn utf8_boundary_inputs_never_panic() {
        let mut c = cfg();
        c.command = "\u{1F600}".repeat(40000);
        assert!(validate_spawn_config(&c).is_err());
        let mut c = cfg();
        c.command = "日本語シェル".into();
        assert!(validate_spawn_config(&c).is_ok());
        let mut c = cfg();
        c.args = vec![String::new()];
        assert!(validate_spawn_config(&c).is_ok());
    }
}
