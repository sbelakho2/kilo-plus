//! One environment and command authority for every spawned child.
//!
//! [`EnvSpec`] is the only way a child environment is constructed: process
//! creation ALWAYS clears the inherited environment and applies the resolved
//! spec. Daemon secrets never leak implicitly — a spec either copies an
//! explicitly approved NAME from the daemon environment, or sets an exact
//! value. A deny-set of secret-shaped names (the daemon server password,
//! provider API keys, `*_SECRET`/`*_TOKEN`/`*PASSWORD*` names) is enforced on
//! every variant, so even an explicit entry cannot smuggle a configured
//! secret into a child.
//!
//! [`CommandSpec`] is the only way a command is expressed: generated
//! compiler/test/git commands are [`CommandSpec::Program`] (argv, never a
//! shell), and user/model snippets are [`CommandSpec::Shell`] with
//! [`ShellKind::PlatformDefault`] mapping to `/bin/sh` on unix and
//! `cmd.exe` on Windows (a configured platform shell overrides that).

use std::ffi::{OsStr, OsString};
use std::process::Command;

use crate::error::Error;

/// The universal child-process safety default: git must never wait for
/// interactive credential input. It is only added when the spec does not set
/// its own value.
pub const GIT_TERMINAL_PROMPT: &str = "GIT_TERMINAL_PROMPT";

/// Environment names copied from the daemon for ordinary tool children
/// (PATH resolution and the platform bits tools need). Nothing secret-shaped
/// is in this list, and the deny-set is applied on top.
pub const PLATFORM_ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "SystemRoot",
    "SYSTEMROOT",
    "PATHEXT",
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "TERM",
    "SHELL",
];

/// Environment names copied from the daemon for builds and user commands:
/// the platform baseline plus the approved toolchain homes. A verification
/// child sees exactly these names (or the subset a site documents); the
/// daemon's full environment is never passed.
pub const TOOLCHAIN_ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "SystemRoot",
    "SYSTEMROOT",
    "PATHEXT",
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "CARGO_TARGET_DIR",
    "RUSTFLAGS",
    "RUSTC_WRAPPER",
];

/// Exact secret names that are never forwarded, whatever the spec says.
const DENIED_ENV_EXACT: &[&str] = &["FAKTOR_SERVER_PASSWORD"];

/// Is this environment name in the secret deny-set? Fail closed on names
/// that are not valid UTF-8 (they cannot be audited, so they never pass).
pub fn env_name_is_denied(name: &OsStr) -> bool {
    let Some(name) = name.to_str() else {
        return true;
    };
    let upper = name.to_ascii_uppercase();
    DENIED_ENV_EXACT.contains(&upper.as_str())
        || upper.contains("API_KEY")
        || upper.contains("SECRET")
        || upper.contains("PASSWORD")
        || upper.ends_with("_TOKEN")
}

/// The child environment authority (audit P0-40).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EnvSpec {
    /// `env_clear` only (plus the universal `GIT_TERMINAL_PROMPT=0`
    /// safety default): the child sees nothing else.
    #[default]
    Minimal,
    /// `env_clear`, then copy exactly these daemon names when set. Names in
    /// the deny-set are never copied.
    Allowlisted(Vec<String>),
    /// `env_clear`, then these exact entries. An entry with an empty value
    /// means "copy the daemon's value for that key" (left unset when the
    /// daemon does not carry it). Later entries override earlier ones.
    Explicit(Vec<(OsString, OsString)>),
}

impl EnvSpec {
    /// The safe platform baseline (see [`PLATFORM_ENV_ALLOWLIST`]).
    pub fn default_baseline() -> EnvSpec {
        EnvSpec::Allowlisted(
            PLATFORM_ENV_ALLOWLIST
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        )
    }

    /// The verification/toolchain baseline (see
    /// [`TOOLCHAIN_ENV_ALLOWLIST`]): PATH for executable resolution, HOME
    /// for tool caches and credential-free config discovery, the approved
    /// Rust toolchain homes CARGO_HOME and RUSTUP_HOME, the optional
    /// CARGO_TARGET_DIR/RUSTFLAGS/RUSTC_WRAPPER knobs, and the documented
    /// platform bits (TMPDIR/TEMP/TMP, locale, terminal). The deny-set
    /// still removes secret-shaped names. This is the default environment
    /// for verification checks and generated toolchain commands.
    pub fn toolchain() -> EnvSpec {
        EnvSpec::Allowlisted(
            TOOLCHAIN_ENV_ALLOWLIST
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        )
    }

    /// The entries as DECLARED by the spec, before deny-set filtering and
    /// daemon-value copying: allowlisted names carry an empty placeholder
    /// value. Pre-spawn validation bound-checks this view without touching
    /// process state (the resolve step is where daemon values enter).
    pub fn declared(&self) -> Vec<(OsString, OsString)> {
        match self {
            EnvSpec::Minimal => Vec::new(),
            EnvSpec::Allowlisted(names) => names
                .iter()
                .map(|name| (name.into(), OsString::new()))
                .collect(),
            EnvSpec::Explicit(entries) => entries.clone(),
        }
    }

    /// The resolved child environment: deny-set filtered, deduplicated
    /// (later entries win), with the universal `GIT_TERMINAL_PROMPT=0`
    /// default when the spec does not set it.
    pub fn resolve(&self) -> Vec<(OsString, OsString)> {
        let mut out: Vec<(OsString, OsString)> = Vec::new();
        let mut put = |key: OsString, value: OsString| {
            if env_name_is_denied(&key) {
                return;
            }
            match out
                .iter_mut()
                .find(|(existing, _)| env_name_eq(existing, &key))
            {
                Some(slot) => slot.1 = value,
                None => out.push((key, value)),
            }
        };
        match self {
            EnvSpec::Minimal => {}
            EnvSpec::Allowlisted(names) => {
                for name in names {
                    if env_name_is_denied(OsStr::new(name)) {
                        continue;
                    }
                    if let Some(value) = std::env::var_os(name) {
                        put(name.into(), value);
                    }
                }
            }
            EnvSpec::Explicit(entries) => {
                for (key, value) in entries {
                    if value.is_empty() {
                        if let Some(current) = std::env::var_os(key) {
                            put(key.clone(), current);
                        }
                    } else {
                        put(key.clone(), value.clone());
                    }
                }
            }
        }
        if !out
            .iter()
            .any(|(key, _)| env_name_eq(key, OsStr::new(GIT_TERMINAL_PROMPT)))
        {
            out.push((GIT_TERMINAL_PROMPT.into(), "0".into()));
        }
        out
    }

    /// Apply the spec to a `Command`: the inherited environment is ALWAYS
    /// cleared first, then the resolved environment is set.
    pub fn apply(&self, cmd: &mut Command) {
        cmd.env_clear();
        for (key, value) in self.resolve() {
            cmd.env(key, value);
        }
    }
}

fn env_name_eq(a: &OsStr, b: &OsStr) -> bool {
    match (a.to_str(), b.to_str()) {
        (Some(a), Some(b)) => a.eq_ignore_ascii_case(b),
        _ => a == b,
    }
}

/// How a shell command selects its interpreter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShellKind {
    /// `/bin/sh` on unix, `cmd.exe` on Windows — never Git Bash. A
    /// configured platform shell overrides the choice.
    PlatformDefault,
    /// POSIX `sh -c` (`/bin/sh` on unix; `sh` via PATH elsewhere).
    PosixSh,
    /// Windows `cmd.exe /d /s /c`.
    Cmd,
    /// `powershell.exe -NoProfile -NonInteractive -Command`. Scripts are
    /// prefixed with [`POWERSHELL_UTF8_PRELUDE`] (the terminal-wide UTF-8
    /// stdout convention).
    PowerShell,
}

/// The typed command form: argv (`Program`) or a shell snippet (`Shell`).
/// Generated compiler/test/git commands are always `Program`; only user or
/// model snippets use `Shell`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommandSpec {
    Program {
        executable: OsString,
        args: Vec<OsString>,
    },
    Shell {
        script: String,
        shell: ShellKind,
    },
}

/// Bound on a shell snippet before any process exists.
pub const COMMAND_SCRIPT_MAX_BYTES: usize = 512 * 1024;

/// The terminal-wide PowerShell convention: every [`ShellKind::PowerShell`]
/// script is prefixed with this prelude. A redirected Windows PowerShell
/// (5.1) stdout defaults to the OEM codepage (437), so non-ASCII output is
/// mangled (`日本語` becomes `???`) or fails to write at all; pinning both
/// `[Console]::OutputEncoding` (the stdout writer) and `$OutputEncoding`
/// (native-command piping) to UTF-8 makes the captured bytes decode as
/// UTF-8 deterministically on Windows PowerShell 5.1 and PowerShell 7. The
/// prefixed script still rides as ONE `-Command` argv element.
pub const POWERSHELL_UTF8_PRELUDE: &str =
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; $OutputEncoding=[System.Text.Encoding]::UTF8; ";

/// Reserved prefix of a materialized cmd script (see
/// [`materialize_cmd_script`]). The process supervisor deletes only files
/// carrying this prefix under the system temp dir, so a caller-supplied
/// `cmd /c <script>` is never touched.
pub const CMD_SCRIPT_PREFIX: &str = "faktor-cmd-";

/// Abandoned materialized cmd scripts (lowering created the file but the
/// supervisor never got to delete it — a crash in between) are swept after
/// this age, so the system temp dir stays bounded. The age is far past any
/// plausible supervised deadline, so the sweep can never race a LIVE cmd
/// still reading its batch file.
const CMD_SCRIPT_TTL: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// Sweep every Nth materialization: the sweep is opportunistic, never a
/// per-command cost.
const CMD_SCRIPT_SWEEP_EVERY: u64 = 64;

static CMD_SCRIPT_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Materialize a shell script that must reach cmd.exe VERBATIM: cmd's `/C`
/// quote handling strips and re-parses embedded quotes when the snippet is
/// passed on the command line, so the snippet is written to a unique
/// `faktor-cmd-*.cmd` file in the system temp dir and cmd is handed the
/// path (`cmd.exe /D /C <path>`; std quotes the path when it contains
/// spaces, and with `/s` absent cmd preserves that single quoted executable
/// name — `cmd /?` rule 1). The file is a direct-use temp: no rename and no
/// publication step, so it is NOT an atomic-write sequence. The process
/// supervisor removes it as soon as the child exits; a file abandoned by a
/// crash is swept past [`CMD_SCRIPT_TTL`].
fn materialize_cmd_script(script: &str) -> Result<OsString, Error> {
    let dir = std::env::temp_dir();
    let seq = CMD_SCRIPT_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if seq.is_multiple_of(CMD_SCRIPT_SWEEP_EVERY) {
        sweep_abandoned_cmd_scripts(&dir);
    }
    let name = format!(
        "{CMD_SCRIPT_PREFIX}{}-{}-{seq}.cmd",
        std::process::id(),
        uuid::Uuid::new_v4()
    );
    let script_path = dir.join(name);
    std::fs::write(&script_path, script)
        .map_err(|e| Error::internal(format!("materialize cmd script: {e}")))?;
    Ok(script_path.into_os_string())
}

fn sweep_abandoned_cmd_scripts(dir: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with(CMD_SCRIPT_PREFIX) || !name.ends_with(".cmd") {
            continue;
        }
        let abandoned = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > CMD_SCRIPT_TTL);
        if abandoned {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// The argv of a materialized cmd script run: `cmd.exe /d /c <path>`. `/d`
/// skips AutoRun and `/c` runs the script; without `/s` cmd keeps a single
/// quoted path argument intact (`cmd /?` rule 1), which is exactly how std
/// hands over a temp path containing spaces. The path is handed over as a
/// BARE `OsString` argument — never pre-quoted here: the process-spawn layer
/// applies the MSVCRT argument-quoting rules (a quote only when the path
/// contains whitespace or a quote), and cmd's rule 1 then runs the single
/// quoted token as the script. A literal quote added here would be escaped
/// by the spawn layer and break the path.
fn cmd_script_argv(script: &str) -> Result<(OsString, Vec<OsString>), Error> {
    let script_path = materialize_cmd_script(script)?;
    Ok((
        OsString::from("cmd.exe"),
        vec![OsString::from("/d"), OsString::from("/c"), script_path],
    ))
}

/// A lowered command: the exact program + argv handed to the spawn layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCommand {
    pub program: OsString,
    pub args: Vec<OsString>,
}

impl CommandSpec {
    pub fn program(
        executable: impl Into<OsString>,
        args: impl IntoIterator<Item = impl Into<OsString>>,
    ) -> Self {
        CommandSpec::Program {
            executable: executable.into(),
            args: args.into_iter().map(Into::into).collect(),
        }
    }

    pub fn shell(script: impl Into<String>, shell: ShellKind) -> Self {
        CommandSpec::Shell {
            script: script.into(),
            shell,
        }
    }

    pub fn lower(&self) -> Result<ResolvedCommand, Error> {
        self.lower_with(None)
    }

    /// Lower to program + argv. `configured_platform_shell` overrides the
    /// [`ShellKind::PlatformDefault`] executable only; every other kind is
    /// fixed. Hostile input (NUL, empty, oversized) is refused typed before
    /// any process exists. A cmd shell snippet is materialized to a unique
    /// temp `.cmd` file and lowered to `cmd.exe /d /c <path>` (see
    /// [`materialize_cmd_script`]); PowerShell snippets stay one argv
    /// element and carry [`POWERSHELL_UTF8_PRELUDE`]; PosixSh snippets stay
    /// one argv element.
    pub fn lower_with(
        &self,
        configured_platform_shell: Option<&OsStr>,
    ) -> Result<ResolvedCommand, Error> {
        match self {
            CommandSpec::Program { executable, args } => {
                if executable.is_empty() {
                    return Err(Error::malformed("command executable is empty"));
                }
                if executable.to_string_lossy().contains('\0') {
                    return Err(Error::malformed("command executable contains NUL"));
                }
                for arg in args {
                    if arg.to_string_lossy().contains('\0') {
                        return Err(Error::malformed("command argument contains NUL"));
                    }
                }
                Ok(ResolvedCommand {
                    program: executable.clone(),
                    args: args.clone(),
                })
            }
            CommandSpec::Shell { script, shell } => {
                if script.is_empty() {
                    return Err(Error::malformed("shell script is empty"));
                }
                if script.len() > COMMAND_SCRIPT_MAX_BYTES {
                    return Err(Error::oversized(format!(
                        "shell script exceeds {COMMAND_SCRIPT_MAX_BYTES} bytes"
                    )));
                }
                if script.contains('\0') {
                    return Err(Error::malformed("shell script contains NUL"));
                }
                // A cmd snippet must reach cmd.exe VERBATIM: cmd's `/C`
                // quote handling mangles embedded quotes riding on the
                // command line, so the snippet is materialized to a temp
                // `.cmd` file and only the path is passed. Every other shell
                // keeps the one-argv-element snippet.
                let materialized = matches!(shell, ShellKind::Cmd)
                    || (matches!(shell, ShellKind::PlatformDefault)
                        && !cfg!(unix)
                        && configured_platform_shell.is_none());
                let (program, mut args): (OsString, Vec<OsString>) = match shell {
                    ShellKind::PlatformDefault => {
                        #[cfg(unix)]
                        {
                            platform_default_shell(configured_platform_shell)
                        }
                        #[cfg(not(unix))]
                        {
                            if configured_platform_shell.is_none() {
                                cmd_script_argv(script)?
                            } else {
                                platform_default_shell(configured_platform_shell)
                            }
                        }
                    }
                    ShellKind::PosixSh => {
                        #[cfg(unix)]
                        {
                            ("/bin/sh".into(), vec!["-c".into()])
                        }
                        #[cfg(not(unix))]
                        {
                            ("sh".into(), vec!["-c".into()])
                        }
                    }
                    ShellKind::Cmd => cmd_script_argv(script)?,
                    ShellKind::PowerShell => (
                        "powershell.exe".into(),
                        vec![
                            "-NoProfile".into(),
                            "-NonInteractive".into(),
                            "-Command".into(),
                        ],
                    ),
                };
                if let Some(configured) = configured_platform_shell {
                    if matches!(shell, ShellKind::PlatformDefault) {
                        if configured.is_empty() {
                            return Err(Error::malformed("configured platform shell is empty"));
                        }
                        if configured.to_string_lossy().contains('\0') {
                            return Err(Error::malformed("configured platform shell contains NUL"));
                        }
                    }
                }
                if !materialized {
                    // PowerShell scripts carry the terminal-wide UTF-8
                    // stdout prelude (see [`POWERSHELL_UTF8_PRELUDE`]).
                    let script = if matches!(shell, ShellKind::PowerShell) {
                        format!("{POWERSHELL_UTF8_PRELUDE}{script}")
                    } else {
                        script.clone()
                    };
                    args.push(script.into());
                }
                Ok(ResolvedCommand { program, args })
            }
        }
    }
}

#[cfg(unix)]
fn platform_default_shell(configured: Option<&OsStr>) -> (OsString, Vec<OsString>) {
    (
        configured
            .map(|s| s.to_os_string())
            .unwrap_or_else(|| OsString::from("/bin/sh")),
        vec!["-c".into()],
    )
}

#[cfg(not(unix))]
fn platform_default_shell(configured: Option<&OsStr>) -> (OsString, Vec<OsString>) {
    (
        configured
            .map(|s| s.to_os_string())
            .unwrap_or_else(|| OsString::from("cmd.exe")),
        vec!["/d".into(), "/s".into(), "/c".into()],
    )
}

/// What a sandbox policy requires of the spawn layer's network isolation
/// (the sandbox DECIDES; the terminal ENFORCES). Platform enforcement is
/// never pre-judged by policy code: a `DenyAll` spawn either runs isolated
/// or fails closed typed at the spawn layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkIsolationRequirement {
    DenyAll,
    Inherit,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimal_is_empty_but_keeps_the_git_prompt_default() {
        let resolved = EnvSpec::Minimal.resolve();
        assert_eq!(
            resolved,
            vec![(GIT_TERMINAL_PROMPT.into(), "0".into())],
            "minimal is env_clear + the one safety default"
        );
        assert!(EnvSpec::default() == EnvSpec::Minimal);
    }

    #[test]
    fn allowlisted_copies_only_named_daemon_values() {
        std::env::set_var("KP_ENV_TEST_VISIBLE", "visible");
        std::env::set_var("KP_ENV_TEST_SECRET_TOKEN", "never");
        let spec = EnvSpec::Allowlisted(vec![
            "KP_ENV_TEST_VISIBLE".into(),
            "KP_ENV_TEST_SECRET_TOKEN".into(),
        ]);
        let resolved = spec.resolve();
        assert!(resolved
            .iter()
            .any(|(k, v)| k == "KP_ENV_TEST_VISIBLE" && v == "visible"));
        assert!(
            !resolved
                .iter()
                .any(|(k, _)| k == "KP_ENV_TEST_SECRET_TOKEN"),
            "the deny-set drops secret-shaped names even when allowlisted: {resolved:?}"
        );
        std::env::remove_var("KP_ENV_TEST_VISIBLE");
        std::env::remove_var("KP_ENV_TEST_SECRET_TOKEN");
    }

    #[test]
    fn explicit_empty_value_copies_and_later_entries_win() {
        std::env::set_var("KP_ENV_TEST_COPY", "from-daemon");
        let spec = EnvSpec::Explicit(vec![
            ("KP_ENV_TEST_COPY".into(), OsString::new()),
            ("KP_ENV_TEST_EXACT".into(), "exact".into()),
            ("KP_ENV_TEST_COPY".into(), "override".into()),
        ]);
        let resolved = spec.resolve();
        assert!(resolved
            .iter()
            .any(|(k, v)| k == "KP_ENV_TEST_COPY" && v == "override"));
        assert!(resolved
            .iter()
            .any(|(k, v)| k == "KP_ENV_TEST_EXACT" && v == "exact"));
        std::env::remove_var("KP_ENV_TEST_COPY");
    }

    #[test]
    fn deny_set_blocks_server_password_provider_keys_and_private_secrets() {
        for name in [
            "FAKTOR_SERVER_PASSWORD",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "GITHUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "FAKTOR_PRIVATE_SECRET",
            "TEST_PRIVATE_SECRET",
            "DB_PASSWORD",
        ] {
            assert!(env_name_is_denied(OsStr::new(name)), "{name}");
        }
        for name in ["PATH", "HOME", "CARGO_HOME", "RUSTUP_HOME", "LANG"] {
            assert!(!env_name_is_denied(OsStr::new(name)), "{name}");
        }
        let spec = EnvSpec::Explicit(vec![
            ("FAKTOR_SERVER_PASSWORD".into(), "leak".into()),
            ("OPENAI_API_KEY".into(), "leak".into()),
        ]);
        assert_eq!(
            spec.resolve(),
            vec![(GIT_TERMINAL_PROMPT.into(), "0".into())],
            "even explicit entries cannot smuggle denied names"
        );
    }

    #[test]
    fn toolchain_baseline_documents_and_carries_the_approved_vars() {
        // The verification default: PATH + HOME + the approved toolchain
        // homes. The allowlist is the contract, so assert it directly.
        for name in [
            "PATH",
            "HOME",
            "CARGO_HOME",
            "RUSTUP_HOME",
            "CARGO_TARGET_DIR",
        ] {
            assert!(
                TOOLCHAIN_ENV_ALLOWLIST.contains(&name),
                "{name} must be in the documented toolchain allowlist"
            );
        }
        let EnvSpec::Allowlisted(names) = EnvSpec::toolchain() else {
            panic!("toolchain() must be an allowlist, never inherit-all");
        };
        assert!(names.iter().any(|n| n == "PATH"));
        assert!(names.iter().any(|n| n == "CARGO_HOME"));
        assert!(names.iter().any(|n| n == "RUSTUP_HOME"));
        // Declared entries expose names for pre-spawn validation, and the
        // deny-set applies to the resolved form exactly like every spec.
        let declared = EnvSpec::toolchain().declared();
        assert!(declared
            .iter()
            .any(|(k, _)| k.as_os_str() == OsStr::new("CARGO_HOME")));
        let mut declared = EnvSpec::Explicit(vec![("KEY".into(), "v\0".into())]).declared();
        declared.retain(|(k, _)| k.as_os_str() != OsStr::new("nope"));
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].0.as_os_str(), OsStr::new("KEY"));
    }

    #[test]
    fn platform_default_shell_selection_and_override() {
        let resolved = CommandSpec::shell("echo hi", ShellKind::PlatformDefault)
            .lower()
            .unwrap();
        #[cfg(unix)]
        assert_eq!(resolved.program, OsString::from("/bin/sh"));
        #[cfg(not(unix))]
        assert_eq!(resolved.program, OsString::from("cmd.exe"));
        let args: Vec<String> = resolved
            .args
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        #[cfg(unix)]
        assert_eq!(args.last().unwrap(), "echo hi");
        #[cfg(not(unix))]
        {
            let script = std::path::PathBuf::from(&resolved.args[2]);
            assert!(
                script
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with(CMD_SCRIPT_PREFIX)),
                "platform-default cmd scripts are materialized: {args:?}"
            );
            assert_eq!(std::fs::read_to_string(&script).unwrap(), "echo hi");
            let _ = std::fs::remove_file(script);
        }

        let configured = OsString::from("/bin/dash");
        let resolved = CommandSpec::shell("echo hi", ShellKind::PlatformDefault)
            .lower_with(Some(&configured))
            .unwrap();
        assert_eq!(resolved.program, configured);
    }

    #[cfg(unix)]
    #[test]
    fn pure_resolver_platform_default_is_bin_sh_never_git_bash() {
        // Pure resolver test (runs on darwin): the platform default is
        // exactly `/bin/sh -c`, never a bash/login shell and never PATH
        // lookup. PosixSh agrees on unix.
        for kind in [ShellKind::PlatformDefault, ShellKind::PosixSh] {
            let resolved = CommandSpec::shell("echo hi", kind).lower().unwrap();
            assert_eq!(resolved.program, OsString::from("/bin/sh"), "{kind:?}");
            assert_eq!(
                resolved.args,
                vec![OsString::from("-c"), OsString::from("echo hi")]
            );
            assert!(
                !resolved.program.to_string_lossy().contains("bash"),
                "{kind:?} must never resolve to Git Bash"
            );
        }
        // Hostile snippets stay one argv element, exactly as typed.
        let resolved = CommandSpec::shell("echo 'a b'; rm -rf /", ShellKind::PlatformDefault)
            .lower()
            .unwrap();
        assert_eq!(resolved.args.len(), 2);
        assert_eq!(resolved.args[1], OsString::from("echo 'a b'; rm -rf /"));
    }

    #[test]
    fn cmd_shell_snippets_are_materialized_never_embedded_on_the_command_line() {
        // `echo "a b"` through a raw `cmd.exe /C <script>` command line is
        // mangled by cmd's quote stripping (2026-09 Windows failure: exit 1,
        // empty stdout). The lowering must hand cmd a materialized script
        // file instead, with the raw snippet absent from the command line.
        let script = "echo \"a b\"";
        let resolved = CommandSpec::shell(script, ShellKind::Cmd).lower().unwrap();
        assert_eq!(resolved.program, OsString::from("cmd.exe"));
        assert_eq!(resolved.args[0], OsString::from("/d"));
        assert_eq!(resolved.args[1], OsString::from("/c"));
        assert_eq!(resolved.args.len(), 3);
        let path = std::path::PathBuf::from(&resolved.args[2]);
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        assert!(
            name.starts_with(CMD_SCRIPT_PREFIX) && name.ends_with(".cmd"),
            "cmd scripts carry the reserved materialized name: {name}"
        );
        assert_eq!(path.parent(), Some(std::env::temp_dir().as_path()));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), script);
        assert!(
            !resolved
                .args
                .iter()
                .any(|a| a.to_string_lossy().contains(script)),
            "the raw script must never ride on the cmd command line: {:?}",
            resolved.args
        );
        // Every lowering materializes its own unique script file.
        let second = CommandSpec::shell(script, ShellKind::Cmd).lower().unwrap();
        assert_ne!(resolved.args[2], second.args[2]);
        // PowerShell stays a single -Command argv element, carrying the
        // terminal-wide UTF-8 prelude before the verbatim script.
        let ps = CommandSpec::shell(script, ShellKind::PowerShell)
            .lower()
            .unwrap();
        assert_eq!(ps.args.len(), 4);
        assert_eq!(
            ps.args[3],
            OsString::from(format!("{POWERSHELL_UTF8_PRELUDE}{script}"))
        );
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(std::path::PathBuf::from(&second.args[2]));
    }

    #[test]
    fn explicit_shell_kinds_select_fixed_interpreters() {
        let cmd = CommandSpec::shell("dir", ShellKind::Cmd).lower().unwrap();
        assert_eq!(cmd.program, OsString::from("cmd.exe"));
        assert_eq!(cmd.args[0], OsString::from("/d"));
        assert_eq!(cmd.args[1], OsString::from("/c"));
        let _ = std::fs::remove_file(std::path::PathBuf::from(&cmd.args[2]));
        let ps = CommandSpec::shell("Get-Date", ShellKind::PowerShell)
            .lower()
            .unwrap();
        assert_eq!(ps.program, OsString::from("powershell.exe"));
        assert_eq!(ps.args[1], OsString::from("-NonInteractive"));
        assert_eq!(
            ps.args.last().unwrap(),
            &OsString::from(format!("{POWERSHELL_UTF8_PRELUDE}Get-Date"))
        );
        #[cfg(unix)]
        {
            let sh = CommandSpec::shell("true", ShellKind::PosixSh)
                .lower()
                .unwrap();
            assert_eq!(sh.program, OsString::from("/bin/sh"));
        }
    }

    #[test]
    fn powershell_scripts_carry_the_utf8_stdout_prelude_convention() {
        // The terminal-wide PowerShell convention, frozen here so the
        // invariant is testable on every host: fixed flags, one -Command
        // argv element, and the exact UTF-8 prelude before the verbatim
        // script (a redirected 5.1 stdout defaults to OEM 437 and would
        // mangle non-ASCII output).
        assert_eq!(
            POWERSHELL_UTF8_PRELUDE,
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; \
             $OutputEncoding=[System.Text.Encoding]::UTF8; "
        );
        let script = "Write-Output '日本語'";
        let ps = CommandSpec::shell(script, ShellKind::PowerShell)
            .lower()
            .unwrap();
        assert_eq!(ps.program, OsString::from("powershell.exe"));
        assert_eq!(
            ps.args,
            vec![
                OsString::from("-NoProfile"),
                OsString::from("-NonInteractive"),
                OsString::from("-Command"),
                OsString::from(format!("{POWERSHELL_UTF8_PRELUDE}{script}")),
            ]
        );
        assert!(
            ps.args[3]
                .to_string_lossy()
                .ends_with("Write-Output '日本語'"),
            "the script itself must stay verbatim after the prelude: {:?}",
            ps.args[3]
        );
    }

    #[test]
    fn program_commands_are_never_shell_wrapped() {
        let spec = CommandSpec::program("cargo", ["test", "--package", "a; rm -rf /"]);
        let resolved = spec.lower().unwrap();
        assert_eq!(resolved.program, OsString::from("cargo"));
        assert_eq!(resolved.args.len(), 3);
        assert_eq!(resolved.args[2], OsString::from("a; rm -rf /"));
    }

    #[test]
    fn hostile_inputs_are_typed_and_never_panic() {
        assert!(CommandSpec::shell("", ShellKind::PlatformDefault)
            .lower()
            .is_err());
        assert!(
            CommandSpec::shell("echo\0owned", ShellKind::PlatformDefault)
                .lower()
                .is_err()
        );
        assert!(
            CommandSpec::shell("x".repeat(COMMAND_SCRIPT_MAX_BYTES + 1), ShellKind::PosixSh)
                .lower()
                .is_err()
        );
        assert!(CommandSpec::program("", Vec::<OsString>::new())
            .lower()
            .is_err());
        assert!(CommandSpec::program("bin\0x", Vec::<OsString>::new())
            .lower()
            .is_err());
        assert!(CommandSpec::program("bin", vec![OsString::from("a\0b")])
            .lower()
            .is_err());
        assert!(CommandSpec::shell("true", ShellKind::PlatformDefault)
            .lower_with(Some(OsStr::new("")))
            .is_err());
        assert!(CommandSpec::shell("true", ShellKind::PlatformDefault)
            .lower_with(Some(OsStr::new("sh\0")))
            .is_err());
        let big = CommandSpec::shell("x".repeat(10_000), ShellKind::Cmd)
            .lower()
            .unwrap();
        let _ = std::fs::remove_file(std::path::PathBuf::from(&big.args[2]));
    }
}
