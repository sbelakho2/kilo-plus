//! Toolchain probing. A task whose language toolchain is missing on the
//! runner is a documented SKIP, never a failure (CI images differ; the
//! corpus must stay runnable where a toolchain exists).

use crate::corpus::Lang;

/// True when `name` resolves to an executable on PATH.
pub fn in_path(name: &str) -> bool {
    let Some(paths) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&paths).any(|dir| {
        let candidate = dir.join(name);
        if !candidate.is_file() {
            return false;
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            candidate
                .metadata()
                .map(|m| m.permissions().mode() & 0o111 != 0)
                .unwrap_or(false)
        }
        #[cfg(not(unix))]
        {
            candidate.is_file()
        }
    })
}

/// Run `node -p ...`-style version probing: returns `(major, minor)` or an
/// error string.
fn node_version() -> Result<(u64, u64), String> {
    let out = std::process::Command::new("node")
        .arg("--version")
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("node --version exited {}", out.status));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let text = text.trim().strip_prefix('v').unwrap_or(text.trim());
    let mut parts = text.split('.');
    let major = parts
        .next()
        .and_then(|p| p.parse::<u64>().ok())
        .ok_or_else(|| format!("unparseable node version {text:?}"))?;
    let minor = parts
        .next()
        .and_then(|p| p.parse::<u64>().ok())
        .unwrap_or(0);
    Ok((major, minor))
}

/// The TypeScript task runs real `.ts` sources through node's built-in
/// type stripping (no npm install, no build step — corpus policy). The
/// matrix: node >= 22.6 with `--experimental-strip-types`; >= 23.6 runs
/// without the flag. The task's `verify.sh` mirrors this exactly.
fn node_supports_type_stripping() -> Result<(), String> {
    let (major, minor) = node_version()?;
    if major >= 23 || (major == 22 && minor >= 6) {
        return Ok(());
    }
    Err(format!(
        "node {major}.{minor} cannot strip TypeScript types; node >= 22.6 required"
    ))
}

/// Check the full toolchain of `lang`. `Ok(())` = every binary is on PATH
/// (and the version gates pass); `Err(detail)` = a documented skip reason.
pub fn require_toolchain(lang: Lang) -> Result<(), String> {
    if lang == Lang::TypeScript {
        if !in_path("node") {
            return Err("toolchain missing: node is not on PATH".into());
        }
        if let Err(e) = node_supports_type_stripping() {
            return Err(format!("toolchain gate: {e}"));
        }
        return Ok(());
    }
    let mut missing = Vec::new();
    for tool in lang.toolchain() {
        if !in_path(tool) {
            missing.push(*tool);
        }
    }
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "toolchain missing for {}: {} not on PATH",
            lang,
            missing.join(", ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_toolchain_present_here() {
        // The crate itself builds with cargo, so the rust toolchain must
        // always probe present in this environment.
        assert!(require_toolchain(Lang::Rust).is_ok());
    }
}
