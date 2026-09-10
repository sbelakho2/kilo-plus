//! Integration tests for `scripts/supply-chain.sh` (audit 89/90 release
//! evidence): the fast path must exit 0 with every skip RECORDED, and the
//! `TAMPER=1` self-test must exit non-zero when a recorded artifact
//! checksum is corrupted.
//!
//! `SUPPLY_CHAIN_SKIP_AUDIT=1` keeps these tests hermetic (no network, no
//! advisory-database fetch): the skip is itself part of what is asserted.

use std::path::{Path, PathBuf};
use std::process::Command;

fn script_path() -> PathBuf {
    crate::fixtures::repo_root().join("scripts/supply-chain.sh")
}

fn run_script(out_dir: &Path, extra_env: &[(&str, &str)]) -> std::process::Output {
    let mut command = Command::new("bash");
    command
        .arg(script_path())
        .current_dir(crate::fixtures::repo_root())
        .env("SUPPLY_CHAIN_OUT_DIR", out_dir)
        .env("SUPPLY_CHAIN_FAST", "1")
        .env("SUPPLY_CHAIN_SKIP_AUDIT", "1")
        .env_remove("TAMPER");
    for (key, value) in extra_env {
        command.env(key, value);
    }
    command.output().expect("bash runs the supply-chain script")
}

fn read_json(path: &Path) -> serde_json::Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("parse {} as JSON: {e}", path.display()))
}

#[test]
fn fast_path_exits_zero_with_recorded_skips() {
    let out = tempfile::tempdir().unwrap();
    let output = run_script(out.path(), &[]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "fast path must exit 0\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    // SBOM: real cargo metadata projection, valid JSON, non-empty.
    let sbom = read_json(&out.path().join("sbom.json"));
    assert_eq!(sbom["schema"], "faktor-sbom/1");
    assert!(
        sbom["package_count"].as_u64().unwrap_or(0) > 0,
        "SBOM must inventory packages: {sbom}"
    );

    // Status: pass, with audit skips recorded because SKIP_AUDIT=1.
    let status = read_json(&out.path().join("supply-chain-status.json"));
    assert_eq!(status["status"], "pass");
    let skips = status["skips"].as_array().expect("skips array");
    let names: Vec<&str> = skips
        .iter()
        .filter_map(|skip| skip["name"].as_str())
        .collect();
    assert!(names.contains(&"cargo-audit"), "recorded skips: {names:?}");
    assert!(names.contains(&"cargo-deny"), "recorded skips: {names:?}");
    assert!(
        names.contains(&"artifact-build"),
        "fast path must record the skipped build: {names:?}"
    );

    // Artifacts: either hashed (tool pass) or honestly skipped.
    let artifact_count = status["artifact_count"].as_u64().unwrap_or(0);
    let tool_names: Vec<&str> = status["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect();
    if artifact_count > 0 {
        assert!(tool_names.contains(&"checksum-verify"));
    } else {
        assert!(
            names.contains(&"artifacts"),
            "no artifacts hashed and no skip: {status}"
        );
    }

    // Never silent: skips are also printed.
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.to_ascii_lowercase().contains("skip"),
        "skips must be visible in the output:\n{combined}"
    );
}

#[test]
fn tamper_self_test_rejects_a_corrupted_checksum() {
    let out = tempfile::tempdir().unwrap();
    let output = run_script(out.path(), &[("TAMPER", "1")]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "TAMPER=1 must exit non-zero\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("tampered checksum rejected"),
        "self-test must report the rejection:\n{combined}"
    );
    assert!(
        combined.contains("CHECKSUM MISMATCH") || combined.contains("tampered checksum rejected"),
        "self-test evidence missing:\n{combined}"
    );
}
