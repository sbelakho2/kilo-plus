//! faktor-verify — the verification engine (audit: verification must not
//! depend on the model's discretion).
//!
//! The runtime records test commands the model happened to run; this engine
//! is the counterpart that decides what a change REQUIRES. Given a project
//! type and the files a turn changed it derives the applicable checks
//! (deterministic, max 3, commands ≤ 512 chars, filters are single
//! sanitized `[a-zA-Z0-9_-]` tokens — hostile names are dropped, never
//! interpolated), and [`acceptance`] reduces run results to a strict
//! required-only PASS/FAIL/Pending verdict. The crate is std-only.

use std::sync::Arc;

/// Hard cap on the checks one derivation may return.
pub const MAX_CHECKS: usize = 3;

/// Hard cap on one check command length; longer commands are dropped.
pub const MAX_COMMAND_CHARS: usize = 512;

/// Project kind detected from a repo file map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectType {
    Rust,
    Node,
    Python,
    Go,
    Java,
    Gradle,
    CMake,
    DotNet,
    Unknown,
}

/// What a check does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckKind {
    Compile,
    Test,
    Lint,
}

/// One derived verification check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Check {
    /// Stable id; failed required checks are durably recorded under it.
    pub id: String,
    pub kind: CheckKind,
    /// Shell command executed verbatim (never interpolates hostile input).
    pub command: String,
    /// Changed files the check applies to (empty for project-wide hints).
    pub affects: Vec<String>,
    /// Required checks gate the acceptance; optional ones are hints that
    /// never affect PASS/FAIL.
    pub required: bool,
}

/// Required-only acceptance over derived checks and run results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acceptance {
    Pass,
    Fail,
    Pending,
}

/// The runtime-facing execution handle: a bounded closure that runs one
/// check command and reports Ok on exit code 0. Infra (supervisor) errors,
/// timeouts and non-zero exits are all `Err`.
pub struct Verifier {
    pub run: Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>,
}

impl Verifier {
    pub fn new(run: Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync>) -> Self {
        Self { run }
    }
}

fn has_file(files: &[String], marker: &str) -> bool {
    let needle = format!("/{marker}");
    files
        .iter()
        .any(|f| f == marker || f.ends_with(&needle))
}

/// Detect the project kind from a repo file map (bounded, sorted, root-ish
/// paths). Marker precedence is fixed: Cargo.toml, package.json,
/// pyproject.toml|setup.py, go.mod, pom.xml, build.gradle, CMakeLists.txt,
/// *.sln|*.csproj, else Unknown.
pub fn detect_project_type(files: &[String]) -> ProjectType {
    if has_file(files, "Cargo.toml") {
        return ProjectType::Rust;
    }
    if has_file(files, "package.json") {
        return ProjectType::Node;
    }
    if has_file(files, "pyproject.toml") || has_file(files, "setup.py") {
        return ProjectType::Python;
    }
    if has_file(files, "go.mod") {
        return ProjectType::Go;
    }
    if has_file(files, "pom.xml") {
        return ProjectType::Java;
    }
    if has_file(files, "build.gradle") {
        return ProjectType::Gradle;
    }
    if has_file(files, "CMakeLists.txt") {
        return ProjectType::CMake;
    }
    if files
        .iter()
        .any(|f| f.ends_with(".sln") || f.ends_with(".csproj"))
    {
        return ProjectType::DotNet;
    }
    ProjectType::Unknown
}

fn is_clean_segment(seg: &str) -> bool {
    !seg.is_empty() && seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn file_stem(path: &str) -> Option<&str> {
    let name = path.rsplit('/').next()?;
    let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
    (!stem.is_empty()).then_some(stem)
}

/// The Rust test filter for one changed file: the sanitized stem, only when
/// the stem is a single clean token. Hostile stems yield None — the command
/// is dropped, never interpolated.
fn rust_test_filter(path: &str) -> Option<String> {
    let stem = file_stem(path)?;
    is_clean_segment(stem).then(|| stem.to_string())
}

/// True when the file lives under a directory component equal to `dir`.
fn under_dir(path: &str, dir: &str) -> bool {
    path.split('/').any(|c| c == dir)
}

fn under_any_dir(path: &str, dirs: &[&str]) -> bool {
    dirs.iter().any(|d| under_dir(path, d))
}

/// A changed `.rs` path is test-related when it lives under a `tests/`
/// component or its stem contains "test".
fn is_test_related(path: &str) -> bool {
    under_dir(path, "tests")
        || file_stem(path)
            .map(|s| s.contains("test"))
            .unwrap_or(false)
}

fn parent_dir(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[..i],
        None => "",
    }
}

/// The single-token compileall target for a changed `.py` file: the first
/// (root-most) clean segment of its parent directory — a superset that
/// recursion compiles, and never a path outside the workspace. Unclean or
/// root-less parents yield None (the command is dropped).
fn compileall_token(path: &str) -> Option<String> {
    let parent = parent_dir(path);
    let seg = parent.split('/').next().unwrap_or("");
    is_clean_segment(seg).then(|| seg.to_string())
}

fn push(checks: &mut Vec<Check>, check: Check) {
    if check.command.len() <= MAX_COMMAND_CHARS {
        checks.push(check);
    }
}

/// Derive the checks one changed-file set requires for a project. Rules per
/// project; deterministic (sorted, deduped input, required checks first);
/// bounded (max [`MAX_CHECKS`], commands ≤ [`MAX_COMMAND_CHARS`]); every
/// derived filter is a single sanitized token or the check is dropped.
pub fn derive_checks(project: ProjectType, changed: &[String]) -> Vec<Check> {
    let mut files: Vec<String> = changed.to_vec();
    files.sort();
    files.dedup();
    let mut checks: Vec<Check> = Vec::new();
    match project {
        ProjectType::Rust => {
            let rs: Vec<String> = files
                .iter()
                .filter(|f| f.ends_with(".rs"))
                .cloned()
                .collect();
            if !rs.is_empty() {
                push(
                    &mut checks,
                    Check {
                        id: "rust_check".into(),
                        kind: CheckKind::Compile,
                        command: "cargo check".into(),
                        affects: rs.clone(),
                        required: true,
                    },
                );
            }
            let test_related: Vec<String> = rs
                .iter()
                .filter(|f| is_test_related(f))
                .cloned()
                .collect();
            if !test_related.is_empty() {
                if let Some(stem) = rust_test_filter(&test_related[0]) {
                    push(
                        &mut checks,
                        Check {
                            id: format!("rust_test:{stem}"),
                            kind: CheckKind::Test,
                            command: format!("cargo test {stem}"),
                            affects: test_related,
                            required: true,
                        },
                    );
                }
            } else if !rs.is_empty() {
                push(
                    &mut checks,
                    Check {
                        id: "rust_test_lib".into(),
                        kind: CheckKind::Test,
                        command: "cargo test --lib".into(),
                        affects: vec![],
                        required: false,
                    },
                );
            }
        }
        ProjectType::Node => {
            let ts: Vec<String> = files
                .iter()
                .filter(|f| f.ends_with(".ts") || f.ends_with(".tsx"))
                .cloned()
                .collect();
            if !ts.is_empty() {
                push(
                    &mut checks,
                    Check {
                        id: "node_tsc".into(),
                        kind: CheckKind::Compile,
                        command: "npx tsc --noEmit".into(),
                        affects: ts,
                        required: true,
                    },
                );
            }
            let test_files: Vec<String> = files
                .iter()
                .filter(|f| under_any_dir(f, &["test", "tests", "spec"]))
                .cloned()
                .collect();
            if !test_files.is_empty() {
                push(
                    &mut checks,
                    Check {
                        id: "node_test".into(),
                        kind: CheckKind::Test,
                        command: "npm test".into(),
                        affects: test_files,
                        required: false,
                    },
                );
            }
        }
        ProjectType::Python => {
            let py: Vec<String> = files
                .iter()
                .filter(|f| f.ends_with(".py"))
                .cloned()
                .collect();
            let mut tokens: Vec<String> = py.iter().filter_map(|f| compileall_token(f)).collect();
            tokens.sort();
            tokens.dedup();
            for token in tokens {
                let affected: Vec<String> = py
                    .iter()
                    .filter(|f| compileall_token(f).as_deref() == Some(token.as_str()))
                    .cloned()
                    .collect();
                push(
                    &mut checks,
                    Check {
                        id: format!("python_compile:{token}"),
                        kind: CheckKind::Compile,
                        command: format!("python -m compileall -q {token}"),
                        affects: affected,
                        required: true,
                    },
                );
            }
            let test_files: Vec<String> = py
                .iter()
                .filter(|f| under_dir(f, "tests"))
                .cloned()
                .collect();
            if !test_files.is_empty() {
                push(
                    &mut checks,
                    Check {
                        id: "python_test".into(),
                        kind: CheckKind::Test,
                        command: "pytest -q".into(),
                        affects: test_files,
                        required: false,
                    },
                );
            }
        }
        ProjectType::Go => {
            let go: Vec<String> = files
                .iter()
                .filter(|f| f.ends_with(".go"))
                .cloned()
                .collect();
            if !go.is_empty() {
                push(
                    &mut checks,
                    Check {
                        id: "go_build".into(),
                        kind: CheckKind::Compile,
                        command: "go build ./...".into(),
                        affects: go.clone(),
                        required: true,
                    },
                );
                push(
                    &mut checks,
                    Check {
                        id: "go_test".into(),
                        kind: CheckKind::Test,
                        command: "go test ./...".into(),
                        affects: go,
                        required: false,
                    },
                );
            }
        }
        ProjectType::Java => {
            let java: Vec<String> = files
                .iter()
                .filter(|f| f.ends_with(".java"))
                .cloned()
                .collect();
            if !java.is_empty() {
                push(
                    &mut checks,
                    Check {
                        id: "java_compile".into(),
                        kind: CheckKind::Compile,
                        command: "mvn -q -DskipTests compile".into(),
                        affects: java.clone(),
                        required: true,
                    },
                );
                push(
                    &mut checks,
                    Check {
                        id: "java_test".into(),
                        kind: CheckKind::Test,
                        command: "mvn -q test".into(),
                        affects: java,
                        required: false,
                    },
                );
            }
        }
        ProjectType::Gradle | ProjectType::CMake | ProjectType::DotNet | ProjectType::Unknown => {}
    }
    // Required checks first, then deterministic by id; bounded to MAX_CHECKS.
    checks.sort_by(|a, b| b.required.cmp(&a.required).then_with(|| a.id.cmp(&b.id)));
    checks.truncate(MAX_CHECKS);
    checks
}

/// Required-only acceptance: every required check present and passed is
/// Pass; ANY required check failed is Fail; a required check without a
/// result is Pending. Optional checks never influence the verdict.
pub fn acceptance(checks: &[Check], results: &[(String, bool)]) -> Acceptance {
    let mut missing = false;
    for check in checks.iter().filter(|c| c.required) {
        match results.iter().find(|(id, _)| *id == check.id) {
            Some((_, true)) => {}
            Some((_, false)) => return Acceptance::Fail,
            None => missing = true,
        }
    }
    if missing {
        Acceptance::Pending
    } else {
        Acceptance::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn required(checks: &[Check]) -> Vec<&Check> {
        checks.iter().filter(|c| c.required).collect()
    }

    fn optional(checks: &[Check]) -> Vec<&Check> {
        checks.iter().filter(|c| !c.required).collect()
    }

    // ---------------------------------------------------------- detection

    #[test]
    fn detect_marker_matrix() {
        assert_eq!(detect_project_type(&[]), ProjectType::Unknown);
        assert_eq!(
            detect_project_type(&["Cargo.toml".into(), "src/lib.rs".into()]),
            ProjectType::Rust
        );
        assert_eq!(
            detect_project_type(&["crates/x/Cargo.toml".into()]),
            ProjectType::Rust
        );
        assert_eq!(
            detect_project_type(&["package.json".into(), "index.js".into()]),
            ProjectType::Node
        );
        assert_eq!(
            detect_project_type(&["web/package.json".into()]),
            ProjectType::Node
        );
        assert_eq!(detect_project_type(&["pyproject.toml".into()]), ProjectType::Python);
        assert_eq!(detect_project_type(&["setup.py".into()]), ProjectType::Python);
        assert_eq!(detect_project_type(&["go.mod".into()]), ProjectType::Go);
        assert_eq!(detect_project_type(&["pom.xml".into()]), ProjectType::Java);
        assert_eq!(
            detect_project_type(&["build.gradle".into()]),
            ProjectType::Gradle
        );
        assert_eq!(
            detect_project_type(&["CMakeLists.txt".into()]),
            ProjectType::CMake
        );
        assert_eq!(detect_project_type(&["app.sln".into()]), ProjectType::DotNet);
        assert_eq!(
            detect_project_type(&["src/App.csproj".into()]),
            ProjectType::DotNet
        );
        assert_eq!(
            detect_project_type(&["README.md".into(), "LICENSE".into()]),
            ProjectType::Unknown
        );
    }

    #[test]
    fn detect_marker_precedence_is_fixed() {
        let mixed = vec![
            "package.json".to_string(),
            "Cargo.toml".to_string(),
            "go.mod".to_string(),
        ];
        assert_eq!(detect_project_type(&mixed), ProjectType::Rust);
        let node_python = vec!["pyproject.toml".to_string(), "package.json".to_string()];
        assert_eq!(detect_project_type(&node_python), ProjectType::Node);
        let python_go = vec!["go.mod".to_string(), "setup.py".to_string()];
        assert_eq!(detect_project_type(&python_go), ProjectType::Python);
    }

    #[test]
    fn detect_markers_match_at_any_depth_with_fixed_precedence() {
        // A vendored package.json still reads Node: markers are matched at
        // any depth and the FIRST marker in the fixed order wins.
        let files = vec![
            "pyproject.toml".to_string(),
            "vendor/package.json".to_string(),
        ];
        assert_eq!(detect_project_type(&files), ProjectType::Node);
        let go_python = vec!["go.mod".to_string(), "setup.py".to_string()];
        assert_eq!(detect_project_type(&go_python), ProjectType::Python);
    }

    // ------------------------------------------------------------ rust

    #[test]
    fn rust_source_change_requires_cargo_check() {
        let checks = derive_checks(ProjectType::Rust, &["src/a.rs".into()]);
        assert_eq!(
            required(&checks)
                .into_iter()
                .map(|c| (c.id.as_str(), c.command.as_str()))
                .collect::<Vec<_>>(),
            vec![("rust_check", "cargo check")]
        );
        let optional: Vec<&Check> = optional(&checks);
        assert!(
            optional.iter().any(|c| c.id == "rust_test_lib"),
            "lib-test hint present: {optional:?}"
        );
        assert!(checks.len() <= MAX_CHECKS);
    }

    #[test]
    fn rust_non_rs_changes_derive_nothing() {
        assert_eq!(
            derive_checks(ProjectType::Rust, &["README.md".into(), "Cargo.toml".into()]),
            vec![]
        );
    }

    #[test]
    fn rust_test_change_requires_single_token_test_command() {
        let checks = derive_checks(ProjectType::Rust, &["tests/foo.rs".into()]);
        let req = required(&checks);
        assert_eq!(req.len(), 2);
        assert_eq!(req[0].command, "cargo check");
        assert_eq!(req[1].id, "rust_test:foo");
        assert_eq!(req[1].command, "cargo test foo");
        assert!(req[1].required);
    }

    #[test]
    fn rust_stem_containing_test_is_test_related() {
        let checks = derive_checks(ProjectType::Rust, &["src/payments_test.rs".into()]);
        let test_checks: Vec<&Check> = required(&checks)
            .into_iter()
            .filter(|c| c.id.starts_with("rust_test"))
            .collect();
        assert_eq!(test_checks.len(), 1);
        assert_eq!(test_checks[0].command, "cargo test payments_test");
    }

    #[test]
    fn rust_tests_dir_file_with_sanitizable_stem_uses_stem() {
        let checks = derive_checks(ProjectType::Rust, &["tests/mod_x-2.rs".into()]);
        let test_checks: Vec<&Check> = required(&checks)
            .into_iter()
            .filter(|c| c.id.starts_with("rust_test"))
            .collect();
        assert_eq!(test_checks.len(), 1);
        assert_eq!(test_checks[0].command, "cargo test mod_x-2");
    }

    #[test]
    fn rust_hostile_stem_never_reaches_a_command() {
        let checks = derive_checks(ProjectType::Rust, &["tests/x; rm -rf .rs".into()]);
        assert!(
            checks
                .iter()
                .all(|c| !c.command.contains(';') && !c.command.contains("rm")),
            "hostile stem must never be interpolated: {checks:?}"
        );
        assert_eq!(
            required(&checks).len(),
            1,
            "only the cargo check survives a hostile test stem: {checks:?}"
        );
        // A path that traverses out of the workspace also yields nothing.
        let checks = derive_checks(ProjectType::Rust, &["../escape.rs".into()]);
        assert!(required(&checks).iter().all(|c| c.command != "cargo test .."));
    }

    #[test]
    fn rust_derivation_is_deterministic_and_bounded() {
        let mut files: Vec<String> = (0..12)
            .map(|i| format!("tests/unit/t{i}.rs"))
            .collect();
        files.extend(vec!["src/lib.rs".into(), "tests/unit/a.rs".into()]);
        let a = derive_checks(ProjectType::Rust, &files);
        files.reverse();
        let b = derive_checks(ProjectType::Rust, &files);
        assert_eq!(a, b, "input order must not matter");
        assert!(a.len() <= MAX_CHECKS);
        assert!(a.iter().all(|c| c.command.len() <= MAX_COMMAND_CHARS));
        assert_eq!(a.iter().filter(|c| c.id.starts_with("rust_test:")).count(), 1);
    }

    // ------------------------------------------------------------ node

    #[test]
    fn node_ts_change_requires_tsc_and_test_dir_changes_hint_npm_test() {
        let checks = derive_checks(ProjectType::Node, &["src/app.ts".into()]);
        assert_eq!(
            required(&checks)
                .into_iter()
                .map(|c| (c.id.as_str(), c.command.as_str()))
                .collect::<Vec<_>>(),
            vec![("node_tsc", "npx tsc --noEmit")]
        );
        let checks =
            derive_checks(ProjectType::Node, &["spec/app.test.tsx".into(), "src/x.ts".into()]);
        assert_eq!(
            required(&checks).len(),
            1,
            "tsc stays required; npm test is optional"
        );
        assert_eq!(
            optional(&checks)
                .into_iter()
                .map(|c| c.command.as_str())
                .collect::<Vec<_>>(),
            vec!["npm test"]
        );
        assert!(checks.len() <= MAX_CHECKS);
    }

    #[test]
    fn node_plain_js_changes_derive_nothing() {
        assert_eq!(
            derive_checks(ProjectType::Node, &["src/index.js".into(), "package.json".into()]),
            vec![]
        );
    }

    // ---------------------------------------------------------- python

    #[test]
    fn python_change_requires_compileall_of_the_parent_token() {
        let checks = derive_checks(ProjectType::Python, &["src/app.py".into()]);
        assert_eq!(
            required(&checks)
                .into_iter()
                .map(|c| (c.id.as_str(), c.command.as_str()))
                .collect::<Vec<_>>(),
            vec![("python_compile:src", "python -m compileall -q src")]
        );
        // Nested parents collapse onto the root-most clean segment (a
        // superset for recursive compilation).
        let checks = derive_checks(ProjectType::Python, &["src/ops/deep.py".into()]);
        assert_eq!(
            required(&checks)[0].command,
            "python -m compileall -q src"
        );
        // Sibling changes share one compile check.
        let checks = derive_checks(
            ProjectType::Python,
            &["src/a.py".into(), "src/b.py".into()],
        );
        assert_eq!(required(&checks).len(), 1);
        assert_eq!(required(&checks)[0].affects.len(), 2);
    }

    #[test]
    fn python_hostile_or_unrooted_parents_drop_the_compile_check() {
        for hostile in [
            "x; rm -rf/x.py",
            "../escape.py",
            "..",
            ".hidden/evil.py",
            "app.py",
        ] {
            let checks = derive_checks(ProjectType::Python, &[hostile.to_string()]);
            assert!(
                checks
                    .iter()
                    .all(|c| !c.command.contains(';') && !c.command.contains("..")),
                "{hostile:?} must never reach a command: {checks:?}"
            );
            assert!(
                required(&checks).is_empty(),
                "unrooted/hostile parent yields no required check: {checks:?}"
            );
        }
    }

    #[test]
    fn python_tests_change_hints_pytest_and_keeps_compileall() {
        let checks = derive_checks(ProjectType::Python, &["tests/test_app.py".into()]);
        let req = required(&checks);
        assert_eq!(req.len(), 1);
        assert_eq!(req[0].command, "python -m compileall -q tests");
        assert_eq!(
            optional(&checks)
                .into_iter()
                .map(|c| c.command.as_str())
                .collect::<Vec<_>>(),
            vec!["pytest -q"]
        );
    }

    #[test]
    fn python_compile_checks_are_capped() {
        let files: Vec<String> = (0..10)
            .map(|i| format!("pkg{i}/mod.py"))
            .collect();
        let checks = derive_checks(ProjectType::Python, &files);
        assert!(checks.len() <= MAX_CHECKS);
        assert!(checks.iter().all(|c| c.command.len() <= MAX_COMMAND_CHARS));
    }

    // ------------------------------------------------------------- go

    #[test]
    fn go_change_requires_build_and_hints_test() {
        let checks = derive_checks(ProjectType::Go, &["cmd/main.go".into()]);
        assert_eq!(
            checks
                .iter()
                .map(|c| (c.id.as_str(), c.command.as_str(), c.required))
                .collect::<Vec<_>>(),
            vec![
                ("go_build", "go build ./...", true),
                ("go_test", "go test ./...", false),
            ]
        );
        assert_eq!(derive_checks(ProjectType::Go, &["go.mod".into()]), vec![]);
    }

    // ------------------------------------------------------------ java

    #[test]
    fn java_change_requires_mvn_compile_and_hints_test() {
        let checks = derive_checks(ProjectType::Java, &["src/Main.java".into()]);
        assert_eq!(
            checks
                .iter()
                .map(|c| (c.id.as_str(), c.command.as_str(), c.required))
                .collect::<Vec<_>>(),
            vec![
                ("java_compile", "mvn -q -DskipTests compile", true),
                ("java_test", "mvn -q test", false),
            ]
        );
    }

    #[test]
    fn projects_without_change_rules_derive_nothing() {
        for p in [
            ProjectType::Gradle,
            ProjectType::CMake,
            ProjectType::DotNet,
            ProjectType::Unknown,
        ] {
            assert_eq!(
                derive_checks(p, &["build.gradle".into(), "x.cpp".into(), "a.cs".into()]),
                vec![],
                "{p:?}"
            );
        }
    }

    // --------------------------------------------------------- acceptance

    #[test]
    fn acceptance_required_only_matrix() {
        let checks = derive_checks(ProjectType::Rust, &["tests/foo.rs".into()]);
        let ids: Vec<String> = required(&checks).iter().map(|c| c.id.clone()).collect();
        assert_eq!(ids.len(), 2);
        // All required passed => Pass.
        let all_ok: Vec<(String, bool)> = ids.iter().map(|i| (i.clone(), true)).collect();
        assert_eq!(acceptance(&checks, &all_ok), Acceptance::Pass);
        // Any required failed => Fail.
        let one_fail: Vec<(String, bool)> = vec![("rust_check".into(), true), ("rust_test:foo".into(), false)];
        assert_eq!(acceptance(&checks, &one_fail), Acceptance::Fail);
        // Missing required result => Pending.
        let partial: Vec<(String, bool)> = vec![("rust_check".into(), true)];
        assert_eq!(acceptance(&checks, &partial), Acceptance::Pending);
        // A failed result for an OPTIONAL check never fails the verdict.
        let optional_fail: Vec<(String, bool)> = all_ok
            .iter()
            .cloned()
            .chain(vec![("rust_test_lib".into(), false)])
            .collect();
        assert_eq!(acceptance(&checks, &optional_fail), Acceptance::Pass);
        // Fail dominates a concurrently missing required check.
        let fail_and_missing: Vec<(String, bool)> = vec![("rust_check".into(), false)];
        assert_eq!(acceptance(&checks, &fail_and_missing), Acceptance::Fail);
    }

    #[test]
    fn acceptance_empty_or_hint_only_is_pass() {
        assert_eq!(acceptance(&[], &[]), Acceptance::Pass);
        let hint_only = derive_checks(ProjectType::Rust, &["src/a.rs".into()]);
        assert!(!required(&hint_only).is_empty());
        let res: Vec<(String, bool)> = required(&hint_only)
            .iter()
            .map(|c| (c.id.clone(), true))
            .collect();
        assert_eq!(acceptance(&hint_only, &res), Acceptance::Pass);
        // Optional checks alone can never turn a verdict.
        let go = derive_checks(ProjectType::Go, &["main.go".into()]);
        let res = vec![("go_build".into(), true)];
        assert_eq!(acceptance(&go, &res), Acceptance::Pass);
    }

    #[test]
    fn acceptance_unknown_results_are_treated_as_missing() {
        let checks = derive_checks(ProjectType::Rust, &["tests/a.rs".into()]);
        let unrelated = vec![("rust_check".into(), true), ("not_a_check".into(), false)];
        assert_eq!(acceptance(&checks, &unrelated), Acceptance::Pending);
    }
}
