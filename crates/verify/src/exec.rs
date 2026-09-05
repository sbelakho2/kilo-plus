//! Typed, async, project-aware verification (P0-9/P0-10/P0-11).
//!
//! This module is the modernization target for the legacy string-command
//! `Verifier`: checks are typed `(program, args)` specs — never `sh -c`
//! unless a repository rule itself specifies a shell command — executed on
//! the existing Tokio runtime by [`AsyncCheckExecutor`] with no per-check
//! OS thread and no nested runtime. Every check runs against the exact
//! worktree in [`VerificationContext`] (never the daemon's current
//! directory), is bounded by the context deadline and cancellation token,
//! and is killed process-group-wide so no orphan survives.
//!
//! Budgets ([`budget_for`]) are derived from the check category, the
//! remaining turn budget and a [`VerificationPolicy`] — never a universal
//! ten-second wall cap. [`derive_typed_checks`] makes CMake/Make/Meson/
//! Ninja/Bazel/MSBuild/.csproj/Gradle first-class project types instead of
//! resolving to an empty check list.

use std::ffi::{OsStr, OsString};
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use faktor_core::cancellation::CancellationToken;
use faktor_core::error::Error;

/// How heavy a check is. Drives the execution budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckCategory {
    /// Syntax/type-level checks: 30-60 s class budgets.
    Quick,
    /// Normal unit verification: up to the remaining turn budget.
    Unit,
    /// Expensive full-repository verification (build + full test suite):
    /// task-owned background operation with progress where enabled.
    Full,
}

/// What one derived check does (mirrors the legacy `CheckKind` values).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckKind {
    Compile,
    Test,
    Lint,
}

/// A typed check: program + argv, no shell interpolation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckSpec {
    /// Stable id; failed required checks are recorded durably under it.
    pub id: String,
    pub kind: CheckKind,
    pub category: CheckCategory,
    /// Absolute-or-resolved executable. Resolution happens against the
    /// context root then PATH (see [`AsyncCheckExecutor`]).
    pub program: OsString,
    pub args: Vec<OsString>,
    /// Working directory RELATIVE to the verification root. Absolute or
    /// escaping (`..`) values are rejected before spawn.
    pub cwd_rel: PathBuf,
    /// Changed files the check applies to (empty = project-wide).
    pub affects: Vec<String>,
    /// Required checks gate acceptance; optional ones are hints.
    pub required: bool,
}

impl CheckSpec {
    pub fn new(
        id: impl Into<String>,
        kind: CheckKind,
        category: CheckCategory,
        program: impl Into<OsString>,
        args: impl IntoIterator<Item = impl Into<OsString>>,
        required: bool,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            category,
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
            cwd_rel: PathBuf::from("."),
            affects: Vec::new(),
            required,
        }
    }
}

/// Everything a verification run needs to know about the world. The root is
/// the exact task-owned worktree — never the daemon process cwd.
#[derive(Debug, Clone)]
pub struct VerificationContext {
    pub session_id: u64,
    pub task_id: u64,
    pub operation_id: u64,
    pub workspace_id: u64,
    pub worktree_id: u64,
    pub root: PathBuf,
    pub deadline: Instant,
    pub cancellation: CancellationToken,
}

/// Result of one executed check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckOutcome {
    pub status: CheckRunStatus,
    /// Exit code when the process ran to completion.
    pub exit: Option<i32>,
    pub started_ms: i64,
    pub finished_ms: i64,
    /// Bounded tail of the combined output (last [`SUMMARY_MAX_BYTES`]).
    pub summary: Option<String>,
    /// Output exceeded the capture cap: the child kept running, but only
    /// the bounded tail was retained.
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckRunStatus {
    /// Exit code 0.
    Passed,
    /// Ran to completion with a non-zero exit.
    Failed,
    /// Could not run or did not complete: killed by deadline/cancellation,
    /// program missing, infra error. NEVER treated as a failure of the
    /// code under check — the caller decides (BlockedVerification).
    Unavailable,
}

/// Policy for verification budgets (P0-10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationPolicy {
    /// Quick checks get at most this much wall time.
    pub quick_max: Duration,
    /// Unit checks get at most this much wall time inline.
    pub unit_max: Duration,
    /// Full-repository checks run as a task-owned background operation with
    /// progress instead of blocking the turn.
    pub full_as_background: bool,
    /// Below this much remaining turn budget, checks are not run inline.
    pub min_inline: Duration,
}

impl Default for VerificationPolicy {
    fn default() -> Self {
        Self {
            quick_max: Duration::from_secs(60),
            unit_max: Duration::from_secs(600),
            full_as_background: true,
            min_inline: Duration::from_secs(5),
        }
    }
}

/// A policy of zero budget fails closed: nothing runs inline.
impl VerificationPolicy {
    pub fn disabled() -> Self {
        Self {
            quick_max: Duration::ZERO,
            unit_max: Duration::ZERO,
            full_as_background: true,
            min_inline: Duration::ZERO,
        }
    }
}

/// How a check should be executed given its category and the turn's
/// remaining budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetDecision {
    /// Run now, inline, with at most the given wall budget.
    RunInline(Duration),
    /// Too heavy for the remaining inline budget: the caller should run it
    /// as a task-owned background operation with explicit progress.
    RunAsTaskOwnedOperation,
}

/// Budget for a check category (P0-10): quick syntax/type checks get tens
/// of seconds; unit verification may use the remaining turn budget; full
/// repository verification goes background by policy. Never a universal
/// ~10 s cap, and never a hang: below the minimum inline floor everything
/// goes background.
pub fn budget_for(
    category: CheckCategory,
    policy: &VerificationPolicy,
    remaining_turn_budget: Option<Duration>,
) -> BudgetDecision {
    if policy.quick_max.is_zero() && policy.unit_max.is_zero() {
        return BudgetDecision::RunAsTaskOwnedOperation;
    }
    let (cap, background) = match category {
        CheckCategory::Quick => (policy.quick_max, false),
        CheckCategory::Unit => (policy.unit_max, false),
        CheckCategory::Full => (policy.unit_max, policy.full_as_background),
    };
    if background {
        return BudgetDecision::RunAsTaskOwnedOperation;
    }
    if cap.is_zero() {
        return BudgetDecision::RunAsTaskOwnedOperation;
    }
    let budget = match remaining_turn_budget {
        Some(remaining) => cap.min(remaining),
        None => cap,
    };
    if budget < policy.min_inline {
        return BudgetDecision::RunAsTaskOwnedOperation;
    }
    BudgetDecision::RunInline(budget)
}

const OUTPUT_CAP_BYTES: usize = 1 << 20; // 1 MiB of output retained per stream
const SUMMARY_MAX_BYTES: usize = 4096;

/// Async check executor: tokio child processes with bounded capture,
/// deadline/cancellation kill, process-group cleanup and single reap.
/// No OS thread per check, no nested Tokio runtime.
#[derive(Debug, Clone)]
pub struct AsyncCheckExecutor {
    output_cap: usize,
}

impl Default for AsyncCheckExecutor {
    fn default() -> Self {
        Self {
            output_cap: OUTPUT_CAP_BYTES,
        }
    }
}

impl AsyncCheckExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_output_cap(mut self, cap: usize) -> Self {
        self.output_cap = cap.max(4096);
        self
    }

    fn resolve_program(&self, root: &Path, program: &OsStr) -> PathBuf {
        let p = Path::new(program);
        if p.components().count() > 1 || p.is_absolute() {
            // An explicit path is resolved relative to the verification root
            // first (the worktree may vendor its own tooling), then as-is.
            let rooted = root.join(p);
            if rooted.exists() {
                return rooted;
            }
            return p.to_path_buf();
        }
        // Bare name: resolve against PATH. tokio::process does that itself
        // when the program is a bare name, so pass it through.
        p.to_path_buf()
    }

    /// Validate a relative cwd: must be relative, no `..`, no absolute.
    fn validate_cwd_rel(root: &Path, cwd_rel: &Path) -> Result<PathBuf, Error> {
        if cwd_rel.is_absolute() {
            return Err(Error::malformed(format!(
                "check cwd must be relative to the verification root, got absolute {cwd_rel:?}"
            )));
        }
        for component in cwd_rel.components() {
            if let Component::ParentDir | Component::RootDir | Component::Prefix(_) = component {
                return Err(Error::malformed(format!(
                    "check cwd escapes the verification root: {cwd_rel:?}"
                )));
            }
        }
        let joined = root.join(cwd_rel);
        let meta = std::fs::metadata(&joined)
            .map_err(|e| Error::internal(format!("check cwd {}: {e}", joined.display())))?;
        if !meta.is_dir() {
            return Err(Error::internal(format!(
                "check cwd {} is not a directory",
                joined.display()
            )));
        }
        Ok(joined)
    }

    /// Run one check under the context's deadline and cancellation. The
    /// process is spawned in its own process group on unix so a deadline
    /// kill takes grandchildren too (zero orphans). Exactly one `wait`.
    pub async fn run_check(
        &self,
        spec: &CheckSpec,
        ctx: &VerificationContext,
    ) -> Result<CheckOutcome, Error> {
        let started_ms = now_ms();
        let cwd = Self::validate_cwd_rel(&ctx.root, &spec.cwd_rel)?;
        let program = self.resolve_program(&ctx.root, &spec.program);

        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&spec.args).current_dir(&cwd).kill_on_drop(true);
        #[cfg(unix)]
        cmd.process_group(0);
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(CheckOutcome {
                    status: CheckRunStatus::Unavailable,
                    exit: None,
                    started_ms,
                    finished_ms: now_ms(),
                    summary: Some(format!("program not found: {}", program.display())),
                    truncated: false,
                });
            }
            Err(e) => {
                return Err(Error::internal(format!("spawn {}: {e}", program.display())));
            }
        };

        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
        // The capture tasks drain BOTH pipes concurrently with the child's
        // execution: a pipe that is not being read would otherwise fill and
        // block the child (a deadlock the deadline can only paper over).
        let cap = self.output_cap;
        let out_task = tokio::spawn(capture_bounded(stdout, cap));
        let err_task = tokio::spawn(capture_bounded(stderr, cap));

        let deadline = ctx.deadline;
        let cancellation = ctx.cancellation.clone();
        let pid = child.id().unwrap_or(0);

        let outcome = tokio::select! {
            status = child.wait() => {
                let status = status.map_err(|e| {
                    Error::internal(format!("wait {}: {e}", program.display()))
                })?;
                let (out, out_t) = out_task.await.expect("capture task panicked");
                let (err, err_t) = err_task.await.expect("capture task panicked");
                let exit = status.code();
                let run_status = if status.success() {
                    CheckRunStatus::Passed
                } else if exit.is_some() {
                    CheckRunStatus::Failed
                } else {
                    CheckRunStatus::Unavailable
                };
                Ok(CheckOutcome {
                    status: run_status,
                    exit,
                    started_ms,
                    finished_ms: now_ms(),
                    summary: Some(build_summary(&out, &err)),
                    truncated: out_t || err_t,
                })
            }
            _ = sleep_until(deadline) => {
                kill_group(pid);
                let _ = child.wait().await; // single reap
                let (out, out_t) = out_task.await.expect("capture task panicked");
                let (err, err_t) = err_task.await.expect("capture task panicked");
                Ok(CheckOutcome {
                    status: CheckRunStatus::Unavailable,
                    exit: None,
                    started_ms,
                    finished_ms: now_ms(),
                    summary: Some(format!(
                        "killed by deadline; output tail: {}",
                        build_summary(&out, &err)
                    )),
                    truncated: out_t || err_t,
                })
            }
            _ = cancellation.cancelled() => {
                kill_group(pid);
                let _ = child.wait().await; // single reap
                let (_out, _out_t) = out_task.await.expect("capture task panicked");
                let (_err, _err_t) = err_task.await.expect("capture task panicked");
                Ok(CheckOutcome {
                    status: CheckRunStatus::Unavailable,
                    exit: None,
                    started_ms,
                    finished_ms: now_ms(),
                    summary: Some("cancelled".into()),
                    truncated: false,
                })
            }
        };
        outcome
    }
}

/// Kill the whole process group (unix). On non-unix, `kill_on_drop` and the
/// subsequent `wait` cover the direct child only (documented limitation).
fn kill_group(pid: u32) {
    #[cfg(unix)]
    unsafe {
        // Negative pid = the process group the child was placed in via
        // process_group(0) (its pgid == its pid).
        libc::kill(-(pid as libc::pid_t), libc::SIGKILL);
    }
    #[cfg(not(unix))]
    let _ = pid;
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

async fn sleep_until(deadline: Instant) {
    let now = Instant::now();
    if deadline > now {
        tokio::time::sleep(deadline - now).await;
    }
}

/// Read a pipe to EOF, retaining only the LAST `cap` bytes (stream-discard:
/// the child never blocks on a full pipe, memory stays bounded, and the
/// retained tail is what summaries need). Returns (tail, truncated).
async fn capture_bounded<R: tokio::io::AsyncRead + Unpin>(
    mut reader: R,
    cap: usize,
) -> (Vec<u8>, bool) {
    use tokio::io::AsyncReadExt;
    let mut buf = Vec::with_capacity(cap.min(64 * 1024));
    let mut chunk = [0u8; 64 * 1024];
    let mut truncated = false;
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => {
                if buf.len() + n > cap {
                    truncated = true;
                    let keep = cap.min(n);
                    let drop_front = buf.len() + n - cap;
                    if drop_front >= buf.len() {
                        buf.clear();
                    } else {
                        buf.drain(0..drop_front);
                    }
                    buf.extend_from_slice(&chunk[n - keep..n]);
                } else {
                    buf.extend_from_slice(&chunk[..n]);
                }
            }
            Err(_) => break,
        }
    }
    (buf, truncated)
}

fn build_summary(stdout: &[u8], stderr: &[u8]) -> String {
    let mut text = String::new();
    for (label, bytes) in [("stdout", stdout), ("stderr", stderr)] {
        if bytes.is_empty() {
            continue;
        }
        let tail: String = String::from_utf8_lossy(bytes)
            .chars()
            .rev()
            .take(SUMMARY_MAX_BYTES)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        text.push_str(&format!("[{label}]\n{tail}\n"));
    }
    text
}

// ------------------------------------------------------------------ budget

/// Full-repository builder markers, checked in manifest order. Root-aware
/// derivation reads ONLY the named manifest files, bounded, and produces
/// typed specs whose `cwd_rel` pins the exact build directory. Program
/// availability is NOT checked here — the executor reports Unavailable.
const MAX_MANIFEST_READ: u64 = 256 * 1024;

fn has_file(files: &[String], marker: &str) -> bool {
    let needle = format!("/{marker}");
    files.iter().any(|f| f == marker || f.ends_with(&needle))
}

fn changed_of(files: &[String], exts: &[&str]) -> bool {
    files.iter().any(|f| exts.iter().any(|e| f.ends_with(e)))
}

const C_SOURCES: &[&str] = &[
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".rs",
];
const CMAKE_FILES: &[&str] = &["CMakeLists.txt", "CMakePresets.json"];

/// Bound-read a manifest and probe it for a target line. Never more than
/// MAX_MANIFEST_READ bytes; a bigger file yields `None` (documented skip —
/// never a partial parse used as truth).
fn manifest_has_target(root: &Path, rel: &str, needles: &[&str]) -> Option<bool> {
    let path = root.join(rel);
    let meta = std::fs::metadata(&path).ok()?;
    if meta.len() > MAX_MANIFEST_READ {
        return None;
    }
    let bytes = std::fs::read(&path).ok()?;
    let text = String::from_utf8_lossy(&bytes);
    let mut result = false;
    for line in text.lines().take(4096) {
        let line = line.trim();
        // Target definitions look like `name:` at column 0 followed by
        // non-space (rule) or nothing (phony-ish). Colons inside recipes
        // are indented.
        if line.starts_with('\t') || line.starts_with(' ') || line.is_empty() {
            continue;
        }
        if let Some((target, _)) = line.split_once(':') {
            if needles.iter().any(|n| *n == target.trim()) {
                result = true;
            }
        }
    }
    Some(result)
}

/// Root-aware typed check derivation for the C/C++/builder families
/// (P0-11/P0-98). Returns an EMPTY list only when nothing applies; every
/// recognized manifest produces at least a required build check.
pub fn derive_typed_checks(root: &Path, files: &[String]) -> Vec<CheckSpec> {
    let mut checks: Vec<CheckSpec> = Vec::new();
    let mut sorted: Vec<String> = files.to_vec();
    sorted.sort();
    sorted.dedup();

    let has_cmake = has_file(&sorted, "CMakeLists.txt");
    let has_cmake_presets = has_file(&sorted, "CMakePresets.json");
    let has_makefile = has_file(&sorted, "Makefile")
        || has_file(&sorted, "makefile")
        || has_file(&sorted, "GNUmakefile");
    let has_meson = has_file(&sorted, "meson.build");
    let has_ninja = has_file(&sorted, "build.ninja");
    let has_bazel = has_file(&sorted, "MODULE.bazel") || has_file(&sorted, "WORKSPACE");
    let has_gradle = has_file(&sorted, "build.gradle")
        || has_file(&sorted, "build.gradle.kts")
        || has_file(&sorted, "settings.gradle.kts");
    let has_gradlew = has_file(&sorted, "gradlew");
    let has_dotnet = sorted
        .iter()
        .any(|f| f.ends_with(".sln") || f.ends_with(".csproj"));
    let c_changed = changed_of(&sorted, C_SOURCES)
        || has_cmake
        || has_cmake_presets
        || has_makefile
        || has_meson
        || has_ninja;

    // CMake: preset when a CMakePresets.json with a readable first
    // configurePreset name exists, else plain configure+build into the
    // deterministic build dir; ctest when tests are discoverable from the
    // file map.
    if has_cmake {
        let preset_name = if has_cmake_presets {
            default_preset_name(root)
        } else {
            String::new()
        };
        let use_preset = !preset_name.is_empty();
        let (cfg_args, build_args): (Vec<OsString>, Vec<OsString>) = if use_preset {
            (
                vec!["--preset".into(), preset_name.clone().into()],
                vec![
                    "--build".into(),
                    "--preset".into(),
                    preset_name.clone().into(),
                ],
            )
        } else {
            (
                vec!["-S".into(), ".".into(), "-B".into(), BUILD_DIR.into()],
                vec!["--build".into(), BUILD_DIR.into()],
            )
        };
        checks.push(CheckSpec {
            id: "cmake_configure".into(),
            kind: CheckKind::Compile,
            category: CheckCategory::Unit,
            program: "cmake".into(),
            args: cfg_args,
            cwd_rel: PathBuf::from("."),
            affects: sorted
                .iter()
                .filter(|f| CMAKE_FILES.iter().any(|m| f.ends_with(m)))
                .cloned()
                .collect(),
            required: true,
        });
        checks.push(CheckSpec {
            id: "cmake_build".into(),
            kind: CheckKind::Compile,
            category: CheckCategory::Unit,
            program: "cmake".into(),
            args: build_args,
            cwd_rel: PathBuf::from("."),
            affects: vec![],
            required: true,
        });
        let tests_discoverable = sorted.iter().any(|f| {
            f.ends_with("CTestTestfile.cmake")
                || f.starts_with("tests/")
                || f.starts_with("test/")
                || f.contains("/tests/")
                || f.contains("/test/")
        });
        if tests_discoverable || use_preset {
            let mut cargs = vec!["--output-on-failure".into()];
            if use_preset {
                cargs.push("--preset".into());
                cargs.push(preset_name.into());
            } else {
                cargs.push("--test-dir".into());
                cargs.push(BUILD_DIR.into());
            }
            checks.push(CheckSpec {
                id: "cmake_ctest".into(),
                kind: CheckKind::Test,
                category: CheckCategory::Full,
                program: "ctest".into(),
                args: cargs,
                cwd_rel: PathBuf::from("."),
                affects: vec![],
                required: true,
            });
        }
    } else if has_makefile && c_changed {
        // Make: `make -j` build is required when C/C++ sources changed;
        // the test/check target is derived only when the Makefile actually
        // defines one (bounded probe) — never guessed into a required
        // failing command.
        checks.push(CheckSpec {
            id: "make_build".into(),
            kind: CheckKind::Compile,
            category: CheckCategory::Unit,
            program: "make".into(),
            args: vec!["-j".into()],
            cwd_rel: PathBuf::from("."),
            affects: vec![],
            required: true,
        });
        for (target, id) in [("test", "make_test"), ("check", "make_check")] {
            if manifest_has_target(root, "Makefile", &[target]) == Some(true) {
                checks.push(CheckSpec {
                    id: id.into(),
                    kind: CheckKind::Test,
                    category: CheckCategory::Full,
                    program: "make".into(),
                    args: vec![target.into()],
                    cwd_rel: PathBuf::from("."),
                    affects: vec![],
                    required: true,
                });
            }
        }
    } else if has_meson && c_changed {
        checks.push(CheckSpec {
            id: "meson_setup".into(),
            kind: CheckKind::Compile,
            category: CheckCategory::Unit,
            program: "meson".into(),
            args: vec!["setup".into(), BUILD_DIR.into()],
            cwd_rel: PathBuf::from("."),
            affects: vec![],
            required: true,
        });
        checks.push(CheckSpec {
            id: "meson_compile".into(),
            kind: CheckKind::Compile,
            category: CheckCategory::Unit,
            program: "meson".into(),
            args: vec!["compile".into(), "-C".into(), BUILD_DIR.into()],
            cwd_rel: PathBuf::from("."),
            affects: vec![],
            required: true,
        });
        checks.push(CheckSpec {
            id: "meson_test".into(),
            kind: CheckKind::Test,
            category: CheckCategory::Full,
            program: "meson".into(),
            args: vec!["test".into(), "-C".into(), BUILD_DIR.into()],
            cwd_rel: PathBuf::from("."),
            affects: vec![],
            required: true,
        });
    } else if has_ninja && c_changed {
        let dir = ninja_build_dir(&sorted);
        checks.push(CheckSpec {
            id: "ninja_build".into(),
            kind: CheckKind::Compile,
            category: CheckCategory::Unit,
            program: "ninja".into(),
            args: vec!["-C".into(), dir.clone().into()],
            cwd_rel: PathBuf::from("."),
            affects: vec![],
            required: true,
        });
        checks.push(CheckSpec {
            id: "ninja_test".into(),
            kind: CheckKind::Test,
            category: CheckCategory::Full,
            program: "ninja".into(),
            args: vec!["-C".into(), dir.into(), "test".into()],
            cwd_rel: PathBuf::from("."),
            affects: vec![],
            required: false,
        });
    } else if has_bazel && c_changed {
        checks.push(CheckSpec {
            id: "bazel_test_affected".into(),
            kind: CheckKind::Test,
            category: CheckCategory::Full,
            program: "bazel".into(),
            args: vec!["test".into(), "//...".into()],
            cwd_rel: PathBuf::from("."),
            affects: vec![],
            required: true,
        });
    }

    // Gradle: wrapper-only (never a global gradle); compile is required
    // when sources changed; tests optional unless a test file changed.
    if has_gradle || has_gradlew {
        let wrapper = if has_gradlew {
            "./gradlew".to_string()
        } else {
            return checks; // no wrapper: no derived checks (documented)
        };
        let java_kotlin_changed = changed_of(&sorted, &[".java", ".kt", ".kts"]) || has_gradle;
        if java_kotlin_changed {
            checks.push(CheckSpec {
                id: "gradle_classes".into(),
                kind: CheckKind::Compile,
                category: CheckCategory::Unit,
                program: wrapper.clone().into(),
                args: vec!["classes".into()],
                cwd_rel: PathBuf::from("."),
                affects: vec![],
                required: true,
            });
            let tests_changed = sorted.iter().any(|f| {
                f.contains("/src/test/") || f.ends_with("Test.kt") || f.ends_with("Test.java")
            });
            if tests_changed {
                checks.push(CheckSpec {
                    id: "gradle_test".into(),
                    kind: CheckKind::Test,
                    category: CheckCategory::Full,
                    program: wrapper.into(),
                    args: vec!["test".into()],
                    cwd_rel: PathBuf::from("."),
                    affects: vec![],
                    required: true,
                });
            }
        }
    }

    // .NET: dotnet build required; dotnet test when a test project exists.
    if has_dotnet {
        let has_test_project = sorted
            .iter()
            .any(|f| f.ends_with(".csproj") && f.contains("Test"));
        checks.push(CheckSpec {
            id: "dotnet_build".into(),
            kind: CheckKind::Compile,
            category: CheckCategory::Unit,
            program: "dotnet".into(),
            args: vec!["build".into()],
            cwd_rel: PathBuf::from("."),
            affects: vec![],
            required: true,
        });
        if has_test_project {
            checks.push(CheckSpec {
                id: "dotnet_test".into(),
                kind: CheckKind::Test,
                category: CheckCategory::Full,
                program: "dotnet".into(),
                args: vec!["test".into()],
                cwd_rel: PathBuf::from("."),
                affects: vec![],
                required: true,
            });
        }
    }

    checks
}

/// Build directory used by derived CMake/Meson checks: deterministic,
/// inside the verification worktree, never the source tree root itself.
pub const BUILD_DIR: &str = ".faktor-verify-build";

fn ninja_build_dir(files: &[String]) -> String {
    // build.ninja usually lives in a build dir; we know only the file map.
    // A single-level parent dir is derived safely; nested or top-level
    // build.ninja files run from the root (ninja finds them via the -C
    // dir's own rules when the default is a build dir; top-level files are
    // rare generated artifacts — documented).
    for f in files {
        if f.ends_with("build.ninja") {
            let parent = f.trim_end_matches("build.ninja").trim_end_matches('/');
            if !parent.is_empty() && !parent.contains('/') {
                return parent.to_string();
            }
        }
    }
    ".".to_string()
}

fn default_preset_name(root: &Path) -> String {
    // Bounded read of CMakePresets.json; the first configurePreset name is
    // the deterministic default. Malformed/hostile -> "default" is NOT
    // guessed: fall back to a plain configure (empty preset list = no
    // presets usable). The caller's spec builder handles absence.
    let path = root.join("CMakePresets.json");
    let Ok(bytes) = std::fs::read(&path) else {
        return String::new();
    };
    if bytes.len() as u64 > MAX_MANIFEST_READ {
        return String::new();
    }
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return String::new();
    };
    v.pointer("/configurePresets/0/name")
        .and_then(|n| n.as_str())
        .unwrap_or_default()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(root: &Path, deadline: Duration) -> VerificationContext {
        VerificationContext {
            session_id: 1,
            task_id: 1,
            operation_id: 1,
            workspace_id: 1,
            worktree_id: 1,
            root: root.to_path_buf(),
            deadline: Instant::now() + deadline,
            cancellation: CancellationToken::new(),
        }
    }

    fn sh_spec(id: &str, script: &str) -> CheckSpec {
        CheckSpec::new(
            id,
            CheckKind::Test,
            CheckCategory::Quick,
            "/bin/sh",
            ["-c", script],
            true,
        )
    }

    #[tokio::test]
    async fn passing_check_reports_passed_exit_zero() {
        let dir = tempfile::tempdir().unwrap();
        let ex = AsyncCheckExecutor::default();
        let c = ctx(dir.path(), Duration::from_secs(30));
        let out = ex.run_check(&sh_spec("ok", "exit 0"), &c).await.unwrap();
        assert_eq!(out.status, CheckRunStatus::Passed);
        assert_eq!(out.exit, Some(0));
    }

    #[tokio::test]
    async fn failing_check_reports_failed_with_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        let ex = AsyncCheckExecutor::default();
        let c = ctx(dir.path(), Duration::from_secs(30));
        let out = ex
            .run_check(&sh_spec("bad", "echo boom; exit 7"), &c)
            .await
            .unwrap();
        assert_eq!(out.status, CheckRunStatus::Failed);
        assert_eq!(out.exit, Some(7));
        let summary = out.summary.unwrap_or_default();
        assert!(summary.contains("boom"), "{summary}");
    }

    #[tokio::test]
    async fn missing_program_is_unavailable_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let ex = AsyncCheckExecutor::default();
        let c = ctx(dir.path(), Duration::from_secs(30));
        let spec = CheckSpec::new(
            "missing",
            CheckKind::Compile,
            CheckCategory::Quick,
            "/nonexistent-tool-xyz",
            [] as [&str; 0],
            true,
        );
        let out = ex.run_check(&spec, &c).await.unwrap();
        assert_eq!(out.status, CheckRunStatus::Unavailable);
        assert!(out.summary.unwrap_or_default().contains("not found"));
    }

    #[tokio::test]
    async fn deadline_kills_the_whole_process_group_no_orphans() {
        let dir = tempfile::tempdir().unwrap();
        let ex = AsyncCheckExecutor::default();
        let c = ctx(dir.path(), Duration::from_millis(400));
        // The check records its own pid AND spawns a grandchild recording
        // its pid, then waits forever. Both must die with the group.
        let script = "echo $$ > leader.pid; sleep 60 & echo $! > grand.pid; wait";
        let out = ex.run_check(&sh_spec("slow", script), &c).await.unwrap();
        assert_eq!(out.status, CheckRunStatus::Unavailable);
        assert!(out.exit.is_none());
        let leader: i32 = std::fs::read_to_string(dir.path().join("leader.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let grand: i32 = std::fs::read_to_string(dir.path().join("grand.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // The group was SIGKILLed: both pids vanish (retry briefly for reap).
        let deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let l = (unsafe { libc::kill(leader, 0) }) == 0;
            let g = (unsafe { libc::kill(grand, 0) }) == 0;
            if !l && !g {
                break;
            }
            assert!(Instant::now() < deadline, "orphans: leader={l} grand={g}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[tokio::test]
    async fn cancellation_stops_a_running_check() {
        let dir = tempfile::tempdir().unwrap();
        let ex = AsyncCheckExecutor::default();
        let c = ctx(dir.path(), Duration::from_secs(60));
        let cancel = c.cancellation.clone();
        let handle = tokio::spawn(async move {
            let spec = sh_spec("cancel-me", "sleep 60");
            ex.run_check(&spec, &c).await.unwrap()
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        cancel.cancel();
        let out = handle.await.unwrap();
        assert_eq!(out.status, CheckRunStatus::Unavailable);
        assert_eq!(out.summary.as_deref(), Some("cancelled"));
    }

    #[tokio::test]
    async fn huge_output_is_capped_not_deadlocked() {
        let dir = tempfile::tempdir().unwrap();
        let ex = AsyncCheckExecutor::default().with_output_cap(64 * 1024);
        let c = ctx(dir.path(), Duration::from_secs(30));
        let out = ex
            .run_check(
                &sh_spec("noisy", "dd if=/dev/zero bs=1M count=4 2>/dev/null"),
                &c,
            )
            .await
            .unwrap();
        assert_eq!(out.status, CheckRunStatus::Passed, "{out:?}");
        assert!(out.truncated);
        assert!(out.summary.unwrap_or_default().len() < 16 * 1024);
    }

    #[tokio::test]
    async fn hostile_cwd_values_are_rejected_before_spawn() {
        let dir = tempfile::tempdir().unwrap();
        let ex = AsyncCheckExecutor::default();
        let c = ctx(dir.path(), Duration::from_secs(30));
        let mut spec = sh_spec("escape", "exit 0");
        spec.cwd_rel = PathBuf::from("..");
        assert!(ex
            .run_check(&spec, &c)
            .await
            .unwrap_err()
            .message
            .contains("escapes"));
        spec.cwd_rel = PathBuf::from("/etc");
        let err = ex.run_check(&spec, &c).await.unwrap_err();
        assert!(err.message.contains("absolute"), "{err:?}");
    }

    // ------------------------------------------------------------------ budget

    #[test]
    fn budget_matrix_never_uses_a_universal_ten_second_cap() {
        let p = VerificationPolicy::default();
        // Quick checks get the 60 s class budget (or the remaining turn).
        assert_eq!(
            budget_for(CheckCategory::Quick, &p, None),
            BudgetDecision::RunInline(Duration::from_secs(60))
        );
        assert_eq!(
            budget_for(CheckCategory::Quick, &p, Some(Duration::from_secs(10))),
            BudgetDecision::RunInline(Duration::from_secs(10))
        );
        // Unit checks may use the remaining turn budget up to the cap.
        assert_eq!(
            budget_for(CheckCategory::Unit, &p, Some(Duration::from_secs(300))),
            BudgetDecision::RunInline(Duration::from_secs(300))
        );
        // Full checks go background by default.
        assert_eq!(
            budget_for(CheckCategory::Full, &p, None),
            BudgetDecision::RunAsTaskOwnedOperation
        );
        // Below the inline floor nothing runs inline.
        assert_eq!(
            budget_for(CheckCategory::Unit, &p, Some(Duration::from_millis(100))),
            BudgetDecision::RunAsTaskOwnedOperation
        );
        // A disabled policy fails closed (background, never a hang).
        let off = VerificationPolicy::disabled();
        assert_eq!(
            budget_for(CheckCategory::Quick, &off, None),
            BudgetDecision::RunAsTaskOwnedOperation
        );
    }

    // ---------------------------------------------------- typed derivation

    fn fixture(files: &[(&str, &str)]) -> (tempfile::TempDir, Vec<String>) {
        let dir = tempfile::tempdir().unwrap();
        let mut names = Vec::new();
        for (name, content) in files {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, content).unwrap();
            names.push(name.to_string());
        }
        (dir, names)
    }

    #[test]
    fn cmake_derives_configure_build_and_ctest_when_tests_exist() {
        let (dir, names) = fixture(&[
            ("CMakeLists.txt", "cmake_minimum_required(VERSION 3.16)\nproject(x C)\nadd_executable(x main.c)\ninclude(CTest)\n"),
            ("src/main.c", "int main(void){return 0;}\n"),
            ("tests/CMakeLists.txt", "add_test(NAME t COMMAND x)\n"),
        ]);
        let checks = derive_typed_checks(dir.path(), &names);
        let ids: Vec<&str> = checks.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"cmake_configure"), "{ids:?}");
        assert!(ids.contains(&"cmake_build"), "{ids:?}");
        assert!(ids.contains(&"cmake_ctest"), "{ids:?}");
        assert!(checks.iter().all(|c| c.required));
    }

    #[test]
    fn cmake_presets_are_used_when_readable() {
        let preset = r#"{"configurePresets":[{"name":"dev","generator":"Unix Makefiles"}]}"#;
        let (dir, names) = fixture(&[
            (
                "CMakeLists.txt",
                "cmake_minimum_required(VERSION 3.16)\nproject(x)\n",
            ),
            ("CMakePresets.json", preset),
            ("src/main.c", "int main(void){return 0;}\n"),
        ]);
        let checks = derive_typed_checks(dir.path(), &names);
        let configure = checks
            .iter()
            .find(|c| c.id == "cmake_configure")
            .expect("configure check");
        assert_eq!(configure.args[0], "--preset");
        assert_eq!(configure.args[1], "dev");
        // A hostile oversized preset file yields plain configure, never a
        // guessed preset and never a panic.
        let (dir2, names2) = fixture(&[
            (
                "CMakeLists.txt",
                "cmake_minimum_required(VERSION 3.16)\nproject(x)\n",
            ),
            ("CMakePresets.json", &"x".repeat(300 * 1024)),
            ("src/main.c", "int main(void){return 0;}\n"),
        ]);
        let checks2 = derive_typed_checks(dir2.path(), &names2);
        let configure2 = checks2
            .iter()
            .find(|c| c.id == "cmake_configure")
            .expect("configure check 2");
        assert_eq!(configure2.args[0], "-S");
    }

    #[test]
    fn makefile_test_target_is_probed_not_guessed() {
        let (dir, names) = fixture(&[
            ("Makefile", "all:\n\ttrue\ntest:\n\ttrue\n"),
            ("main.c", "int main(void){return 0;}\n"),
        ]);
        let checks = derive_typed_checks(dir.path(), &names);
        let ids: Vec<&str> = checks.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"make_build"), "{ids:?}");
        assert!(ids.contains(&"make_test"), "{ids:?}");
        // A Makefile WITHOUT a test target must not derive a required test.
        let (dir2, names2) = fixture(&[("Makefile", "all:\n\ttrue\n"), ("main.c", "x\n")]);
        let checks2 = derive_typed_checks(dir2.path(), &names2);
        assert!(!checks2.iter().any(|c| c.id == "make_test"), "{checks2:?}");
        // A huge hostile Makefile skips the probe (no partial parse truth).
        let (dir3, names3) = fixture(&[
            (
                "Makefile",
                &format!("all:\n\ttrue\n# {}\n", "y".repeat(300 * 1024)),
            ),
            ("main.c", "x\n"),
        ]);
        let checks3 = derive_typed_checks(dir3.path(), &names3);
        assert!(checks3.iter().any(|c| c.id == "make_build"), "{checks3:?}");
        assert!(!checks3.iter().any(|c| c.id == "make_test"));
    }

    #[test]
    fn dotnet_and_gradle_derive_when_wrapper_or_test_project_known() {
        let (dir, names) = fixture(&[
            ("x.sln", ""),
            ("App/App.csproj", "<Project/>"),
            ("App.Tests/App.Tests.csproj", "<Project/>"),
            ("App/Program.cs", "class P {}"),
        ]);
        let checks = derive_typed_checks(dir.path(), &names);
        let ids: Vec<&str> = checks.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"dotnet_build"), "{ids:?}");
        assert!(ids.contains(&"dotnet_test"), "{ids:?}");

        let (dir2, names2) = fixture(&[
            ("gradlew", "#!/bin/sh\nexit 0\n"),
            ("build.gradle", "task classes {}\n"),
            ("src/main/java/A.java", "class A {}\n"),
        ]);
        let checks2 = derive_typed_checks(dir2.path(), &names2);
        let ids2: Vec<&str> = checks2.iter().map(|c| c.id.as_str()).collect();
        assert!(ids2.contains(&"gradle_classes"), "{ids2:?}");
    }

    #[test]
    fn unknown_repos_derive_nothing_without_error() {
        let (dir, names) = fixture(&[("x.zig", "x"), ("README.md", "x")]);
        assert!(derive_typed_checks(dir.path(), &names).is_empty());
    }
}
