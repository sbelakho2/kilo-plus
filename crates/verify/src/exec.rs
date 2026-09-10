//! Typed, async, project-aware verification (P0-9/P0-10/P0-11).
//!
//! This module is the modernization target for the legacy string-command
//! `Verifier`: checks are typed `(program, args)` specs — never `sh -c`
//! unless a repository rule itself specifies a shell command — executed
//! through the workspace's ONE [`faktor_terminal::ProcessSupervisor`] by
//! [`AsyncCheckExecutor`] with no per-check OS thread, no nested runtime and
//! no second process layer (audit P0-5/P0-6). Every check runs against the
//! exact worktree in [`VerificationContext`] (never the daemon's current
//! directory), is bounded by the context deadline and cancellation token,
//! and is killed process-group-wide by the supervisor so no orphan survives.
//!
//! Budgets ([`budget_for`]) are derived from the check category, the
//! remaining turn budget and a [`VerificationPolicy`] — never a universal
//! ten-second wall cap. [`derive_typed_checks`] makes CMake/Make/Meson/
//! Ninja/Bazel/MSBuild/.csproj/Gradle first-class project types instead of
//! resolving to an empty check list.

use std::ffi::{OsStr, OsString};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use faktor_core::cancellation::CancellationToken;
use faktor_core::error::Error;
use faktor_core::id::SessionId;

/// How heavy a check is. Drives the execution budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum CheckKind {
    Compile,
    Test,
    Lint,
}

/// A typed check: program + argv, no shell interpolation. Serde: the durable
/// background-check rows (faktor-session verification jobs) serialize the
/// spec at enqueue and re-parse it when the job runs after a restart.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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

/// Result of one executed check. Serde: job results ride durable
/// verification-job rows (see the CheckSpec note).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// Default wall deadline of one job-driven background check when the policy
/// leaves no explicit job cap (jobs store their own budget at enqueue).
pub const DEFAULT_JOB_BUDGET_MS: u64 = 600_000;

/// Async check executor backed by THE workspace [`ProcessSupervisor`]
/// (audit P0-5/P0-6 consolidation): one process runtime for the whole
/// daemon — verification never spawns its own process layer. The supervisor
/// owns process-group creation, whole-tree kill on deadline/cancellation,
/// bounded capture (ring + CAS spill) and exactly-once reaping.
///
/// `run_check` keeps its signature and [`CheckOutcome`] semantics: a check
/// that cannot run or is killed reports `Unavailable` (never an error of the
/// code under check); only pre-spawn validation failures (a cwd that escapes
/// the verification root, ...) are `Err`.
#[derive(Debug, Clone)]
pub struct AsyncCheckExecutor {
    supervisor: Arc<faktor_terminal::ProcessSupervisor>,
    /// Effective per-stream capture ceiling requested from the supervisor.
    artifact_max: usize,
    /// Network-isolation requirement every check spawn derives its
    /// [`faktor_terminal::NetworkIsolation`] from. The sandbox policy
    /// DECIDES (`Required` → `DenyAll`, `BestEffort`/`None` → `Inherit`);
    /// the daemon wires the requirement here. The default claims nothing
    /// (`Inherit`) and always becomes an explicit spawn mode — never an
    /// accidental hardcoded one.
    network_requirement: faktor_terminal::NetworkIsolationRequirement,
}

impl Default for AsyncCheckExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncCheckExecutor {
    /// The executor over [`ProcessSupervisor::shared`] (the in-process
    /// supervisor; crate-level tests and hosts without a daemon supervisor).
    pub fn new() -> Self {
        Self::from_supervisor(faktor_terminal::ProcessSupervisor::shared())
    }

    /// The executor over an explicit supervisor — the daemon graph wires
    /// its ONE supervisor here so verification shares the process runtime,
    /// its live-child ceiling, its capture ring and its kill paths.
    pub fn from_supervisor(supervisor: Arc<faktor_terminal::ProcessSupervisor>) -> Self {
        Self {
            supervisor,
            artifact_max: OUTPUT_CAP_BYTES,
            network_requirement: faktor_terminal::NetworkIsolationRequirement::Inherit,
        }
    }

    /// Install the sandbox policy's spawn requirement (audit P0-39) for
    /// every check this executor runs: `DenyAll` makes each check spawn
    /// isolated or fail closed typed BEFORE exec; `Inherit` claims nothing.
    pub fn with_network_requirement(
        mut self,
        requirement: faktor_terminal::NetworkIsolationRequirement,
    ) -> Self {
        self.network_requirement = requirement;
        self
    }

    pub fn supervisor(&self) -> &Arc<faktor_terminal::ProcessSupervisor> {
        &self.supervisor
    }

    pub fn with_output_cap(mut self, cap: usize) -> Self {
        self.artifact_max = cap.max(4096);
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
        // Bare name: resolve against PATH. The check env is the
        // [`faktor_terminal::EnvSpec`] toolchain allowlist (PATH + approved
        // toolchain vars), so a bare name spawns like tokio's did.
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

    /// Run one check under the context's deadline and cancellation through
    /// the supervisor. The child runs in its own process group (supervisor
    /// spawn semantics): a deadline/cancellation kill takes grandchildren
    /// too (zero orphans), and the supervisor performs the single reap.
    pub async fn run_check(
        &self,
        spec: &CheckSpec,
        ctx: &VerificationContext,
    ) -> Result<CheckOutcome, Error> {
        let started_ms = now_ms();
        let cwd = Self::validate_cwd_rel(&ctx.root, &spec.cwd_rel)?;
        let program = self.resolve_program(&ctx.root, &spec.program);

        let cfg = faktor_terminal::SpawnConfig {
            cmd: program.to_string_lossy().into_owned(),
            args: spec
                .args
                .iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect(),
            cwd,
            // THE verification environment authority: the toolchain
            // allowlist (PATH + approved vars — HOME, CARGO_HOME,
            // RUSTUP_HOME, CARGO_TARGET_DIR/RUSTFLAGS/RUSTC_WRAPPER, and
            // the platform baseline). The child is env-cleared first, the
            // deny-set drops every configured secret name, and the daemon's
            // full environment never crosses.
            env: faktor_terminal::EnvSpec::toolchain(),
            owner: faktor_terminal::ProcessOwner::Verification(SessionId::new(ctx.session_id)),
            capture: true,
            artifact_max: self.artifact_max,
            network_isolation: faktor_terminal::NetworkIsolation::from(self.network_requirement),
        };
        let deadline = ctx
            .deadline
            .saturating_duration_since(Instant::now())
            .max(Duration::from_millis(1));
        match self
            .supervisor
            .run(cfg.clone(), deadline, ctx.cancellation.clone())
            .await
        {
            Ok(output) => {
                let exit = output.exit_code;
                let run_status = if exit == Some(0) {
                    CheckRunStatus::Passed
                } else if exit.is_some() {
                    CheckRunStatus::Failed
                } else {
                    CheckRunStatus::Unavailable
                };
                // The supervisor's excerpt is the bounded ring tail of the
                // combined output (already capped); re-cap to the outcome's
                // documented tail bound and report truncation when the ring
                // dropped bytes (artifact_truncated) or the excerpt itself
                // was clipped.
                let summary = build_summary_from_text(&output.excerpt);
                Ok(CheckOutcome {
                    status: run_status,
                    exit,
                    started_ms,
                    finished_ms: now_ms(),
                    summary,
                    truncated: output.artifact_truncated
                        || output.excerpt.len() > SUMMARY_MAX_BYTES,
                })
            }
            Err(e) => {
                let summary = if e.kind == faktor_core::error::ErrorKind::Cancelled {
                    "cancelled".to_string()
                } else if e.kind == faktor_core::error::ErrorKind::Timeout {
                    format!(
                        "killed by deadline after {}ms (supervisor group kill); no verdict",
                        deadline.as_millis()
                    )
                } else if e.kind == faktor_core::error::ErrorKind::NotFound {
                    // The supervisor maps every spawn failure to not_found.
                    format!("program not found: {} ({e})", program.display())
                } else {
                    return Err(Error::internal(format!(
                        "verification spawn through the supervisor failed: {e}"
                    )));
                };
                Ok(CheckOutcome {
                    status: CheckRunStatus::Unavailable,
                    exit: None,
                    started_ms,
                    finished_ms: now_ms(),
                    summary: Some(summary),
                    truncated: false,
                })
            }
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Bound the supervisor's combined-output excerpt to the outcome's documented
/// `SUMMARY_MAX_BYTES` tail (the ring already holds the LAST lines).
fn build_summary_from_text(excerpt: &str) -> Option<String> {
    if excerpt.trim().is_empty() {
        return None;
    }
    let tail: String = excerpt
        .chars()
        .rev()
        .take(SUMMARY_MAX_BYTES)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    Some(tail)
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

// ------------------------------------------------------------------ bridge
// Legacy string-command checks -> typed specs (the P0-9/10 migration).
// The legacy derivation arms for Rust/Node/Python/Go/Java remain the
// deterministic source of per-change checks; this bridge re-expresses their
// commands as typed (program, argv) specs with STRICT simple-token rules.
// A command that cannot be tokenized (any shell metacharacter or quote) is
// rejected with a typed reason and NEVER executed through `sh -c` — the
// caller records it unavailable.

/// The typed category a bridged legacy check runs under (P0-10): compile and
/// lint checks are the 30-60 s "Quick" class; tests are normal "Unit"
/// verification that may use the remaining turn budget. Nothing maps to
/// "Full": full-repository verification derives typed specs directly (see
/// [`derive_typed_checks`]), it is never bridged from a shell string.
pub fn category_for_kind(kind: crate::CheckKind) -> CheckCategory {
    match kind {
        crate::CheckKind::Compile | crate::CheckKind::Lint => CheckCategory::Quick,
        crate::CheckKind::Test => CheckCategory::Unit,
    }
}

/// Why one legacy command cannot become a typed spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BridgeRejection {
    /// Stable id of the rejected check (the caller records it under this).
    pub id: String,
    /// The rejected shell command, kept verbatim for durable records.
    pub command: String,
    pub reason: String,
}

/// One legacy check's bridge result, index-aligned with the input list.
#[derive(Debug, Clone)]
pub enum CheckBridge {
    Spec(CheckSpec),
    Rejected(BridgeRejection),
}

/// Strict simple-token bridge from a legacy [`crate::Check`] command to a
/// typed spec. Token rules: the command is split on ASCII whitespace and
/// every token must be a plain word — NO shell metacharacters or quoting
/// (`& | ; < > $ ` \ " ' ( ) { } [ ] * ? ! # ~` and backslash are all
/// rejected). Legacy derivations only ever produce such commands (their
/// filters are single sanitized tokens); a hostile or drifted command is a
/// typed rejection, never a shell string handed to a shell. The typed
/// category comes from [`category_for_kind`]; `cwd_rel` is the workspace
/// root (legacy commands always ran repo-rooted).
pub fn check_to_spec(check: &crate::Check) -> CheckBridge {
    const METACHARS: &[char] = &[
        '&', '|', ';', '<', '>', '$', '`', '\\', '"', '\'', '(', ')', '{', '}', '[', ']', '*', '?',
        '!', '#', '~',
    ];
    let tokens: Vec<&str> = check.command.split_whitespace().collect();
    let reject = |reason: String| {
        CheckBridge::Rejected(BridgeRejection {
            id: check.id.clone(),
            command: check.command.clone(),
            reason,
        })
    };
    let Some((program, args)) = tokens.split_first() else {
        return reject("empty command".into());
    };
    for token in std::iter::once(program).chain(args.iter()) {
        if let Some(bad) = token.chars().find(|c| METACHARS.contains(c)) {
            return reject(format!(
                "token {token:?} carries shell metacharacter {bad:?}; refused (never sh -c)"
            ));
        }
    }
    CheckBridge::Spec(CheckSpec {
        id: check.id.clone(),
        kind: match check.kind {
            crate::CheckKind::Compile => CheckKind::Compile,
            crate::CheckKind::Test => CheckKind::Test,
            crate::CheckKind::Lint => CheckKind::Lint,
        },
        category: category_for_kind(check.kind),
        program: program.into(),
        args: args.iter().map(|a| OsString::from(*a)).collect(),
        cwd_rel: PathBuf::from("."),
        affects: check.affects.clone(),
        required: check.required,
    })
}

/// Bridge a whole legacy check list; one [`CheckBridge`] per input check,
/// index-aligned (callers pair by index or by the stable check id).
pub fn checks_to_specs(checks: &[crate::Check]) -> Vec<CheckBridge> {
    checks.iter().map(check_to_spec).collect()
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

    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_kills_the_whole_supervisor_process_group_no_orphans() {
        // Adversarial (audit P0-5/P0-6 whole-tree semantics): the check
        // spawns a GRANDCHILD and waits forever; cancellation of the turn
        // token must kill the whole process group through the supervisor —
        // leader AND grandchild die (the executor holds no private process
        // layer of its own; the supervisor's kill path is the ONLY one).
        let dir = tempfile::tempdir().unwrap();
        let ex = AsyncCheckExecutor::default();
        let c = ctx(dir.path(), Duration::from_secs(120));
        let cancel = c.cancellation.clone();
        let script = "echo $$ > leader.pid; sleep 60 & echo $! > grand.pid; wait";
        let handle = tokio::spawn(async move {
            let spec = sh_spec("tree-cancel", script);
            ex.run_check(&spec, &c).await.unwrap()
        });
        // Let the script record both pids, then cancel the whole tree.
        tokio::time::sleep(Duration::from_millis(600)).await;
        cancel.cancel();
        let out = handle.await.unwrap();
        assert_eq!(out.status, CheckRunStatus::Unavailable);
        assert_eq!(out.summary.as_deref(), Some("cancelled"));
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

    #[test]
    fn the_executor_contains_no_tokio_process_layer_anymore() {
        // Adversarial source-scan (audit P0-5/P0-6): verification must not
        // spawn its own process runtime — every child of a check rides THE
        // workspace ProcessSupervisor. tokio::process::Command inside
        // crates/verify/src is a regression of the second process layer.
        let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offending = Vec::new();
        for entry in std::fs::read_dir(&src_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            for (i, line) in text.lines().enumerate() {
                let trimmed = line.trim_start();
                if !line.contains("tokio::process") {
                    continue;
                }
                // Comments and this scanner's own probe text are not usage.
                if trimmed.starts_with("//") || line.contains("line.contains") {
                    continue;
                }
                offending.push(format!("{}:{i}: {line}", path.display()));
            }
        }
        assert!(
            offending.is_empty(),
            "crates/verify/src must not spawn its own processes:\n{}",
            offending.join("\n")
        );
    }

    #[tokio::test]
    async fn executor_reports_its_supervisor_owner_for_sessions() {
        // The SpawnConfig owner is ProcessOwner::Verification(session): the
        // daemon can kill exactly the verification children of one session
        // (kill_all_for) without touching its other children.
        let dir = tempfile::tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let supervisor = faktor_terminal::ProcessSupervisor::new(cas);
        let ex = AsyncCheckExecutor::from_supervisor(supervisor.clone());
        assert!(Arc::ptr_eq(ex.supervisor(), &supervisor));
        let c = ctx(dir.path(), Duration::from_secs(30));
        let c2 = c.clone();
        let cancel = c2.cancellation.clone();
        let handle = tokio::spawn(async move {
            let spec = sh_spec("owner", "sleep 60");
            ex.run_check(&spec, &c2).await.unwrap()
        });
        // Wait until the child is registered under the Verification owner.
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let alive = supervisor.alive();
            if alive
                .iter()
                .any(|h| matches!(h.owner, faktor_terminal::ProcessOwner::Verification(s) if s == SessionId::new(c.session_id)))
            {
                break;
            }
            assert!(Instant::now() < deadline, "check never registered");
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        // Session-scoped kill: exactly this session's verification tree dies.
        let killed = supervisor.kill_all_for(faktor_terminal::ProcessOwner::Verification(
            SessionId::new(c.session_id),
        ));
        assert!(!killed.is_empty(), "kill_all_for must find the check");
        let out = handle.await.unwrap();
        assert_eq!(out.status, CheckRunStatus::Unavailable, "{out:?}");
        let _ = cancel;
    }

    #[tokio::test]
    async fn checks_run_with_the_toolchain_allowlist_and_never_see_secrets() {
        // One environment authority: a verification child sees PATH and the
        // approved CARGO_HOME value, never a configured secret name (even
        // when the parent carries it), and never an undeclared daemon var.
        std::env::set_var("CARGO_HOME", "/tmp/kp-verify-cargo");
        std::env::set_var("OPENAI_API_KEY", "sk-verify-secret");
        std::env::set_var("TEST_PRIVATE_SECRET", "private");
        std::env::set_var("KP_VERIFY_UNDECLARED", "must-not-arrive");
        let dir = tempfile::tempdir().unwrap();
        let ex = AsyncCheckExecutor::default();
        let c = ctx(dir.path(), Duration::from_secs(30));
        let spec = CheckSpec::new(
            "env-exact",
            CheckKind::Test,
            CheckCategory::Quick,
            "/bin/sh",
            [
                "-c",
                "test -n \"$PATH\" || exit 11; \
                 test \"$CARGO_HOME\" = /tmp/kp-verify-cargo || exit 12; \
                 test -z \"$OPENAI_API_KEY\" || exit 13; \
                 test -z \"$TEST_PRIVATE_SECRET\" || exit 14; \
                 test -z \"$KP_VERIFY_UNDECLARED\" || exit 15; \
                 echo verify-env-exact",
            ],
            true,
        );
        let out = ex.run_check(&spec, &c).await.unwrap();
        assert_eq!(out.status, CheckRunStatus::Passed, "{out:?}");
        assert!(out.summary.unwrap_or_default().contains("verify-env-exact"));
        std::env::remove_var("CARGO_HOME");
        std::env::remove_var("OPENAI_API_KEY");
        std::env::remove_var("TEST_PRIVATE_SECRET");
        std::env::remove_var("KP_VERIFY_UNDECLARED");
    }

    #[test]
    fn spawn_isolation_is_derived_from_the_policy_requirement() {
        // Every check spawn derives its mode through
        // NetworkIsolation::from(requirement): the policy's DenyAll demand
        // becomes a DenyAll spawn (fail closed typed), the default claims
        // nothing. No hardcoded mode exists in this file.
        let ex = AsyncCheckExecutor::default()
            .with_network_requirement(faktor_terminal::NetworkIsolationRequirement::DenyAll);
        assert_eq!(
            ex.network_requirement,
            faktor_terminal::NetworkIsolationRequirement::DenyAll
        );
        assert_eq!(
            faktor_terminal::NetworkIsolation::from(ex.network_requirement),
            faktor_terminal::NetworkIsolation::DenyAll
        );
        let ex = AsyncCheckExecutor::default();
        assert_eq!(
            ex.network_requirement,
            faktor_terminal::NetworkIsolationRequirement::Inherit
        );
        assert_ne!(
            faktor_terminal::NetworkIsolation::from(ex.network_requirement),
            faktor_terminal::NetworkIsolation::DenyAll,
            "the default claims no OS-level isolation"
        );
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

    // ------------------------------------------------------ legacy bridge

    fn legacy_check(
        id: &str,
        command: &str,
        kind: crate::CheckKind,
        required: bool,
    ) -> crate::Check {
        crate::Check {
            id: id.into(),
            kind,
            command: command.into(),
            affects: vec![],
            required,
        }
    }

    fn spec_of(bridge: &CheckBridge) -> &CheckSpec {
        match bridge {
            CheckBridge::Spec(s) => s,
            CheckBridge::Rejected(r) => panic!("unexpected rejection: {r:?}"),
        }
    }

    fn rejection_of(bridge: &CheckBridge) -> &BridgeRejection {
        match bridge {
            CheckBridge::Rejected(r) => r,
            CheckBridge::Spec(s) => panic!("unexpected spec: {s:?}"),
        }
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn bridge_turns_clean_legacy_commands_into_typed_argv() {
        let cases: Vec<(&str, &str, crate::CheckKind, bool, &str, &[&str])> = vec![
            (
                "rust_check",
                "cargo check",
                crate::CheckKind::Compile,
                true,
                "cargo",
                &["check"],
            ),
            (
                "rust_test:mod_x-2",
                "cargo test mod_x-2",
                crate::CheckKind::Test,
                true,
                "cargo",
                &["test", "mod_x-2"],
            ),
            (
                "rust_test_lib",
                "cargo test --lib",
                crate::CheckKind::Test,
                false,
                "cargo",
                &["test", "--lib"],
            ),
            (
                "node_tsc",
                "npx tsc --noEmit",
                crate::CheckKind::Compile,
                true,
                "npx",
                &["tsc", "--noEmit"],
            ),
            (
                "python_compile:src",
                "python -m compileall -q src",
                crate::CheckKind::Compile,
                true,
                "python",
                &["-m", "compileall", "-q", "src"],
            ),
            (
                "go_test",
                "go test ./...",
                crate::CheckKind::Test,
                false,
                "go",
                &["test", "./..."],
            ),
            (
                "java_compile",
                "mvn -q -DskipTests compile",
                crate::CheckKind::Compile,
                true,
                "mvn",
                &["-q", "-DskipTests", "compile"],
            ),
        ];
        for (id, command, kind, required, program, args) in cases {
            let bridge = check_to_spec(&legacy_check(id, command, kind, required));
            let spec = spec_of(&bridge);
            assert_eq!(spec.id, id);
            assert_eq!(spec.program, OsString::from(program));
            assert_eq!(
                spec.args,
                args.iter().map(|a| OsString::from(*a)).collect::<Vec<_>>()
            );
            assert_eq!(spec.required, required);
            assert_eq!(spec.cwd_rel, PathBuf::from("."));
            assert_eq!(spec.category, category_for_kind(kind));
            assert_eq!(spec.affects, Vec::<String>::new());
        }
    }

    #[test]
    fn bridge_keeps_required_flag_and_affects() {
        let mut check = legacy_check("python_test", "pytest -q", crate::CheckKind::Test, false);
        check.affects = vec!["tests/x.py".into()];
        let bridge = check_to_spec(&check);
        let spec = spec_of(&bridge);
        assert!(!spec.required);
        assert_eq!(spec.affects, vec!["tests/x.py".to_string()]);
    }

    #[test]
    fn bridge_rejects_shell_metacharacters_and_quotes_never_sh_c() {
        // Every hostile shape is a typed rejection naming the metacharacter;
        // a rejected command NEVER runs through `sh -c` (no execution site
        // consumes a rejected bridge).
        for hostile in [
            "cargo check && rm -rf /",
            "cargo check; rm -rf /",
            "cargo check | grep x",
            "npm test > out.txt",
            "echo $HOME",
            "echo `whoami`",
            "echo \"quoted\"",
            "python -m compileall -q 'x; rm'",
            "ls *",
            "touch a b # comment",
            "cmd ~/x",
            "a\\tb",
            "!(true)",
        ] {
            let bridge = check_to_spec(&legacy_check("h", hostile, crate::CheckKind::Test, true));
            let rejection = rejection_of(&bridge);
            assert_eq!(rejection.id, "h");
            assert_eq!(rejection.command, hostile);
            assert!(
                rejection.reason.contains("shell metacharacter"),
                "{hostile:?} -> {rejection:?}"
            );
        }
        // An empty command is also rejected.
        let empty = legacy_check("empty", "   ", crate::CheckKind::Compile, true);
        assert!(rejection_of(&check_to_spec(&empty))
            .reason
            .contains("empty"));
    }

    #[test]
    fn bridge_maps_whole_lists_index_aligned() {
        let checks = vec![
            legacy_check("a", "cargo check", crate::CheckKind::Compile, true),
            legacy_check("b", "cargo test x && evil", crate::CheckKind::Test, true),
            legacy_check("c", "cargo test --lib", crate::CheckKind::Test, false),
        ];
        let bridges = checks_to_specs(&checks);
        assert_eq!(bridges.len(), 3);
        assert_eq!(spec_of(&bridges[0]).id, "a");
        assert_eq!(rejection_of(&bridges[1]).id, "b");
        assert_eq!(spec_of(&bridges[2]).id, "c");
    }
}
