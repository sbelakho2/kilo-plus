//! Structured completion-review support (audit round 15: P0-12/80 + P0-13).
//!
//! Pure, deterministic, bounded building blocks for the independent
//! completion review. This module owns NO I/O and NO model calls: the agent
//! runtime assembles the before/after file evidence (checkpoint/CAS base),
//! computes hunks through the bounded diff engine, and feeds this module's
//! pure functions, which classify file statuses, compute the test-inventory
//! delta, assess change risk and render the bounded [`ReviewPackage`] a
//! skeptical review model receives.
//!
//! Everything here is bounded and hostile-safe:
//! - the package target is [`REVIEW_PACKAGE_MAX_BYTES`] (64 KiB); a package
//!   beyond it is a hard [`PackageError::Oversized`] — never a silent
//!   truncation, never a partial review;
//! - per-side file content is capped at [`REVIEW_SIDE_MAX_BYTES`] (the diff
//!   engine's own byte bound — a bigger side cannot be diffed honestly);
//! - every scan the inventory heuristic performs is capped and documented as
//!   a heuristic, never proof;
//! - review-model output is parsed by [`parse_review_verdict`] into the
//!   typed [`ReviewVerdict`]; anything that is not exactly that shape is
//!   rejected loudly.
//!
//! The file statuses here are derived from per-change EXISTENCE + content
//! hashes (checkpoint semantics), not from git: statuses therefore never
//! include "untracked" from real flows (no git diff API exists in the
//! workspace — `crates/git` manages worktrees only). [`FileChangeStatus`]
//! still carries `Untracked` for schema completeness; it is documented as
//! unreachable from checkpoint-derived evidence.

use serde::{Deserialize, Serialize};

/// The review package size target: a package that renders to more bytes than
/// this is refused with a hard error — never silently truncated.
pub const REVIEW_PACKAGE_MAX_BYTES: usize = 64 * 1024;

/// Per-side content bound for a diffed file. Equal to the diff engine's own
/// byte bound (`faktor_edit::diff::MAX_DIFF_BYTES`): a side larger than this
/// cannot produce honest line hunks, so the whole review is refused instead
/// of reviewing partial content.
pub const REVIEW_SIDE_MAX_BYTES: usize = 512 * 1024;

/// A review package may carry at most this many changed files. A change set
/// beyond it is a hard error (a review that silently dropped files would
/// repeat the shallow-reviewer flaw at the file-count axis).
pub const REVIEW_MAX_CHANGED_FILES: usize = 64;

/// Per-file cap on diff hunks (a file whose real diff has more hunks is
/// oversized — never truncated).
pub const REVIEW_MAX_HUNKS_PER_FILE: usize = 64;

/// Per-file cap on the diff lines of one hunk set (parity with the diff
/// engine's render bound). A hunk set beyond it is oversized.
pub const REVIEW_MAX_DIFF_LINES_PER_FILE: usize = 2000;

/// Per-hunk cap on line count (the diff engine's context + change groups are
/// far below this; a hostile "one giant replacement" hunk is capped here).
pub const REVIEW_MAX_LINES_PER_HUNK: usize = 2000;

/// Bounded criteria entries the package carries (goal + derived required
/// checks; a task with more entries than this cannot be fully transcribed —
/// the package is refused rather than dropping criteria).
pub const REVIEW_MAX_CRITERIA_ENTRIES: usize = 8;

/// Bounded check-result rows the package carries.
pub const REVIEW_MAX_CHECK_RESULTS: usize = 6;

/// Review-model findings bound: at most this many findings, each at most
/// [`REVIEW_FINDING_MAX_CHARS`] long. Extra findings are dropped (the
/// boundary is the model's own output bound, not a content decision).
pub const REVIEW_MAX_FINDINGS: usize = 8;

/// One review-model finding length bound.
pub const REVIEW_FINDING_MAX_CHARS: usize = 240;

/// Test-inventory heuristic caps (documented; the assertion delta is a
/// bounded count heuristic and is NEVER treated as proof).
pub const REVIEW_INVENTORY_SCAN_LINES_PER_FILE: usize = 2000;
pub const REVIEW_INVENTORY_DISABLED_MAX: usize = 16;
pub const REVIEW_ASSERTION_DELTA_CAP: i64 = 10_000;
/// Disabled/skipped-test annotation markers scanned per ADDED diff line of a
/// changed test file (trimmed line collected as the annotation). Kept as a
/// literal, extension-agnostic, documented set — never a regex.
pub const REVIEW_DISABLE_MARKERS: &[&str] = &[
    "#[ignore",
    "@pytest.mark.skip",
    "pytestmark",
    ".skip",
    "xit(",
    "xdescribe(",
    "xtest",
    "@Disabled",
    "@Ignore",
    "t.Skip(",
    "test.todo",
];

/// The diff line kind (mirrors the diff engine's line kinds as pure data).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffKind {
    Added,
    Removed,
    Context,
}

/// One line of a hunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffLine {
    pub kind: DiffKind,
    pub text: String,
}

impl DiffLine {
    pub fn added(text: impl Into<String>) -> Self {
        Self {
            kind: DiffKind::Added,
            text: text.into(),
        }
    }
    pub fn removed(text: impl Into<String>) -> Self {
        Self {
            kind: DiffKind::Removed,
            text: text.into(),
        }
    }
}

/// One changed-region hunk of one file: 1-based old/new line ranges plus the
/// bounded line list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hunk {
    pub path: String,
    /// 1-based line where the hunk's removed/context range starts in the
    /// BEFORE file; 0 when the hunk is pure context of an empty before.
    pub old_start: usize,
    pub old_count: usize,
    /// 1-based line where the hunk's added/context range starts in the AFTER
    /// file.
    pub new_start: usize,
    pub new_count: usize,
    pub lines: Vec<DiffLine>,
}

/// File change status derived from a checkpoint-state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileChangeStatus {
    /// exists:false -> exists:true with content.
    Added,
    /// exists:true -> exists:true with different content.
    Modified,
    /// exists:true -> exists:false.
    Deleted,
    /// A deleted path whose last content equals an added path's new content
    /// (content-equal move detected from hashes).
    Renamed,
    /// Schema completeness only: no git diff API exists in the workspace, so
    /// checkpoint-derived evidence can never prove a file is untracked.
    Untracked,
}

impl FileChangeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            FileChangeStatus::Added => "added",
            FileChangeStatus::Modified => "modified",
            FileChangeStatus::Deleted => "deleted",
            FileChangeStatus::Renamed => "renamed",
            FileChangeStatus::Untracked => "untracked",
        }
    }
}

/// One changed file's status row for the review package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileStatus {
    pub path: String,
    pub status: FileChangeStatus,
    /// `Some` when this status is a content-equal move from another path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub renamed_from: Option<String>,
    /// `Some` when this status is a content-equal move to another path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub renamed_to: Option<String>,
    /// Before-side byte count when the before content was available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_before: Option<u64>,
    /// After-side byte count when the after content was available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_after: Option<u64>,
}

/// Per-change input to [`classify_file_statuses`]: existence + content
/// hashes exactly as the checkpoint rows carry them. Hashes are hex strings
/// (the empty string means "no content" on a missing side).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileChange {
    pub path: String,
    pub before_exists: bool,
    pub before_hash_hex: Option<String>,
    pub after_exists: bool,
    pub after_hash_hex: Option<String>,
}

/// One executed-or-derived check row the review model sees.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckResult {
    pub id: String,
    /// `not_run` at review time (checks run after the review decision) or a
    /// later `passed`/`failed`/`unavailable` when a caller re-renders.
    pub status: String,
    pub summary: String,
}

/// The bounded structured diff package a skeptical review model receives
/// (P0-12/80): acceptance criteria + per-file hunks + statuses + the derived
/// test/CI/deletion index lists + check results. Everything a reviewer needs
/// and NOTHING else — never the implementation transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewPackage {
    pub criteria: Vec<String>,
    pub changed_hunks: Vec<Hunk>,
    pub file_statuses: Vec<FileStatus>,
    /// Changed paths that look like test files (added/modified/renamed-to
    /// only; deletions ride `deleted_tests`).
    pub test_files_changed: Vec<String>,
    /// Changed CI build/workflow definition paths.
    pub ci_build_files_changed: Vec<String>,
    /// Changed test files whose change is a deletion (including a rename
    /// whose target is not a test file).
    pub deleted_tests: Vec<String>,
    pub check_results: Vec<CheckResult>,
}

/// Why a review package could not be built/render — every failure is a HARD
/// refusal (never a silent truncation, never a partial review).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackageError {
    /// The rendered package exceeds [`REVIEW_PACKAGE_MAX_BYTES`].
    Oversized { bytes: usize },
    /// The change set exceeds [`REVIEW_MAX_CHANGED_FILES`].
    TooManyFiles { files: usize },
    /// One file side exceeds [`REVIEW_SIDE_MAX_BYTES`] or the diff engine's
    /// honest-diff bounds: no line hunks are possible.
    FileSideTooLarge { path: String, side: &'static str },
    /// One file's hunk set exceeds the per-file caps.
    HunksTooLarge { path: String },
    /// The criteria list exceeds [`REVIEW_MAX_CRITERIA_ENTRIES`].
    CriteriaTooMany { entries: usize },
}

impl std::fmt::Display for PackageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PackageError::Oversized { bytes } => write!(
                f,
                "review package of {bytes} bytes exceeds the {REVIEW_PACKAGE_MAX_BYTES} byte bound"
            ),
            PackageError::TooManyFiles { files } => write!(
                f,
                "review package of {files} changed files exceeds the {REVIEW_MAX_CHANGED_FILES} file bound"
            ),
            PackageError::FileSideTooLarge { path, side } => write!(
                f,
                "review package: {side} side of {path} exceeds the {REVIEW_SIDE_MAX_BYTES} byte diff bound"
            ),
            PackageError::HunksTooLarge { path } => write!(
                f,
                "review package: hunk set of {path} exceeds the per-file diff bound"
            ),
            PackageError::CriteriaTooMany { entries } => write!(
                f,
                "review package: {entries} criteria entries exceed the {REVIEW_MAX_CRITERIA_ENTRIES} bound"
            ),
        }
    }
}

/// True when the path looks like a test file (same deterministic rule the
/// head-signal scan uses): a path component token is one of
/// test/tests/spec/specs (word-boundaried — "contest" never matches).
pub fn path_is_test(path: &str) -> bool {
    path.split(|c: char| !c.is_alphanumeric())
        .map(|s| s.to_lowercase())
        .any(|s| matches!(s.as_str(), "test" | "tests" | "spec" | "specs"))
}

/// True when the path is a CI build/workflow definition: a workflow file
/// under a `.github/workflows` component, or one of the well-known
/// root-level CI definition basenames.
pub fn path_is_ci_build(path: &str) -> bool {
    let comps: Vec<&str> = path.split('/').collect();
    let workflows = comps
        .windows(2)
        .any(|w| w[0] == ".github" && w[1] == "workflows")
        && comps
            .last()
            .is_some_and(|b| b.ends_with(".yml") || b.ends_with(".yaml"));
    let circle = comps
        .windows(2)
        .any(|w| w[0] == ".circleci" && w[1] == "config.yml");
    let known = comps.last().is_some_and(|b| {
        matches!(
            *b,
            "Jenkinsfile"
                | "buildspec.yml"
                | "azure-pipelines.yml"
                | "bitbucket-pipelines.yml"
                | ".travis.yml"
                | ".gitlab-ci.yml"
                | "azure-pipelines.yaml"
                | ".gitlab-ci.yaml"
        )
    });
    workflows || circle || known
}

/// Deterministic path/extension/topic risk classification (P0-13): HIGH when
/// any changed path carries a risky token/manifest/workflow shape or when a
/// test file was deleted. The map is a fixed literal table — provider/model
/// agnostic, never configurable at runtime.
pub fn assess_change_risk(changed: &[String], deleted_tests: &[String]) -> RiskAssessment {
    let mut reasons: Vec<RiskReason> = Vec::new();
    for path in changed {
        if let Some(token) = risky_path_token(path) {
            reasons.push(RiskReason {
                path: path.clone(),
                rule: format!("path token {token:?} names a high-impact topic"),
            });
        } else if let Some(rule) = risky_shape(path) {
            reasons.push(RiskReason {
                path: path.clone(),
                rule: rule.to_string(),
            });
        }
    }
    for path in deleted_tests {
        reasons.push(RiskReason {
            path: path.clone(),
            rule: "a test file was deleted".into(),
        });
    }
    reasons.sort_by(|a, b| a.path.cmp(&b.path).then_with(|| a.rule.cmp(&b.rule)));
    reasons.dedup();
    RiskAssessment {
        level: if reasons.is_empty() {
            RiskLevel::Low
        } else {
            RiskLevel::High
        },
        reasons,
    }
}

/// Path-component tokens that mark a change as high-impact. Deliberately a
/// fixed literal table (documented): security/unsafe/FFI/process/budget/
/// routing/migration/persistence topics plus the workspace's own
/// persistence/authority crates (cas/store/session/checkpoint/snapshot/
/// ledger/router/budget/sandbox).
const RISKY_PATH_TOKENS: &[&str] = &[
    "auth",
    "authorization",
    "budget",
    "billing",
    "capability",
    "cas",
    "cert",
    "certificate",
    "checkpoint",
    "cipher",
    "crypto",
    "credential",
    "database",
    "decrypt",
    "encrypt",
    "exec",
    "ffi",
    "firewall",
    "kernel",
    "keyring",
    "ledger",
    "migrate",
    "migration",
    "migrations",
    "network",
    "password",
    "permission",
    "persist",
    "persistence",
    "process",
    "pty",
    "router",
    "routing",
    "sandbox",
    "schema",
    "secret",
    "security",
    "session",
    "shell",
    "snapshot",
    "socket",
    "spawn",
    "store",
    "sudo",
    "syscall",
    "terminal",
    "tls",
    "token",
    "unsafe",
];

/// Manifest/workflow/secret basenames and extensions that mark a change as
/// high-impact regardless of tokens.
fn risky_shape(path: &str) -> Option<&'static str> {
    let base = path.rsplit('/').next().unwrap_or(path);
    if base == "Cargo.toml"
        || base == "Cargo.lock"
        || base == "package.json"
        || base == "package-lock.json"
        || base == "yarn.lock"
        || base == "pnpm-lock.yaml"
        || base == "go.mod"
        || base == "go.sum"
        || base == "pom.xml"
        || base == "build.gradle"
        || base == "Dockerfile"
        || base == "docker-compose.yml"
    {
        return Some("manifest/build-definition change");
    }
    if path_is_ci_build(path) {
        return Some("CI workflow change");
    }
    for ext in [".pem", ".key", ".p12", ".pfx", ".jks", ".env", ".crt"] {
        if path.ends_with(ext) {
            return Some("secret/credential material");
        }
    }
    None
}

/// The first risky path-component token of `path`, lowercased.
fn risky_path_token(path: &str) -> Option<&'static str> {
    for tok in path
        .split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
    {
        if let Some(risky) = RISKY_PATH_TOKENS.iter().find(|r| **r == tok) {
            return Some(risky);
        }
    }
    None
}

/// The risk level of one turn's change set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    High,
}

/// One deterministic reason a change set is high-risk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskReason {
    pub path: String,
    pub rule: String,
}

/// The deterministic risk assessment of a change set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RiskAssessment {
    pub level: RiskLevel,
    pub reasons: Vec<RiskReason>,
}

/// One disabled/skipped-test annotation found in an ADDED diff line of a
/// changed test file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DisabledTest {
    pub path: String,
    pub annotation: String,
}

/// The deterministic before/after test inventory of one change set (P0-80):
/// structured findings fed to the reviewer alongside the hunk text — never
/// only hunk prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestInventoryDelta {
    pub removed_tests: Vec<String>,
    pub added_tests: Vec<String>,
    pub disabled_or_skipped: Vec<DisabledTest>,
    /// Bounded count heuristic (added minus removed assert/expect tokens over
    /// the changed test files' diff lines, capped per file). A heuristic —
    /// NEVER treated as proof — documented as such on the wire.
    pub assertion_delta: i64,
    pub ci_workflow_changes: Vec<String>,
}

/// Deleted test-file paths of a status list, rename-aware: a rename whose
/// target is NOT a test path counts as a test deletion; a test→test rename
/// is a move and does not.
pub fn deleted_test_paths(statuses: &[FileStatus]) -> Vec<String> {
    let mut out = Vec::new();
    for s in statuses {
        if s.status == FileChangeStatus::Deleted && path_is_test(&s.path) {
            out.push(s.path.clone());
        } else if s.status == FileChangeStatus::Renamed {
            // Only the source side of the move carries renamed_to; a test
            // moved to a NON-test path is a suite deletion, a test->test
            // move is not.
            if let Some(to) = &s.renamed_to {
                if path_is_test(&s.path) && !path_is_test(to) {
                    out.push(s.path.clone());
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Changed paths that are test files and not deletions.
pub fn test_files_changed(statuses: &[FileStatus]) -> Vec<String> {
    let mut out = Vec::new();
    for s in statuses {
        if path_is_test(&s.path)
            && !matches!(
                s.status,
                FileChangeStatus::Deleted | FileChangeStatus::Renamed
            )
        {
            out.push(s.path.clone());
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Classify per-change statuses (existence transitions + content hashes) and
/// surface content-equal renames across the change set. Deterministic:
/// statuses follow the changed-list order, rename pairing is stable.
pub fn classify_file_statuses(changes: &[FileChange]) -> Vec<FileStatus> {
    let mut statuses: Vec<FileStatus> = Vec::with_capacity(changes.len());
    for c in changes {
        let status = match (c.before_exists, c.after_exists) {
            (false, true) => FileChangeStatus::Added,
            (true, false) => FileChangeStatus::Deleted,
            (true, true) => FileChangeStatus::Modified,
            // No recorded transition either way: the row pair records no
            // change; keep it Modified (a no-op is not a change list member).
            _ => FileChangeStatus::Modified,
        };
        statuses.push(FileStatus {
            path: c.path.clone(),
            status,
            renamed_from: None,
            renamed_to: None,
            bytes_before: None,
            bytes_after: None,
        });
    }
    // Content-equal rename detection from hashes: a Deleted path whose
    // before-content hash equals an Added path's after-content hash is a
    // move. (A second Deleted/Added pair with the same content is the same
    // detection — deduping below keeps the first pair only.)
    let deleted: Vec<(usize, &FileChange)> = changes
        .iter()
        .enumerate()
        .filter(|(_, c)| c.before_exists && !c.after_exists)
        .collect();
    let added: Vec<(usize, &FileChange)> = changes
        .iter()
        .enumerate()
        .filter(|(_, c)| !c.before_exists && c.after_exists)
        .collect();
    let mut paired: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (di, d) in &deleted {
        let d_hash = d.before_hash_hex.as_deref().unwrap_or_default();
        if d_hash.is_empty() {
            continue;
        }
        if let Some((ai, a)) = added
            .iter()
            .find(|(ai, a)| {
                !paired.contains(ai) && a.after_hash_hex.as_deref() == Some(d_hash) && *ai != *di
            })
            .copied()
        {
            paired.insert(*di);
            paired.insert(ai);
            statuses[*di].status = FileChangeStatus::Renamed;
            statuses[*di].renamed_to = Some(a.path.clone());
            statuses[ai].status = FileChangeStatus::Renamed;
            statuses[ai].renamed_from = Some(d.path.clone());
        }
    }
    statuses
}

/// Assemble the review package over pre-classified inputs. Computes the
/// test/CI/deletion index lists; every bounded list is derived here so the
/// package and the evidence summaries cannot drift.
pub fn build_package(
    criteria: &[String],
    statuses: Vec<FileStatus>,
    hunks: Vec<Hunk>,
    check_results: Vec<CheckResult>,
) -> Result<ReviewPackage, PackageError> {
    if criteria.len() > REVIEW_MAX_CRITERIA_ENTRIES {
        return Err(PackageError::CriteriaTooMany {
            entries: criteria.len(),
        });
    }
    if statuses.len() > REVIEW_MAX_CHANGED_FILES {
        return Err(PackageError::TooManyFiles {
            files: statuses.len(),
        });
    }
    for hunk in &hunks {
        if hunk.lines.len() > REVIEW_MAX_LINES_PER_HUNK {
            return Err(PackageError::HunksTooLarge {
                path: hunk.path.clone(),
            });
        }
    }
    let mut per_file: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut per_file_lines: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    for h in &hunks {
        *per_file.entry(h.path.as_str()).or_default() += 1;
        *per_file_lines.entry(h.path.as_str()).or_default() += h.lines.len();
        if per_file[h.path.as_str()] > REVIEW_MAX_HUNKS_PER_FILE
            || per_file_lines[h.path.as_str()] > REVIEW_MAX_DIFF_LINES_PER_FILE
        {
            return Err(PackageError::HunksTooLarge {
                path: h.path.clone(),
            });
        }
    }
    if check_results.len() > REVIEW_MAX_CHECK_RESULTS {
        // Never silently drop a check row: refuse the package instead.
        return Err(PackageError::TooManyFiles {
            files: check_results.len(),
        });
    }
    let test_files = test_files_changed(&statuses);
    let deleted_tests = deleted_test_paths(&statuses);
    let ci_build_files_changed = ci_build_files_changed(&statuses);
    let package = ReviewPackage {
        criteria: criteria.to_vec(),
        changed_hunks: hunks,
        file_statuses: statuses,
        test_files_changed: test_files,
        ci_build_files_changed,
        deleted_tests,
        check_results,
    };
    Ok(package)
}

/// CI workflow definition paths of a status list.
pub fn ci_build_files_changed(statuses: &[FileStatus]) -> Vec<String> {
    let mut out: Vec<String> = statuses
        .iter()
        .filter(|s| path_is_ci_build(&s.path))
        .map(|s| s.path.clone())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Validate every hard bound and render the package as canonical JSON.
/// `Err(PackageError::Oversized { .. })` beyond the byte bound — a hostile
/// giant diff must never yield a partial package.
pub fn render_package(package: &ReviewPackage) -> Result<String, PackageError> {
    let json = serde_json::to_string(package)
        .map_err(|_| PackageError::Oversized { bytes: usize::MAX })?;
    let bytes = json.len();
    if bytes > REVIEW_PACKAGE_MAX_BYTES {
        return Err(PackageError::Oversized { bytes });
    }
    Ok(json)
}

/// Bounded assertion/disable heuristic over one file's hunk lines. Counts
/// lines (not occurrences) carrying an assertion token on the added side and
/// on the removed side, scanning at most [`REVIEW_INVENTORY_SCAN_LINES_PER_FILE`]
/// lines per side, each line counted at most once. The difference is
/// saturating per file. Documented heuristic — never proof.
fn assertion_delta_of(lines: &[&DiffLine]) -> (usize, usize) {
    let mut added = 0usize;
    let mut removed = 0usize;
    let mut added_scanned = 0usize;
    let mut removed_scanned = 0usize;
    for line in lines {
        if line.kind == DiffKind::Added {
            if added_scanned >= REVIEW_INVENTORY_SCAN_LINES_PER_FILE {
                continue;
            }
            added_scanned += 1;
            if line_assertion_tokens(&line.text) > 0 {
                added = added.saturating_add(1);
            }
        } else if line.kind == DiffKind::Removed {
            if removed_scanned >= REVIEW_INVENTORY_SCAN_LINES_PER_FILE {
                continue;
            }
            removed_scanned += 1;
            if line_assertion_tokens(&line.text) > 0 {
                removed = removed.saturating_add(1);
            }
        }
    }
    (added, removed)
}

/// Assertion-token lines: any whitespace-split token that starts with
/// "assert" or equals/prefixes "expect" (assert_eq!, expect(...), …).
fn line_assertion_tokens(line: &str) -> usize {
    line.split_whitespace()
        .filter(|w| {
            let t = w.trim_start_matches(|c: char| !c.is_alphanumeric());
            t.starts_with("assert") || t.starts_with("expect")
        })
        .count()
}

/// The bounded inventory delta of one change set (P0-80): structured
/// test-suite findings derived from statuses + hunks.
pub fn compute_test_inventory(statuses: &[FileStatus], hunks: &[Hunk]) -> TestInventoryDelta {
    let removed_tests = deleted_test_paths(statuses);
    let mut added_tests: Vec<String> = statuses
        .iter()
        .filter(|s| s.status == FileChangeStatus::Added && path_is_test(&s.path))
        .map(|s| s.path.clone())
        .collect();
    added_tests.sort();
    added_tests.dedup();
    let ci_workflow_changes = ci_build_files_changed(statuses);

    let test_paths: std::collections::HashSet<String> =
        test_files_changed(statuses).into_iter().collect();
    let mut disabled_or_skipped: Vec<DisabledTest> = Vec::new();
    let mut assertion_delta: i64 = 0;
    let mut handled: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for h in hunks {
        if !test_paths.contains(&h.path) {
            continue;
        }
        let key = h.path.as_str();
        if !handled.insert(key) {
            continue;
        }
        let mut delta_file: i64 = 0;
        for line in &h.lines {
            match line.kind {
                DiffKind::Added => {
                    let trimmed = line.text.trim();
                    if REVIEW_DISABLE_MARKERS.iter().any(|m| trimmed.contains(m))
                        && disabled_or_skipped.len() < REVIEW_INVENTORY_DISABLED_MAX
                    {
                        disabled_or_skipped.push(DisabledTest {
                            path: h.path.clone(),
                            annotation: truncated_chars(trimmed, 120).to_string(),
                        });
                    }
                }
                DiffKind::Removed => {}
                DiffKind::Context => {}
            }
        }
        // Hunk line scan is per FILE, not per hunk: assertion counts scan all
        // hunks of the file but each line belongs to exactly one hunk.
        for h2 in hunks.iter().filter(|h2| h2.path == h.path) {
            let lines: Vec<&DiffLine> = h2.lines.iter().collect();
            let (added, removed) = assertion_delta_of(&lines);
            delta_file = delta_file
                .saturating_add(added as i64)
                .saturating_sub(removed as i64);
        }
        delta_file = delta_file.clamp(-REVIEW_ASSERTION_DELTA_CAP, REVIEW_ASSERTION_DELTA_CAP);
        assertion_delta = assertion_delta
            .saturating_add(delta_file)
            .clamp(-REVIEW_ASSERTION_DELTA_CAP, REVIEW_ASSERTION_DELTA_CAP);
    }
    disabled_or_skipped.sort_by(|a, b| {
        a.path
            .cmp(&b.path)
            .then_with(|| a.annotation.cmp(&b.annotation))
    });
    disabled_or_skipped.dedup();
    TestInventoryDelta {
        removed_tests,
        added_tests,
        disabled_or_skipped,
        assertion_delta,
        ci_workflow_changes,
    }
}

fn truncated_chars(s: &str, max: usize) -> &str {
    if s.chars().count() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// The independent reviewer's typed verdict over the package.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewVerdictKind {
    /// No findings: the change is clean.
    Clean,
    /// Advisory concerns: findings listed, nothing blocking.
    Concern,
    /// Blocking findings: the change must not complete as-is.
    Block,
}

/// Typed review-model output (P0-13). Findings are bounded on parse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReviewVerdict {
    pub verdict: ReviewVerdictKind,
    #[serde(default)]
    pub findings: Vec<String>,
}

/// Parse a review model's text into the typed verdict. Tolerant of a single
/// \`\`\`json fenced block; every other shape (prose, missing/unknown verdict,
/// hostile fields, empty output) is `None` — a caller can never mistake an
/// unparseable review for a clean one. Findings are truncated and capped on
/// parse (the model's own output bound, not a content decision).
pub fn parse_review_verdict(text: &str) -> Option<ReviewVerdict> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let body = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|t| t.strip_suffix("```").unwrap_or(t))
        .unwrap_or(trimmed);
    let start = body.find('{')?;
    let end = body.rfind('}')?;
    if end <= start {
        return None;
    }
    let slice = &body[start..=end];
    let parsed: ReviewVerdict = serde_json::from_str(slice).ok()?;
    let mut verdict = parsed;
    for f in &mut verdict.findings {
        *f = truncated_chars(f, REVIEW_FINDING_MAX_CHARS).to_string();
    }
    verdict.findings.truncate(REVIEW_MAX_FINDINGS);
    Some(verdict)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn change(path: &str, before: bool, after: bool) -> FileChange {
        FileChange {
            path: path.into(),
            before_exists: before,
            before_hash_hex: before.then(|| "bb".repeat(32)),
            after_exists: after,
            after_hash_hex: after.then(|| "aa".repeat(32)),
        }
    }

    fn removed(path_text: &str) -> DiffLine {
        DiffLine {
            kind: DiffKind::Removed,
            text: path_text.into(),
        }
    }
    fn added(text: impl Into<String>) -> DiffLine {
        DiffLine {
            kind: DiffKind::Added,
            text: text.into(),
        }
    }
    fn context(text: &str) -> DiffLine {
        DiffLine {
            kind: DiffKind::Context,
            text: text.into(),
        }
    }

    // ------------------------------------------------------- statuses

    #[test]
    fn classify_statuses_added_modified_deleted() {
        let statuses = classify_file_statuses(&[
            change("src/new.rs", false, true),
            change("src/old.rs", true, false),
            change("src/changed.rs", true, true),
            change("README.md", true, true),
        ]);
        // Equal-state rows keep Modified (no transition recorded).
        let by: std::collections::HashMap<_, _> = statuses
            .iter()
            .map(|s| (s.path.as_str(), s.status))
            .collect();
        assert_eq!(by["src/new.rs"], FileChangeStatus::Added);
        assert_eq!(by["src/old.rs"], FileChangeStatus::Deleted);
        assert_eq!(by["src/changed.rs"], FileChangeStatus::Modified);
        assert_eq!(by["README.md"], FileChangeStatus::Modified);
    }

    #[test]
    fn classify_statuses_surfaces_a_content_equal_rename() {
        // A hostile write that "deletes a.rs and adds b.rs" with identical
        // content must surface as a rename, not as a bare delete+add.
        let mut del = change("src/a.rs", true, false);
        del.before_hash_hex = Some("c0".repeat(32));
        let mut add = change("src/b.rs", false, true);
        add.after_hash_hex = Some("c0".repeat(32));
        let statuses = classify_file_statuses(&[del, add]);
        assert_eq!(statuses[0].status, FileChangeStatus::Renamed);
        assert_eq!(statuses[0].renamed_to.as_deref(), Some("src/b.rs"));
        assert_eq!(statuses[1].status, FileChangeStatus::Renamed);
        assert_eq!(statuses[1].renamed_from.as_deref(), Some("src/a.rs"));
    }

    #[test]
    fn classify_statuses_no_false_rename_on_distinct_content() {
        let statuses = classify_file_statuses(&[
            change("src/a.rs", true, false),
            change("src/b.rs", false, true),
        ]);
        assert_eq!(statuses[0].status, FileChangeStatus::Deleted);
        assert_eq!(statuses[1].status, FileChangeStatus::Added);
    }

    // ------------------------------------------------------- package lists

    #[test]
    fn deleted_tests_rename_aware() {
        // tests/old.rs -> src/legacy.rs (a test moved out of the suite) is a
        // test deletion; tests/old.rs -> tests/new.rs is a move, not one.
        let mut moved_out = change("tests/old.rs", true, false);
        moved_out.before_hash_hex = Some("c1".repeat(32));
        let mut target_out = change("src/legacy.rs", false, true);
        target_out.after_hash_hex = Some("c1".repeat(32));

        let mut moved = change("tests/one.rs", true, false);
        moved.before_hash_hex = Some("c2".repeat(32));
        let mut target = change("tests/two.rs", false, true);
        target.after_hash_hex = Some("c2".repeat(32));
        let mut plain_del = change("tests/gone.rs", true, false);
        plain_del.before_hash_hex = Some("c3".repeat(32));

        let statuses = classify_file_statuses(&[moved_out, target_out, moved, target, plain_del]);
        let deleted = deleted_test_paths(&statuses);
        assert_eq!(
            deleted,
            vec!["tests/gone.rs".to_string(), "tests/old.rs".to_string()],
            "a test renamed to a non-test is a deletion; test->test is a move: {deleted:?}"
        );
    }

    #[test]
    fn test_and_ci_list_detection() {
        let statuses = vec![
            FileStatus {
                path: "tests/unit.rs".into(),
                status: FileChangeStatus::Modified,
                renamed_from: None,
                renamed_to: None,
                bytes_before: None,
                bytes_after: None,
            },
            FileStatus {
                path: ".github/workflows/ci.yml".into(),
                status: FileChangeStatus::Modified,
                renamed_from: None,
                renamed_to: None,
                bytes_before: None,
                bytes_after: None,
            },
            FileStatus {
                path: "src/contest.rs".into(),
                status: FileChangeStatus::Modified,
                renamed_from: None,
                renamed_to: None,
                bytes_before: None,
                bytes_after: None,
            },
        ];
        assert_eq!(
            test_files_changed(&statuses),
            vec!["tests/unit.rs".to_string()]
        );
        assert_eq!(
            ci_build_files_changed(&statuses),
            vec![".github/workflows/ci.yml".to_string()]
        );
        assert!(path_is_test("tests/unit.rs"));
        assert!(!path_is_test("src/contest.rs"));
        assert!(path_is_ci_build(".github/workflows/ci.yml"));
        assert!(path_is_ci_build("path/to/.github/workflows/release.yaml"));
        assert!(path_is_ci_build("Jenkinsfile"));
        assert!(!path_is_ci_build("src/ci.yml"));
        assert!(!path_is_ci_build(".github/other/ci.yml"));
    }

    #[test]
    fn package_build_derives_index_lists() {
        let statuses = classify_file_statuses(&[
            change("src/calc.rs", true, true),
            change("tests/calc.rs", true, true),
            change("tests/gone.rs", true, false),
        ]);
        let pkg = build_package(&["goal: x".into()], statuses, vec![], vec![]).unwrap();
        assert_eq!(pkg.test_files_changed, vec!["tests/calc.rs".to_string()]);
        assert_eq!(pkg.deleted_tests, vec!["tests/gone.rs".to_string()]);
        assert_eq!(pkg.file_statuses.len(), 3);
    }

    #[test]
    fn hostile_giant_package_is_oversized_never_partial() {
        // A change whose diff content exceeds the package bound: the builder
        // MUST return a hard Oversized error — no partial package exists.
        let mut lines = Vec::new();
        let mut bytes = 0usize;
        let mut i = 0usize;
        while bytes < REVIEW_PACKAGE_MAX_BYTES + 8 * 1024 {
            let l = format!("+a very long added line {i} {}", "x".repeat(120));
            bytes += l.len();
            lines.push(DiffLine {
                kind: DiffKind::Added,
                text: l,
            });
            i += 1;
        }
        let hunk = Hunk {
            path: "src/hostile.rs".into(),
            old_start: 1,
            old_count: 0,
            new_start: 1,
            new_count: lines.len(),
            lines,
        };
        let statuses = vec![FileStatus {
            path: "src/hostile.rs".into(),
            status: FileChangeStatus::Modified,
            renamed_from: None,
            renamed_to: None,
            bytes_before: None,
            bytes_after: None,
        }];
        let pkg = build_package(&[], statuses, vec![hunk], vec![]).unwrap();
        match render_package(&pkg) {
            Err(PackageError::Oversized { bytes }) => assert!(bytes > REVIEW_PACKAGE_MAX_BYTES),
            other => panic!("hostile package must be a hard Oversized error, got {other:?}"),
        }
        // Per-file hunk caps are hard bounds too.
        let mut too_many = Vec::new();
        for _ in 0..(REVIEW_MAX_HUNKS_PER_FILE + 1) {
            too_many.push(Hunk {
                path: "src/many.rs".into(),
                old_start: 1,
                old_count: 1,
                new_start: 1,
                new_count: 1,
                lines: vec![DiffLine::added("x")],
            });
        }
        let statuses = vec![FileStatus {
            path: "src/many.rs".into(),
            status: FileChangeStatus::Modified,
            renamed_from: None,
            renamed_to: None,
            bytes_before: None,
            bytes_after: None,
        }];
        let err = build_package(&[], statuses, too_many, vec![]).unwrap_err();
        assert!(matches!(err, PackageError::HunksTooLarge { .. }));
        // Too many changed files is a hard error, never a dropped tail.
        let statuses: Vec<FileStatus> = (0..REVIEW_MAX_CHANGED_FILES + 1)
            .map(|i| FileStatus {
                path: format!("src/f{i}.rs"),
                status: FileChangeStatus::Modified,
                renamed_from: None,
                renamed_to: None,
                bytes_before: None,
                bytes_after: None,
            })
            .collect();
        let err = build_package(&[], statuses, vec![], vec![]).unwrap_err();
        assert!(matches!(err, PackageError::TooManyFiles { .. }));
    }

    // ------------------------------------------------------- inventory

    #[test]
    fn inventory_delta_removed_added_disabled_and_ci() {
        let statuses = classify_file_statuses(&[
            change("tests/calc.rs", true, true),
            change("tests/gone.rs", true, false),
            change(".github/workflows/ci.yml", true, true),
        ]);
        let hunks = vec![Hunk {
            path: "tests/calc.rs".into(),
            old_start: 4,
            old_count: 5,
            new_start: 4,
            new_count: 6,
            lines: vec![
                context("fn adds() {"),
                removed("    assert_eq!(add(1, 2), 3);"),
                added("    // TODO: re-enable once stable"),
                added("    #[ignore]"),
                added("    assert_eq!(add(1, 2), 3);"),
                context("}"),
            ],
        }];
        let inventory = compute_test_inventory(&statuses, &hunks);
        assert_eq!(inventory.removed_tests, vec!["tests/gone.rs".to_string()]);
        assert!(inventory.added_tests.is_empty());
        assert_eq!(inventory.ci_workflow_changes.len(), 1);
        assert_eq!(
            inventory.disabled_or_skipped,
            vec![DisabledTest {
                path: "tests/calc.rs".into(),
                annotation: "#[ignore]".into(),
            }],
            "the ADDED #[ignore] must surface with its annotation"
        );
        // one removed assert, one added assert -> net 0 (a count heuristic,
        // never proof — locked here so the heuristic cannot drift silently).
        assert_eq!(inventory.assertion_delta, 0);
    }

    #[test]
    fn inventory_removes_the_todo_marker_does_not_count_as_assertion() {
        let statuses = classify_file_statuses(&[change("tests/calc.rs", true, true)]);
        let hunks = vec![Hunk {
            path: "tests/calc.rs".into(),
            old_start: 1,
            old_count: 2,
            new_start: 1,
            new_count: 1,
            lines: vec![
                removed("assert_eq!(a, 1);"),
                removed("expect(a).toBe(1);"),
                added("// TODO: restore"),
            ],
        }];
        let inventory = compute_test_inventory(&statuses, &hunks);
        assert_eq!(
            inventory.assertion_delta, -2,
            "removals count, prose does not"
        );
        assert!(inventory.disabled_or_skipped.is_empty());
    }

    #[test]
    fn inventory_disabled_cap_bounds_hostile_output() {
        let statuses = classify_file_statuses(&[change("tests/a.rs", true, true)]);
        let mut lines = Vec::new();
        for i in 0..(REVIEW_INVENTORY_DISABLED_MAX + 20) {
            lines.push(added(format!("#[ignore = \"hostile {i}\"]")));
        }
        let hunks = vec![Hunk {
            path: "tests/a.rs".into(),
            old_start: 1,
            old_count: 0,
            new_start: 1,
            new_count: lines.len(),
            lines,
        }];
        let inventory = compute_test_inventory(&statuses, &hunks);
        assert!(inventory.disabled_or_skipped.len() <= REVIEW_INVENTORY_DISABLED_MAX);
    }

    // ------------------------------------------------------- risk

    #[test]
    fn risk_matrix_deterministic_and_bounded() {
        // Low: an ordinary source+test change.
        let low = assess_change_risk(
            &["src/calc.rs".to_string(), "tests/calc.rs".to_string()],
            &[],
        );
        assert_eq!(low.level, RiskLevel::Low);
        assert!(low.reasons.is_empty());

        // High: security/unsafe/FFI/process/budget/routing/migration/
        // persistence topics + crate locations.
        for path in [
            "crates/security/src/auth.rs",
            "src/unsafe_shim.rs",
            "crates/ffi/src/bindings.rs",
            "crates/process/src/supervisor.rs",
            "crates/budget/src/lib.rs",
            "crates/router/src/lib.rs",
            "src/db/migrations/0001.sql",
            "crates/session/src/persistence.rs",
            "crates/cas/src/blob.rs",
            "crates/context/src/ledger.rs",
            "src/keys.pem",
            ".github/workflows/ci.yml",
            "Cargo.toml",
            "Dockerfile",
        ] {
            let risk = assess_change_risk(&[path.to_string()], &[]);
            assert_eq!(risk.level, RiskLevel::High, "{path} must be risky");
            assert!(!risk.reasons.is_empty(), "{path}");
        }
        // Plain crate code in normal crates stays low (no false positives
        // that would burn a review call on every turn). "handler"/"contest"
        // are the no-false-positive guards: they share shape with risky
        // words ("handler" vs sandbox handlers, "contest" vs test) but are
        // not on the map.
        for path in [
            "src/handler.rs",
            "src/contest.rs",
            "crates/agent/src/runtime.rs",
            "crates/context/src/assembler.rs",
            "crates/verify/src/lib.rs",
            "tests/unit.rs",
            "README.md",
            "src/a.rs",
        ] {
            let risk = assess_change_risk(&[path.to_string()], &[]);
            assert_eq!(risk.level, RiskLevel::Low, "{path} must stay low");
        }
        // A deleted test is always risky.
        let del = assess_change_risk(
            &["tests/gone.rs".to_string(), "src/a.rs".to_string()],
            &["tests/gone.rs".to_string()],
        );
        assert_eq!(del.level, RiskLevel::High);
        // Deterministic + deduped.
        let twice = assess_change_risk(
            &[
                "src/a.rs".to_string(),
                "src/unsafe.rs".to_string(),
                "src/unsafe.rs".to_string(),
            ],
            &[],
        );
        assert_eq!(twice.reasons.len(), 1);
        assert!(twice.reasons[0].path == "src/unsafe.rs");
    }

    // ------------------------------------------------------- verdict parse

    #[test]
    fn verdict_parse_accepts_plain_and_fenced_json() {
        let plain = r#"{"verdict":"block","findings":["removed assertions without replacement"]}"#;
        let v = parse_review_verdict(plain).unwrap();
        assert_eq!(v.verdict, ReviewVerdictKind::Block);
        assert_eq!(v.findings.len(), 1);
        let fenced = "```json\n{\"verdict\": \"concern\", \"findings\": [\"x\"]}\n```";
        let v = parse_review_verdict(fenced).unwrap();
        assert_eq!(v.verdict, ReviewVerdictKind::Concern);
        let clean = parse_review_verdict(r#"{"verdict":"clean"}"#).unwrap();
        assert_eq!(clean.verdict, ReviewVerdictKind::Clean);
        assert!(clean.findings.is_empty());
    }

    #[test]
    fn verdict_parse_rejects_hostile_and_untyped_output() {
        // Prose, empty output, unknown verdicts, hostile shapes: never a
        // verdict a caller could mistake for clean.
        for hostile in [
            "this change looks fine to me",
            "",
            "   ",
            "the verdict is clean but I wrote prose",
            r#"{"verdict":"awesome"}"#,
            r#"{"verdict":"block","findings":[42]}"#,
            r#"{"verdict":"block","extra":true}"#,
            "{}",
            r#"{"findings":["no verdict key"]}"#,
            r#"{"verdict":null}"#,
            "```json\n{\"verdict\":\"clean\"\n```",
        ] {
            assert!(
                parse_review_verdict(hostile).is_none(),
                "hostile review output {hostile:?} must parse to None"
            );
        }
    }

    #[test]
    fn verdict_findings_are_bounded_on_parse() {
        let huge = "y".repeat(REVIEW_FINDING_MAX_CHARS + 200);
        let text = format!(r#"{{"verdict":"block","findings":[{huge:?}]}}"#);
        let v = parse_review_verdict(&text).unwrap();
        assert_eq!(v.findings[0].len(), REVIEW_FINDING_MAX_CHARS);
        let many = (0..REVIEW_MAX_FINDINGS + 5)
            .map(|i| format!("finding {i}"))
            .collect::<Vec<_>>();
        let text = format!(r#"{{"verdict":"concern","findings":{many:?}}}"#);
        let v = parse_review_verdict(&text).unwrap();
        assert_eq!(v.findings.len(), REVIEW_MAX_FINDINGS);
    }
}
