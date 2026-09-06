//! Corpus loading + adversarial validation.
//!
//! Every task directory under `corpus/<task-id>/` holds a CHECKED-IN mini
//! repository plus three benchmark metadata files:
//!
//! - `task.md` — the immutable issue text handed to the model as the prompt;
//! - `criteria.md` — immutable expected criteria in the deterministic
//!   `- crit-NN: <text>` line format (see [`parse_criteria`]);
//! - `verify.sh` — the repository-native verification command the
//!   benchmark runs in a COPY of the workspace after the run.
//!
//! The repo contains a deliberately introduced bug and the passing test
//! suite it breaks: on the pristine copy `verify.sh` must FAIL (non-zero),
//! after the fix it must PASS.

use std::collections::HashSet;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::score::{parse_criteria, Criterion};

/// Relative path of the corpus inside the crate.
pub const CORPUS_DIR: &str = "corpus";

/// Hard caps (bounded everything): hostile or runaway corpora are rejected
/// by the loader before any harness work.
pub const MAX_TASK_FILES: usize = 64;
pub const MAX_CORPUS_FILES: usize = 200;
pub const MAX_CORPUS_BYTES: u64 = 8 * 1024 * 1024;
pub const MAX_TASK_ID_BYTES: usize = 64;
pub const MAX_META_BYTES: u64 = 64 * 1024;
pub const MAX_CRITERIA: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lang {
    Rust,
    C,
    Python,
    Go,
    TypeScript,
    Java,
}

impl Lang {
    /// Directory-name prefix of the corpus task.
    pub fn from_dir_name(name: &str) -> Option<Lang> {
        for (prefix, lang) in [
            ("rust-", Lang::Rust),
            ("c-", Lang::C),
            ("python-", Lang::Python),
            ("go-", Lang::Go),
            ("ts-", Lang::TypeScript),
            ("java-", Lang::Java),
        ] {
            if name.starts_with(prefix) {
                return Some(lang);
            }
        }
        None
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Lang::Rust => "rust",
            Lang::C => "c",
            Lang::Python => "python",
            Lang::Go => "go",
            Lang::TypeScript => "typescript",
            Lang::Java => "java",
        }
    }

    /// The standard-toolchain binaries this task needs on the runner PATH.
    pub const fn toolchain(self) -> &'static [&'static str] {
        match self {
            Lang::Rust => &["cargo"],
            Lang::C => &["cc", "make"],
            Lang::Python => &["python3"],
            Lang::Go => &["go"],
            Lang::TypeScript => &["node"],
            Lang::Java => &["javac"],
        }
    }
}

impl fmt::Display for Lang {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct Task {
    /// `rust-reverse-words`, `c-ringbuf`, … — also the directory name.
    pub id: String,
    pub lang: Lang,
    /// The checked-in task directory (never written by the harness).
    pub dir: PathBuf,
    /// Contents of `task.md` (the immutable issue text / prompt).
    pub task_md: String,
    /// Parsed contents of `criteria.md` (immutable expected criteria).
    pub criteria: Vec<Criterion>,
    /// Contents of `verify.sh` (kept for diagnostics; execution reads the
    /// file from the copy).
    pub verify_sh: String,
    /// Number of files in the task directory (recursive).
    pub file_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorpusError {
    Io { path: String, detail: String },
    InvalidTaskId(String),
    MissingFile { task: String, file: String },
    Oversized { path: String, bytes: u64, cap: u64 },
    EmptyFile { path: String },
    UnreadableText { path: String, detail: String },
    Criteria(String),
    Symlink { path: String },
    UnexpectedEntry { path: String },
    TooManyFiles { total: usize, cap: usize },
}

impl fmt::Display for CorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CorpusError::Io { path, detail } => write!(f, "{path}: {detail}"),
            CorpusError::InvalidTaskId(id) => {
                write!(f, "invalid task id {id:?} (must be [a-z0-9][a-z0-9-]*)")
            }
            CorpusError::MissingFile { task, file } => {
                write!(f, "task {task}: missing required file {file}")
            }
            CorpusError::Oversized { path, bytes, cap } => {
                write!(f, "{path}: {bytes} bytes exceeds the {cap}-byte cap")
            }
            CorpusError::EmptyFile { path } => write!(f, "{path}: empty (or whitespace-only)"),
            CorpusError::UnreadableText { path, detail } => {
                write!(f, "{path}: not valid UTF-8 text: {detail}")
            }
            CorpusError::Criteria(detail) => write!(f, "criteria.md: {detail}"),
            CorpusError::Symlink { path } => {
                write!(
                    f,
                    "{path}: symlink entries are rejected (corpus is plain files only)"
                )
            }
            CorpusError::UnexpectedEntry { path } => {
                write!(f, "{path}: unexpected entry kind (not a file or directory)")
            }
            CorpusError::TooManyFiles { total, cap } => {
                write!(f, "corpus has {total} files, over the {cap}-file cap")
            }
        }
    }
}

impl std::error::Error for CorpusError {}

impl From<CorpusError> for String {
    fn from(e: CorpusError) -> Self {
        e.to_string()
    }
}

fn valid_task_id(name: &str) -> bool {
    if name.is_empty()
        || name.len() > MAX_TASK_ID_BYTES
        || name == "."
        || name == ".."
        || name.starts_with('.')
        || name.starts_with('-')
        || name.ends_with('-')
    {
        return false;
    }
    name.bytes()
        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// A recursive, symlink-free file listing with deterministic order and
/// bounded size — the shared walker of the loader and the immutability
/// snapshotter. Rejects symlinks, non-file/non-dir entries and `..`
/// components (a hostile tree can never escape its root).
pub fn walk_files(
    root: &Path,
    cap_files: usize,
    cap_bytes: u64,
) -> Result<Vec<(PathBuf, Vec<u8>)>, CorpusError> {
    let mut out = Vec::new();
    let mut total_bytes = 0u64;
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        let entries = std::fs::read_dir(&dir).map_err(|e| CorpusError::Io {
            path: dir.display().to_string(),
            detail: e.to_string(),
        })?;
        let mut names: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect::<Vec<_>>();
        names.sort();
        for path in names {
            let rel = path
                .strip_prefix(root)
                .map_err(|_| CorpusError::UnexpectedEntry {
                    path: path.display().to_string(),
                })?;
            if rel
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
            {
                return Err(CorpusError::UnexpectedEntry {
                    path: path.display().to_string(),
                });
            }
            let meta = std::fs::symlink_metadata(&path).map_err(|e| CorpusError::Io {
                path: path.display().to_string(),
                detail: e.to_string(),
            })?;
            let kind = meta.file_type();
            if kind.is_symlink() {
                return Err(CorpusError::Symlink {
                    path: path.display().to_string(),
                });
            }
            if kind.is_dir() {
                dirs.push(path);
            } else if kind.is_file() {
                if out.len() >= cap_files {
                    return Err(CorpusError::TooManyFiles {
                        total: out.len() + 1,
                        cap: cap_files,
                    });
                }
                let bytes = std::fs::read(&path).map_err(|e| CorpusError::Io {
                    path: path.display().to_string(),
                    detail: e.to_string(),
                })?;
                total_bytes = total_bytes.saturating_add(bytes.len() as u64);
                if total_bytes > cap_bytes {
                    return Err(CorpusError::Oversized {
                        path: path.display().to_string(),
                        bytes: total_bytes,
                        cap: cap_bytes,
                    });
                }
                out.push((path, bytes));
            } else {
                return Err(CorpusError::UnexpectedEntry {
                    path: path.display().to_string(),
                });
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Load and adversarially validate the whole corpus. Never writes anything.
pub fn load_corpus(root: &Path) -> Result<Corpus, CorpusError> {
    let mut task_ids = Vec::new();
    let entries = std::fs::read_dir(root).map_err(|e| CorpusError::Io {
        path: root.display().to_string(),
        detail: e.to_string(),
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        let meta = std::fs::symlink_metadata(&path).map_err(|e| CorpusError::Io {
            path: path.display().to_string(),
            detail: e.to_string(),
        })?;
        if meta.file_type().is_symlink() {
            return Err(CorpusError::Symlink {
                path: path.display().to_string(),
            });
        }
        if !meta.is_dir() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        if name == "." || name == ".." || name.starts_with('.') {
            continue;
        }
        task_ids.push(name);
    }
    task_ids.sort();

    let mut tasks = Vec::new();
    let mut corpus_files = 0usize;
    let mut seen_ids = HashSet::new();
    for id in task_ids {
        if !valid_task_id(&id) {
            return Err(CorpusError::InvalidTaskId(id));
        }
        if !seen_ids.insert(id.clone()) {
            return Err(CorpusError::InvalidTaskId(format!(
                "duplicate task id {id}"
            )));
        }
        let dir = root.join(&id);
        let task = load_task(&dir)?;
        corpus_files = corpus_files.saturating_add(task.file_count);
        if corpus_files > MAX_CORPUS_FILES {
            return Err(CorpusError::TooManyFiles {
                total: corpus_files,
                cap: MAX_CORPUS_FILES,
            });
        }
        tasks.push(task);
    }
    Ok(Corpus {
        root: root.to_path_buf(),
        tasks,
    })
}

fn read_text_file(path: &Path, cap: u64) -> Result<String, CorpusError> {
    let bytes = std::fs::read(path).map_err(|e| CorpusError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    if bytes.len() as u64 > cap {
        return Err(CorpusError::Oversized {
            path: path.display().to_string(),
            bytes: bytes.len() as u64,
            cap,
        });
    }
    let text = String::from_utf8(bytes).map_err(|e| CorpusError::UnreadableText {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    if text.trim().is_empty() {
        return Err(CorpusError::EmptyFile {
            path: path.display().to_string(),
        });
    }
    Ok(text)
}

/// Load and validate ONE task directory (used by tests to build synthetic
/// tasks; the corpus loader is the only caller in production paths).
pub fn load_task(dir: &Path) -> Result<Task, CorpusError> {
    let id = dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string();
    let lang = Lang::from_dir_name(&id).ok_or_else(|| CorpusError::InvalidTaskId(id.clone()))?;
    for file in ["task.md", "criteria.md", "verify.sh"] {
        let p = dir.join(file);
        if !p.is_file() {
            return Err(CorpusError::MissingFile {
                task: id.clone(),
                file: file.to_string(),
            });
        }
    }
    let files = walk_files(dir, MAX_TASK_FILES, MAX_META_BYTES * 4)?;
    let task_md = read_text_file(&dir.join("task.md"), MAX_META_BYTES)?;
    let criteria_md = read_text_file(&dir.join("criteria.md"), MAX_META_BYTES)?;
    let verify_sh = read_text_file(&dir.join("verify.sh"), MAX_META_BYTES)?;
    if verify_sh.contains('\0') {
        return Err(CorpusError::UnreadableText {
            path: dir.join("verify.sh").display().to_string(),
            detail: "contains a NUL byte".into(),
        });
    }
    let criteria =
        parse_criteria(&criteria_md).map_err(|e| CorpusError::Criteria(format!("{id}: {e}")))?;
    Ok(Task {
        id,
        lang,
        dir: dir.to_path_buf(),
        task_md,
        criteria,
        verify_sh,
        file_count: files.len(),
    })
}

/// The validated checked-in corpus of this crate.
#[derive(Debug, Clone)]
pub struct Corpus {
    pub root: PathBuf,
    pub tasks: Vec<Task>,
}

impl Corpus {
    pub fn task(&self, id: &str) -> Option<&Task> {
        self.tasks.iter().find(|t| t.id == id)
    }

    pub fn task_ids(&self) -> Vec<&str> {
        self.tasks.iter().map(|t| t.id.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_id_rule_rejects_hostile_names() {
        for bad in [
            "..", "../evil", "a b", "A", "rust_", ".rust-x", "-rust-x", "rust-x-", "",
        ] {
            assert!(!valid_task_id(bad), "{bad:?} must be rejected");
        }
        for good in ["rust-x", "c-ringbuf", "a0", "python-dedup-preserve-order"] {
            assert!(valid_task_id(good), "{good:?} must be accepted");
        }
    }

    #[test]
    fn lang_prefixes_are_disjoint_and_complete() {
        for (name, expect) in [
            ("rust-reverse-words", Some(Lang::Rust)),
            ("c-ringbuf", Some(Lang::C)),
            ("python-dedup", Some(Lang::Python)),
            ("go-sumranges", Some(Lang::Go)),
            ("ts-clamp-sum", Some(Lang::TypeScript)),
            ("java-leap-years", Some(Lang::Java)),
            ("car-bench", None),
        ] {
            assert_eq!(Lang::from_dir_name(name), expect, "{name}");
        }
    }
}
