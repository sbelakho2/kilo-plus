//! Production evidence provider (spec §20): the automatic evidence package
//! behind the daemon's context engine.
//!
//! The runtime calls `evidence_for(session, prompt)` before every reasoning
//! turn. This implementation resolves the session's workspace from the
//! durable store, indexes the workspace ONCE per workspace id (bounded scan;
//! the index is a memory-sidecar, never a session dependency), and asks the
//! search service for an evidence package from concepts of the prompt.
//!
//! Everything here is advisory: any failure (missing session, unreadable
//! root, hostile input, scan caps) yields an empty evidence list and never
//! breaks the turn.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use kilop_agent::{EvidenceProvider, EvidenceQuery};
use kilop_context::assembler::Evidence;
use kilop_core::id::{SessionId, WorkspaceId};
use kilop_index::{tokenize, WorkspaceIndex};
use kilop_search::SearchService;
use kilop_session::SessionManager;

/// Scan bounds (architecture §13 budgets): a workspace larger than this is
/// PARTIALLY indexed (deterministic walk order) — evidence stays bounded
/// even for hostile repos.
const SCAN_MAX_FILES: usize = 4_000;
const SCAN_MAX_DIRS: usize = 16_000;
const SCAN_MAX_BYTES: usize = 64 * 1024 * 1024;
const SCAN_MAX_FILE_BYTES: usize = 1_000_000;
const EVIDENCE_MAX_HITS: usize = 8;
const CONCEPT_MAX: usize = 16;
const CONCEPT_MIN_CHARS: usize = 4;

/// Directories that are never indexed (vcs metadata, dependency trees).
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    ".venv",
    "dist",
];

/// One index scan per workspace id per process; further prompts reuse it.
struct ScanState {
    scanned: HashSet<WorkspaceId>,
    failed: HashSet<WorkspaceId>,
}

pub struct RepoEvidence {
    session: Arc<SessionManager>,
    index: Arc<Mutex<WorkspaceIndex>>,
    search: SearchService,
    scan: Mutex<ScanState>,
    scan_max_files: usize,
    scan_max_dirs: usize,
    scan_max_bytes: usize,
    scan_max_file_bytes: usize,
}

impl RepoEvidence {
    pub fn new(session: Arc<SessionManager>) -> Self {
        let index = Arc::new(Mutex::new(WorkspaceIndex::new()));
        let search = SearchService::new(index.clone(), None);
        Self {
            session,
            index,
            search,
            scan: Mutex::new(ScanState {
                scanned: HashSet::new(),
                failed: HashSet::new(),
            }),
            scan_max_files: SCAN_MAX_FILES,
            scan_max_dirs: SCAN_MAX_DIRS,
            scan_max_bytes: SCAN_MAX_BYTES,
            scan_max_file_bytes: SCAN_MAX_FILE_BYTES,
        }
    }

    #[cfg(test)]
    fn with_caps(
        session: Arc<SessionManager>,
        scan_max_files: usize,
        scan_max_dirs: usize,
        scan_max_bytes: usize,
        scan_max_file_bytes: usize,
    ) -> Self {
        let index = Arc::new(Mutex::new(WorkspaceIndex::new()));
        let search = SearchService::new(index.clone(), None);
        Self {
            session,
            index,
            search,
            scan: Mutex::new(ScanState {
                scanned: HashSet::new(),
                failed: HashSet::new(),
            }),
            scan_max_files,
            scan_max_dirs,
            scan_max_bytes,
            scan_max_file_bytes,
        }
    }

    /// Resolve the session's canonical workspace root (same source of truth
    /// as the tool runtime).
    fn resolve_root(&self, session: SessionId) -> Option<(WorkspaceId, PathBuf)> {
        let handle = self.session.get_session(session).ok()??;
        let row = handle.row().ok()?;
        let root = self
            .session
            .store()
            .workspace_root(row.workspace_id)
            .ok()??
            .into();
        Some((row.workspace_id, root))
    }

    /// Deterministic bounded walk; skips vcs/dependency directories, binary
    /// and oversized files. A cap hit stops the walk early (partial index).
    fn scan_workspace(&self, ws: WorkspaceId, root: &Path) {
        let mut files_scanned = 0usize;
        let mut dirs_visited = 0usize;
        let mut bytes_indexed = 0usize;
        let mut stack = vec![PathBuf::from(root)];
        let mut index = self.index.lock().unwrap();
        while let Some(dir) = stack.pop() {
            dirs_visited += 1;
            if dirs_visited > self.scan_max_dirs {
                break;
            }
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                match entry.file_type() {
                    Ok(t) if t.is_dir() => {
                        if !SKIP_DIRS.contains(&name.as_str()) {
                            stack.push(path);
                        }
                    }
                    Ok(t) if t.is_file() => {
                        if files_scanned >= self.scan_max_files {
                            return;
                        }
                        let meta = match std::fs::metadata(&path) {
                            Ok(m) => m,
                            Err(_) => continue,
                        };
                        if meta.len() > self.scan_max_file_bytes as u64 {
                            continue;
                        }
                        if bytes_indexed >= self.scan_max_bytes {
                            return;
                        }
                        let bytes = match std::fs::read(&path) {
                            Ok(b) => b,
                            Err(_) => continue,
                        };
                        // Binary sniff: NUL in the first 8 KiB → not text.
                        let head = bytes.iter().take(8192).any(|b| *b == 0);
                        if head {
                            continue;
                        }
                        let rel = match path.strip_prefix(root) {
                            Ok(r) => r.to_path_buf(),
                            Err(_) => continue,
                        };
                        let modified_ms = meta
                            .modified()
                            .ok()
                            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                            .map(|d| d.as_millis() as i64)
                            .unwrap_or(0);
                        let _ = index.index_file(ws, &rel, &bytes, modified_ms);
                        files_scanned += 1;
                        bytes_indexed = bytes_indexed.saturating_add(bytes.len());
                    }
                    // Symlinks and special files are never indexed.
                    _ => {}
                }
            }
        }
    }

    /// Concepts from the retrieval signal (spec §20): the prompt's own
    /// words first, then basename tokens of the changed files (so edited
    /// files rank for follow-up), then failure keywords. Bounded, deduped.
    fn concepts(query: &EvidenceQuery) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        let push_tokens = |text: &str, out: &mut Vec<String>, seen: &mut HashSet<String>| {
            for tok in tokenize(text).into_iter().take(512) {
                if tok.len() < CONCEPT_MIN_CHARS || !seen.insert(tok.clone()) {
                    continue;
                }
                out.push(tok);
                if out.len() >= CONCEPT_MAX {
                    return;
                }
            }
        };
        push_tokens(&query.prompt, &mut out, &mut seen);
        if out.len() < CONCEPT_MAX {
            for f in query.changed_files.iter().take(16) {
                let base = f.rsplit('/').next().unwrap_or(f.as_str());
                push_tokens(base, &mut out, &mut seen);
                if out.len() >= CONCEPT_MAX {
                    break;
                }
            }
        }
        if out.len() < CONCEPT_MAX {
            push_tokens(&query.failures.join(" "), &mut out, &mut seen);
        }
        out
    }
}

impl RepoEvidence {
    /// The session ended: drop this workspace's scan state AND index
    /// postings so memory stays bounded across session churn.
    pub fn forget_workspace(&self, ws: WorkspaceId) {
        self.index.lock().unwrap().remove_workspace(ws);
        let mut scan = self.scan.lock().unwrap();
        scan.scanned.remove(&ws);
        scan.failed.remove(&ws);
    }
}

impl EvidenceProvider for RepoEvidence {
    fn evidence_for(&self, session: SessionId, query: &EvidenceQuery) -> Vec<Evidence> {
        let Some((ws, root)) = self.resolve_root(session) else {
            return vec![];
        };
        {
            let scan = self.scan.lock().unwrap();
            if scan.failed.contains(&ws) {
                return vec![];
            }
            if scan.scanned.contains(&ws) {
                return self.evidence_package(ws, query);
            }
        }
        if root.is_dir() {
            self.scan_workspace(ws, &root);
        }
        let mut scan = self.scan.lock().unwrap();
        if !scan.scanned.insert(ws) {
            scan.failed.insert(ws);
        }
        drop(scan);
        self.evidence_package(ws, query)
    }

    fn forget(&self, workspace: WorkspaceId) {
        self.forget_workspace(workspace);
    }
}

impl RepoEvidence {
    fn evidence_package(&self, ws: WorkspaceId, query: &EvidenceQuery) -> Vec<Evidence> {
        if query.prompt.len() > 512 * 1024 {
            return vec![];
        }
        let concepts = Self::concepts(query);
        if concepts.is_empty() {
            return vec![];
        }
        let hits = self
            .search
            .evidence_package(ws, &concepts, EVIDENCE_MAX_HITS);
        hits.into_iter()
            .enumerate()
            .map(|(i, h)| Evidence {
                path: h.path,
                snippet: h.snippet,
                score: 1.0 / (1.0 + i as f64),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn manager() -> Arc<SessionManager> {
        // The dir must outlive the manager (per-call connections reopen the
        // db file); leak it for the process lifetime.
        static KEEP: std::sync::Mutex<Vec<TempDir>> = std::sync::Mutex::new(Vec::new());
        let dir = TempDir::new().unwrap();
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        KEEP.lock().unwrap().push(dir);
        m
    }

    /// Register the workspace + a session on it; returns the session id.
    fn registered_session(m: &Arc<SessionManager>, root: &Path) -> SessionId {
        let ws = m.create_workspace(root.to_str().unwrap()).unwrap();
        m.create_session(ws, "t", "p", "m").unwrap().id()
    }

    fn write(root: &Path, rel: &str, bytes: &[u8]) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    #[test]
    fn evidence_finds_symbols_in_workspace() {
        let root = TempDir::new().unwrap();
        write(
            root.path(),
            "src/lib.rs",
            b"pub fn balance_account() -> i64 { 42 }\n",
        );
        let m = manager();
        let ev = RepoEvidence::new(m.clone());
        let sid = registered_session(&m, root.path());
        let evidence = ev.evidence_for(
            sid,
            &EvidenceQuery {
                prompt: "fix balance_account".into(),
                ..Default::default()
            },
        );
        assert!(
            evidence.iter().any(|e| e.path.ends_with("lib.rs")),
            "evidence must surface the file defining the concept: {evidence:?}"
        );
    }

    #[test]
    fn scan_is_capped_and_never_hangs() {
        let root = TempDir::new().unwrap();
        for i in 0..200 {
            write(
                root.path(),
                &format!("f{i}.rs"),
                format!("pub fn fx{i}() {{}}\n").as_bytes(),
            );
        }
        let m = manager();
        let ev = RepoEvidence::with_caps(m.clone(), 25, 10_000, 64 * 1024 * 1024, 1_000_000);
        // The scan cap must not panic, must terminate, and must bound work.
        let sid = registered_session(&m, root.path());
        let evidence = ev.evidence_for(
            sid,
            &EvidenceQuery {
                prompt: "fx199".into(),
                ..Default::default()
            },
        );
        assert!(evidence.len() <= 8);
        // fx199 may or may not be indexed (25 < 200) — but the call returns.
        assert!(evidence.len() <= EVIDENCE_MAX_HITS);
    }

    #[test]
    fn skips_vcs_and_junk() {
        let root = TempDir::new().unwrap();
        write(root.path(), "src/app.rs", b"fn payments() {}\n");
        write(root.path(), ".git/config", b"fn payments() {}\n");
        write(root.path(), "target/debug/x.rs", b"fn payments() {}\n");
        write(
            root.path(),
            "node_modules/lib.js",
            b"function payments() {}\n",
        );
        let m = manager();
        let ev = RepoEvidence::new(m.clone());
        let sid = registered_session(&m, root.path());
        let evidence = ev.evidence_for(
            sid,
            &EvidenceQuery {
                prompt: "payments".into(),
                ..Default::default()
            },
        );
        assert_eq!(evidence.len(), 1);
        assert!(evidence[0].path.ends_with("src/app.rs"));
    }

    #[test]
    fn hostile_workspace_never_panics() {
        let root = TempDir::new().unwrap();
        let mut bin = vec![0u8; 64];
        bin.extend([1u8; 64]);
        write(root.path(), "bin.dat", &bin);
        let m = manager();
        let ev = RepoEvidence::new(m.clone());
        // Root that is a file, missing root, unknown session.
        let sid = registered_session(&m, root.path());
        let _ = ev.evidence_for(
            sid,
            &EvidenceQuery {
                prompt: "payments".into(),
                ..Default::default()
            },
        );
        let unknown = SessionId::new(99_999);
        assert!(ev
            .evidence_for(
                unknown,
                &EvidenceQuery {
                    prompt: "payments".into(),
                    ..Default::default()
                },
            )
            .is_empty());
        // A session whose store root vanished.
        let missing = TempDir::new().unwrap();
        write(missing.path(), "x.rs", b"fn alpha() {}\n");
        let sid2 = registered_session(&m, missing.path());
        std::fs::remove_dir_all(missing.path()).unwrap();
        let out = ev.evidence_for(
            sid2,
            &EvidenceQuery {
                prompt: "alpha".into(),
                ..Default::default()
            },
        );
        assert!(out.len() <= 8);
    }

    #[test]
    fn hostile_prompt_is_bounded() {
        let root = TempDir::new().unwrap();
        write(root.path(), "src/app.rs", b"fn payments() {}\n");
        let m = manager();
        let ev = RepoEvidence::new(m.clone());
        let sid = registered_session(&m, root.path());
        // 1 MiB prompt: bounded token extraction, bounded output.
        let huge = "a".repeat(1024 * 1024);
        let evidence = ev.evidence_for(
            sid,
            &EvidenceQuery {
                prompt: huge,
                ..Default::default()
            },
        );
        assert!(evidence.len() <= EVIDENCE_MAX_HITS);
        // Empty prompt → nothing.
        assert!(ev.evidence_for(sid, &EvidenceQuery::default()).is_empty());
    }

    #[test]
    fn binary_and_oversized_files_skipped() {
        let root = TempDir::new().unwrap();
        write(root.path(), "big.rs", &[b'x'; 2_000_000]);
        write(root.path(), "good.rs", b"fn target_fn() {}\n");
        let m = manager();
        let ev = RepoEvidence::with_caps(m.clone(), 100, 10_000, 64 * 1024 * 1024, 1_000);
        let sid = registered_session(&m, root.path());
        let evidence = ev.evidence_for(
            sid,
            &EvidenceQuery {
                prompt: "target_fn".into(),
                ..Default::default()
            },
        );
        assert_eq!(evidence.len(), 1);
        assert!(evidence[0].path.ends_with("good.rs"));
    }

    #[test]
    fn changed_file_signal_drives_retrieval_without_prompt_keyword() {
        // Spec §20: retrieval must not depend on the model's wording — a
        // prompt with NO keyword still surfaces evidence for the file the
        // task changed (basename tokens are concepts too).
        let root = TempDir::new().unwrap();
        write(
            root.path(),
            "src/payments_ledger.rs",
            b"pub fn settle_payments() -> i64 { 7 }\n",
        );
        let m = manager();
        let ev = RepoEvidence::new(m.clone());
        let sid = registered_session(&m, root.path());
        let evidence = ev.evidence_for(
            sid,
            &EvidenceQuery {
                prompt: "continue the task please".into(),
                changed_files: vec!["src/payments_ledger.rs".into()],
                ..Default::default()
            },
        );
        assert!(
            evidence
                .iter()
                .any(|e| e.path.ends_with("payments_ledger.rs")),
            "changed-file signal must drive retrieval: {evidence:?}"
        );
    }

    #[tokio::test]
    async fn forget_drops_index_and_rescans_later() {
        // Spec §21: a closed session's index is dropped; a later session on
        // the same workspace rescans (a file created after the first scan is
        // only discoverable after the forget).
        let root = TempDir::new().unwrap();
        write(root.path(), "src/one.rs", b"pub fn alpha_fn() {}\n");
        let m = manager();
        let ev = RepoEvidence::new(m.clone());
        let sid = registered_session(&m, root.path());
        // First scan: alpha_fn found.
        let evidence = ev.evidence_for(
            sid,
            &EvidenceQuery {
                prompt: "alpha_fn".into(),
                ..Default::default()
            },
        );
        assert_eq!(evidence.len(), 1);
        let ws = ev.resolve_root(sid).unwrap().0;
        // The workspace is dropped: scan state AND postings.
        ev.forget_workspace(ws);
        // New file after the forget.
        write(root.path(), "src/two.rs", b"pub fn beta_fn() {}\n");
        let evidence = ev.evidence_for(
            sid,
            &EvidenceQuery {
                prompt: "beta_fn".into(),
                ..Default::default()
            },
        );
        assert!(
            evidence.iter().any(|e| e.path.ends_with("two.rs")),
            "the rescanned index must see the new file: {evidence:?}"
        );
    }
}
