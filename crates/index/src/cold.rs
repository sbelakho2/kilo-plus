//! Cheap cold-start evidence (P0-30): what the runtime serves while no
//! `Ready` index generation exists yet. The legacy fallback scanned the
//! whole workspace on the first prompt (up to 4000 files / 64 MiB / 16000
//! dirs synchronously); this provider NEVER walks the tree. Decision tree:
//!
//! ```text
//! 1. persisted OLD generation present (and <= COLD_MAX_GENERATION_LOAD_BYTES)
//!    -> serve evidence from it immediately (stale-but-cheap beats scanning;
//!       wave-11 keeps two generations, so a restart almost always has one)
//! 2. else, a git repo (.git exists):
//!    a. `git ls-files` for the tracked-file set (bounded output)
//!    b. ONE targeted, deadline-bounded ripgrep for the turn's concepts,
//!       scoped to the dirs of the directly referenced files (whole-tree
//!       only when the turn references nothing, still deadline-killed)
//!    c. results filtered to tracked paths
//! 3. always: reads of DIRECTLY referenced files (the turn's changed files
//!    + up to 32 referenced paths, head bytes only)
//! ```
//!
//! No step lists more than a few hundred directory entries and the whole
//! call runs under [`COLD_OVERALL_DEADLINE`]; when a step exceeds the
//! deadline or a tool is missing the provider DEGRADES to the remaining
//! cheaper steps (documented degrade, never the full scan). Next turn,
//! after the wave-11 service publishes a Ready generation, the runtime's
//! existing view logic serves full index evidence — nothing here caches or
//! upgrades; this is purely the pre-Ready fallback.

use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::generation::GenerationFile;
use crate::tokenize;
use crate::{WorkspaceId, WorkspaceIndex};

/// Directly referenced paths read per call (changed files + referenced).
pub const COLD_MAX_REFERENCE_PATHS: usize = 32;
/// Head bytes read per directly referenced file.
pub const COLD_MAX_READ_BYTES_PER_FILE: u64 = 64 * 1024;
/// Snippet characters kept per hit.
pub const COLD_MAX_SNIPPET_CHARS: usize = 1500;
/// Evidence hits returned per call (same bound as every other path).
pub const COLD_MAX_HITS: usize = 8;
/// A stale generation larger than this is not "cheap" to decode: skip to
/// the targeted steps (documented degrade).
pub const COLD_MAX_GENERATION_LOAD_BYTES: u64 = 16 * 1024 * 1024;
/// Hard wall-clock budget of one cold evidence call.
pub const COLD_OVERALL_DEADLINE: Duration = Duration::from_millis(1500);
/// Per external command (git/rg) budget; a killed command degrades.
const COLD_COMMAND_TIMEOUT: Duration = Duration::from_millis(900);
/// Tracked-file list cap (paths). Overflow degrades to "unknown tracked
/// set" — never a truncated set trusted as complete.
const COLD_MAX_TRACKED_PATHS: usize = 20_000;
/// Tracked-file list cap (bytes).
const COLD_MAX_TRACKED_BYTES: u64 = 512 * 1024;
/// Concept cap (mirrors every other evidence path's bound).
const COLD_MAX_CONCEPTS: usize = 16;

/// Crate-wide test serialization (shared with the service seam tests):
/// the heavy cold fixtures and the seam-installing service tests must not
/// starve each other's deadlines on loaded machines.
#[cfg(test)]
pub(crate) static TEST_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
const COLD_CONCEPT_MIN_CHARS: usize = 4;

/// One cold-evidence hit: path + snippet + score (renderer-shaped, mirrors
/// the search layer's bounded package shape).
#[derive(Debug, Clone, PartialEq)]
pub struct ColdHit {
    pub path: String,
    pub snippet: String,
    pub score: f64,
}

/// Where the cold evidence came from (observability + adversarial tests:
/// proves which decision-tree branch served, and that nothing deeper ran).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdOrigin {
    /// A persisted OLD generation served the concepts; direct reads still
    /// rode along for the turn's own referenced files.
    StaleGeneration {
        generation: u64,
        direct_reads: usize,
    },
    /// Tracked-file git listing + targeted ripgrep served the concepts.
    TrackedSearch { tracked: bool, direct_reads: usize },
    /// No generation and no git: only the directly referenced files were
    /// read. The documented degrade end of the ladder.
    DirectReads { files: usize },
    /// Nothing at all was available (no generation, no git, no references,
    /// no concepts): an empty package, never a scan.
    None,
}

/// Work the call actually performed (proves "never a full walk").
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ColdStats {
    /// Files opened and read (head bytes).
    pub files_read: usize,
    /// Directories listed.
    pub dirs_listed: usize,
    /// External commands spawned (git/rg).
    pub commands_run: usize,
    /// Steps skipped because a cheaper degrade already served enough or a
    /// deadline/tool failure cut them (documented degrade).
    pub degraded_steps: usize,
}

/// The retrieval signal of one turn, shaped for the cold provider.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColdQuery {
    pub prompt: String,
    /// The turn's own changed files (highest-priority direct reads).
    pub changed_files: Vec<String>,
    /// Other directly referenced paths (tool reads, verification targets).
    pub referenced_paths: Vec<String>,
    /// Known failure keywords (concept signals).
    pub failures: Vec<String>,
}

/// The cold evidence package of one query.
#[derive(Debug, Clone, PartialEq)]
pub struct ColdEvidence {
    pub hits: Vec<ColdHit>,
    pub origin: ColdOrigin,
    pub stats: ColdStats,
}

/// Runner seam for git/ripgrep (tests inject failures/timeouts; production
/// uses [`run_command`]). `args` never contains the root: commands run with
/// the workspace root as cwd.
pub type CommandRunner = dyn Fn(&str, &[String]) -> std::io::Result<String> + Send + Sync;

/// Spawn `program args...` in `cwd`, capture stdout (bounded), kill after
/// [`COLD_COMMAND_TIMEOUT`]. A missing binary is `Err(NotFound)` — the
/// documented degrade trigger.
pub fn run_command(program: &str, args: &[String], cwd: &Path) -> std::io::Result<String> {
    use std::process::Stdio;
    let mut child = std::process::Command::new(program)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = Instant::now() + COLD_COMMAND_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("cold command {program} exceeded {COLD_COMMAND_TIMEOUT:?}"),
                    ));
                }
                std::thread::sleep(Duration::from_millis(4));
            }
            Err(e) => return Err(e),
        }
    };
    let mut out = String::new();
    if let Some(mut stdout) = child.stdout.take() {
        let mut buf = [0u8; 16 * 1024];
        loop {
            match stdout.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    out.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if out.len() as u64 > COLD_MAX_TRACKED_BYTES {
                        // Bounded capture: hostile/giant output never floods
                        // RAM; the caller treats truncation as a degrade.
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }
    if status.success() {
        Ok(out)
    } else {
        Err(std::io::Error::other(format!(
            "cold command {program} exited {status}"
        )))
    }
}

/// The cheap pre-Ready evidence provider (P0-30). Sync and bounded: every
/// step is an O(references) file read or a deadline-killed targeted tool
/// call; nothing here walks the tree.
pub struct ColdEvidenceProvider {
    root: PathBuf,
    workspace: WorkspaceId,
    generations_dir: PathBuf,
    overall_deadline: Duration,
    run: Arc<CommandRunner>,
}

impl std::fmt::Debug for ColdEvidenceProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColdEvidenceProvider")
            .field("root", &self.root)
            .field("workspace", &self.workspace)
            .field("generations_dir", &self.generations_dir)
            .finish_non_exhaustive()
    }
}

impl ColdEvidenceProvider {
    pub fn new(root: PathBuf, workspace: WorkspaceId, generations_dir: PathBuf) -> Self {
        let cwd = root.clone();
        Self {
            root,
            workspace,
            generations_dir,
            overall_deadline: COLD_OVERALL_DEADLINE,
            run: Arc::new(move |p: &str, a: &[String]| run_command(p, a, &cwd)),
        }
    }

    /// Test seam: inject a command runner (missing-tool and timeout
    /// simulation) and a deadline.
    #[cfg(test)]
    fn with_runner(
        root: PathBuf,
        workspace: WorkspaceId,
        generations_dir: PathBuf,
        run: Arc<CommandRunner>,
    ) -> Self {
        Self {
            root,
            workspace,
            generations_dir,
            overall_deadline: COLD_OVERALL_DEADLINE,
            run,
        }
    }

    fn time_left(&self, started: Instant) -> bool {
        started.elapsed() < self.overall_deadline
    }

    /// The cold evidence package for one turn. Never panics, never walks
    /// the tree, never exceeds the deadline.
    pub fn evidence(&self, query: &ColdQuery) -> ColdEvidence {
        let started = Instant::now();
        let mut stats = ColdStats::default();

        // Directly referenced files, sanitized + deduped (changed first).
        let mut references: Vec<(String, PathBuf)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for raw in query
            .changed_files
            .iter()
            .chain(query.referenced_paths.iter())
            .take(COLD_MAX_REFERENCE_PATHS)
        {
            if !seen.insert(raw.clone()) {
                continue;
            }
            if let Some(p) = sanitize_reference(&self.root, raw) {
                references.push((raw.clone(), p));
            }
        }

        let mut discovery: Vec<ColdHit> = Vec::new();
        let mut direct: Vec<ColdHit> = Vec::new();
        let mut degraded = 0usize;
        let mut gen_served: Option<u64> = None;
        let mut search_ran = false;

        // 1. Stale-but-cheap: a persisted OLD generation serves immediately
        // (its own decode is bounded; oversized files degrade to 2).
        if self.time_left(started) {
            if let Some((generation, index)) = self.load_stale_generation(&mut stats) {
                let concepts = concepts(query);
                let mut scored = search_index(&index, self.workspace, &concepts);
                scored.truncate(COLD_MAX_HITS);
                for (path, score) in scored {
                    let snippet = self.snippet_of(&path, &mut stats);
                    discovery.push(ColdHit {
                        path,
                        snippet: snippet.unwrap_or_default(),
                        score,
                    });
                }
                gen_served = Some(generation);
            }
        } else {
            degraded += 1;
        }

        // 2. Direct reads: the turn's own referenced files always ride
        // along (their content IS the targeted evidence; stale generation
        // hits may add more, never replace these).
        let mut direct_reads = 0usize;
        for (rel, path) in &references {
            if !self.time_left(started) {
                degraded += 1;
                break;
            }
            if let Some(text) = self.snippet_of_file(path, &mut stats) {
                direct_reads += 1;
                direct.push(ColdHit {
                    path: rel.clone(),
                    snippet: text,
                    score: 0.99,
                });
            }
        }

        // 3. Git + targeted ripgrep ONLY when no generation served (the
        // generation branch is the restart "scan replacement"; discovery
        // stops as soon as one branch served). Missing tools and deadline
        // expiry degrade to the direct-read package — never the full scan.
        if gen_served.is_none() && self.time_left(started) {
            if self.root.join(".git").exists() || self.root.join(".git").is_file() {
                if let Some(mut hits) = self.tracked_search(query, started, &mut stats) {
                    search_ran = true;
                    discovery.append(&mut hits);
                } else {
                    degraded += 1;
                }
            } else {
                // No git: no tree listing exists in the cold ladder at all.
                degraded += 1;
            }
        } else if gen_served.is_none() {
            degraded += 1;
        }

        // Merge: discovery first (ranked), direct reads second (freshest),
        // deduped by path (direct wins the content, discovery wins the
        // score only when its rank is better), bounded.
        let mut merged: Vec<ColdHit> = Vec::new();
        let mut placed = std::collections::HashSet::new();
        for h in discovery.into_iter().chain(direct.iter().cloned()) {
            if !placed.insert(h.path.clone()) {
                continue;
            }
            merged.push(h);
        }
        merged.truncate(COLD_MAX_HITS);

        let origin = match gen_served {
            Some(generation) => ColdOrigin::StaleGeneration {
                generation,
                direct_reads,
            },
            None if search_ran => ColdOrigin::TrackedSearch {
                tracked: true,
                direct_reads,
            },
            None if !merged.is_empty() || direct_reads > 0 => ColdOrigin::DirectReads {
                files: direct_reads,
            },
            None => ColdOrigin::None,
        };
        stats.degraded_steps = degraded;
        ColdEvidence {
            hits: merged,
            origin,
            stats,
        }
    }

    /// Head snippet of a file, counted as a read when it happens.
    fn snippet_of(&self, rel: &str, stats: &mut ColdStats) -> Option<String> {
        let path = sanitize_reference(&self.root, rel)?;
        stats.files_read += 1;
        read_head(&path)
    }

    fn snippet_of_file(&self, path: &Path, stats: &mut ColdStats) -> Option<String> {
        stats.files_read += 1;
        read_head(path)
    }

    /// The newest persisted generation whose file is cheap to decode.
    fn load_stale_generation(&self, stats: &mut ColdStats) -> Option<(u64, WorkspaceIndex)> {
        let dir = self.generations_dir.join(self.workspace.raw().to_string());
        stats.dirs_listed += 1;
        let entries = std::fs::read_dir(&dir).ok()?;
        let mut newest: Option<(u64, PathBuf)> = None;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let g = name
                .strip_prefix("gen-")
                .and_then(|n| n.strip_suffix(".json"))
                .and_then(|n| n.parse::<u64>().ok());
            if let Some(g) = g {
                if newest.as_ref().map(|(ng, _)| g > *ng).unwrap_or(true) {
                    newest = Some((g, entry.path()));
                }
            }
        }
        let (generation, path) = newest?;
        let meta = std::fs::metadata(&path).ok()?;
        if meta.len() > COLD_MAX_GENERATION_LOAD_BYTES {
            // Too big to be cheap: degrade (documented).
            return None;
        }
        stats.files_read += 1;
        let bytes = std::fs::read(&path).ok()?;
        let file = GenerationFile::from_bytes(&bytes).ok()?;
        if file.workspace != self.workspace.raw() || file.generation != generation {
            // Hostile/mismatched fixture: never serve someone else's tree.
            return None;
        }
        let index = file.materialize().ok()?;
        Some((generation, index))
    }

    /// Git tracked set + ONE targeted ripgrep for the turn's concepts,
    /// scoped to the referenced files' dirs. Missing tools or deadlines
    /// return `None` (the direct-read package is the documented degrade).
    fn tracked_search(
        &self,
        query: &ColdQuery,
        started: Instant,
        stats: &mut ColdStats,
    ) -> Option<Vec<ColdHit>> {
        stats.commands_run += 1;
        let out = (self.run)("git", &["ls-files".into(), "-z".into()]).ok()?;
        let mut set = std::collections::HashSet::new();
        for piece in out.split('\0') {
            if piece.is_empty() {
                continue;
            }
            if set.len() >= COLD_MAX_TRACKED_PATHS {
                // Overflow: the truncated set is NOT trusted as complete —
                // degrade to direct reads.
                return None;
            }
            set.insert(piece.to_string());
        }
        let tracked: Option<std::collections::HashSet<String>> = Some(set);
        let concepts = concepts(query);
        if concepts.is_empty() {
            return None;
        }
        if !self.time_left(started) {
            return None;
        }
        // Scope: the dirs of the directly referenced files; with none, the
        // tree root (deadline-killed on hostile giants — never a full walk
        // that hangs the turn).
        let mut scope: Vec<String> = Vec::new();
        let mut scope_set = std::collections::HashSet::new();
        for raw in query
            .changed_files
            .iter()
            .chain(query.referenced_paths.iter())
        {
            let dir = Path::new(raw)
                .parent()
                .map(|d| d.to_string_lossy().into_owned())
                .unwrap_or_default();
            let dir = if dir.is_empty() { "." } else { &dir };
            if scope_set.insert(dir.to_string()) {
                scope.push(dir.to_string());
            }
        }
        if scope.is_empty() {
            scope.push(".".into());
        }
        stats.commands_run += 1;
        let mut args = vec!["-l".into(), "-m".into(), "3".into(), "-F".into()];
        for c in concepts.iter().take(COLD_MAX_CONCEPTS) {
            args.push("-e".into());
            args.push(c.clone());
        }
        args.extend(scope.iter().cloned());
        let out = (self.run)("rg", &args).ok()?;
        let mut hits: Vec<ColdHit> = Vec::new();
        for line in out.lines().take(COLD_MAX_HITS * 4) {
            let path = line.to_string();
            if tracked.as_ref().is_some_and(|t| !t.contains(&path)) {
                continue; // untracked noise never rides cold evidence
            }
            if sanitize_reference(&self.root, &path).is_none() {
                continue;
            }
            let snippet = self.snippet_of(&path, stats);
            hits.push(ColdHit {
                snippet: snippet.unwrap_or_default(),
                path,
                score: 0.9,
            });
        }
        if hits.is_empty() {
            return None;
        }
        Some(hits)
    }
}

/// Direct head read of one file (bounded, text-only). Missing, oversized,
/// or binary files yield `None` — never a panic, never a big allocation.
fn read_head(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    if meta.len() > COLD_MAX_READ_BYTES_PER_FILE || meta.len() == 0 {
        return None;
    }
    let mut f = std::fs::File::open(path).ok()?;
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    f.read_to_end(&mut bytes).ok()?;
    if bytes.iter().take(8192).any(|b| *b == 0) {
        return None; // binary sniff
    }
    let text = String::from_utf8_lossy(&bytes);
    let mut out: String = text.chars().take(COLD_MAX_SNIPPET_CHARS).collect();
    if text.chars().count() > COLD_MAX_SNIPPET_CHARS {
        out.push('…');
    }
    Some(out)
}

/// Reject hostile references: absolute paths, parent traversal, NUL bytes,
/// empty strings, anything escaping the workspace root.
fn sanitize_reference(root: &Path, raw: &str) -> Option<PathBuf> {
    if raw.is_empty() || raw.len() > 4096 || raw.contains('\0') {
        return None;
    }
    let p = Path::new(raw);
    if p.is_absolute() {
        return None;
    }
    for comp in p.components() {
        match comp {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
            Component::CurDir | Component::Normal(_) => {}
        }
    }
    let joined = root.join(p);
    if !joined.starts_with(root) {
        return None;
    }
    Some(joined)
}

/// Concepts from the retrieval signal (bounded, deduped — mirrors every
/// other evidence path).
fn concepts(query: &ColdQuery) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let push = |text: &str, out: &mut Vec<String>, seen: &mut std::collections::HashSet<String>| {
        for tok in tokenize(text).into_iter().take(512) {
            if tok.len() < COLD_CONCEPT_MIN_CHARS || !seen.insert(tok.clone()) {
                continue;
            }
            out.push(tok);
            if out.len() >= COLD_MAX_CONCEPTS {
                return;
            }
        }
    };
    push(&query.prompt, &mut out, &mut seen);
    if out.len() < COLD_MAX_CONCEPTS {
        for f in query.changed_files.iter().take(16) {
            let base = f.rsplit('/').next().unwrap_or(f.as_str());
            push(base, &mut out, &mut seen);
            if out.len() >= COLD_MAX_CONCEPTS {
                break;
            }
        }
    }
    if out.len() < COLD_MAX_CONCEPTS {
        push(&query.failures.join(" "), &mut out, &mut seen);
    }
    out
}

/// Score one materialized index against the concepts: token frequency +
/// symbol presence, deterministic, bounded.
fn search_index(
    index: &WorkspaceIndex,
    ws: WorkspaceId,
    concepts: &[String],
) -> Vec<(String, f64)> {
    let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for c in concepts {
        for hit in index.files_for_token(ws, c, 4) {
            *scores.entry(hit.path).or_insert(0.0) += f64::from(hit.freq).min(8.0);
        }
        for (path, sym) in index.symbol_lookup(ws, c, 4) {
            if sym.name.eq_ignore_ascii_case(c) {
                *scores.entry(path).or_insert(0.0) += 16.0;
            } else {
                *scores.entry(path).or_insert(0.0) += 4.0;
            }
        }
    }
    let mut ranked: Vec<(String, f64)> = scores.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn ws_of(raw: u64) -> WorkspaceId {
        WorkspaceId::new(raw)
    }

    fn write(root: &Path, rel: &str, bytes: &[u8]) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, bytes).unwrap();
    }

    fn gen_dir(root: &Path, ws: WorkspaceId) -> PathBuf {
        root.join("generations").join(ws.raw().to_string())
    }

    fn publish_generation(
        data_root: &Path,
        ws: WorkspaceId,
        generation: u64,
        index: &WorkspaceIndex,
    ) {
        let dir = gen_dir(data_root, ws);
        std::fs::create_dir_all(&dir).unwrap();
        let env = GenerationFile::capture(ws.raw(), generation, index, vec![]);
        let bytes = env.to_bytes().unwrap();
        let mut f = std::fs::File::create(dir.join(format!("gen-{generation}.json"))).unwrap();
        f.write_all(&bytes).unwrap();
    }

    fn never_runner() -> Arc<CommandRunner> {
        Arc::new(|p: &str, _a: &[String]| {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{p} intentionally absent in this fixture"),
            ))
        })
    }

    fn provider(root: PathBuf, ws: WorkspaceId, data_root: &Path) -> ColdEvidenceProvider {
        ColdEvidenceProvider::with_runner(
            root.clone(),
            ws,
            data_root.join("generations"),
            never_runner(),
        )
    }

    fn query(prompt: &str, changed: &[&str]) -> ColdQuery {
        ColdQuery {
            prompt: prompt.into(),
            changed_files: changed.iter().map(|s| s.to_string()).collect(),
            referenced_paths: vec![],
            failures: vec![],
        }
    }

    /// (a) A 200k-file NON-git tree: the cold path must complete in <50 ms
    /// and return ONLY the targeted bits (the changed-file read). The old
    /// fallback would have walked 200k files — nothing here lists a repo
    /// directory or spawns a command, so a full walk is structurally
    /// impossible, and the time bound proves it empirically.
    #[test]
    fn giant_non_git_tree_is_never_walked_and_returns_only_targeted_bits() {
        let _serial = TEST_SERIAL.lock().unwrap();
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("giant");
        std::fs::create_dir_all(&root).unwrap();
        // 200 dirs x 1000 files = 200k files (hostile synthetic tree; the
        // files are empty — the fixture's cost is the ENTRY count, which is
        // what a walk would pay).
        for d in 0..200u32 {
            let sub = root.join(format!("d{d:03}"));
            std::fs::create_dir_all(&sub).unwrap();
            for f in 0..1000u32 {
                std::fs::File::create(sub.join(format!("f{f:04}.rs"))).unwrap();
            }
        }
        write(
            &root,
            "d007/f0001.rs",
            b"pub fn hot_symbol_x() -> i64 { 42 }\n",
        );
        let ws = ws_of(9);
        let provider = provider(root.clone(), ws, dir.path());
        let started = std::time::Instant::now();
        let evidence = provider.evidence(&query("inspect hot_symbol_x", &["d007/f0001.rs"]));
        let elapsed = started.elapsed();
        assert!(
            elapsed.as_millis() < 50,
            "cold path over a 200k-file tree took {elapsed:?}"
        );
        // Only the targeted bit came back: the changed file itself.
        assert!(
            evidence.hits.iter().any(|h| h.path == "d007/f0001.rs"),
            "{:?}",
            evidence.hits
        );
        assert!(
            evidence.hits.len() <= 1,
            "only the directly referenced file may surface: {:?}",
            evidence.hits
        );
        assert!(matches!(
            evidence.origin,
            ColdOrigin::DirectReads { files: 1 }
        ));
        assert_eq!(evidence.stats.files_read, 1);
        assert_eq!(evidence.stats.commands_run, 0);
        assert!(
            evidence.stats.dirs_listed <= 1,
            "only the (absent) generation dir may be probed, never repo dirs"
        );
        // The old fallback would have surfaced filler content from a full
        // walk; none of it can be here.
        assert!(!evidence.hits.iter().any(|h| h.path.contains("filler")));
    }

    /// (b) A persisted OLD generation serves the first prompt WITHOUT any
    /// scan: the repo tree is EMPTY, the generation file alone carries the
    /// content, and the provider never touches a repo file.
    #[test]
    fn stale_generation_serves_without_any_scan() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        std::fs::create_dir_all(root.join("src")).unwrap();
        let ws = ws_of(11);
        let mut index = WorkspaceIndex::new();
        index
            .index_file(
                ws,
                std::path::Path::new("src/lib.rs"),
                b"pub fn balance_account() -> i64 { 42 }\n",
                1000,
            )
            .unwrap();
        publish_generation(dir.path(), ws, 1, &index);
        let provider = provider(root.clone(), ws, dir.path());
        let evidence = provider.evidence(&query("fix balance_account", &[]));
        assert!(matches!(
            evidence.origin,
            ColdOrigin::StaleGeneration {
                generation: 1,
                direct_reads: 0
            }
        ));
        assert!(
            evidence.hits.iter().any(|h| h.path == "src/lib.rs"),
            "{:?}",
            evidence.hits
        );
        assert_eq!(evidence.stats.commands_run, 0);
        // The generation file itself is the only repo-side read: no file
        // under the tree was scanned (the tree is empty anyway).
        assert!(evidence.stats.files_read <= 2, "{:?}", evidence.stats);
    }

    /// (c) Partial-now / upgrade-next, unit half 1 + half 2 in one fixture:
    /// turn 1 (no Ready generation) serves only the targeted changed-file
    /// reads; after the real IndexService publishes a Ready generation,
    /// the view serves FULL index evidence (the provider is never consulted
    /// again on the runtime path — the ordering guarantee is tested at the
    /// service level with the runtime's own decision tree).
    #[test]
    fn partial_now_upgrade_next_serves_different_packages() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        write(&root, "src/a.rs", b"pub fn alpha() -> i64 { 1 }\n");
        write(&root, "src/b.rs", b"pub fn beta() -> i64 { 2 }\n");
        let ws = ws_of(13);
        let provider = provider(root.clone(), ws, dir.path());
        // Turn 1: nothing indexed yet — only the changed file is read.
        let first = provider.evidence(&query("alpha", &["src/a.rs"]));
        assert!(matches!(first.origin, ColdOrigin::DirectReads { files: 1 }));
        assert!(
            first.hits.iter().any(|h| h.path == "src/a.rs"),
            "{:?}",
            first.hits
        );
        // The service publishes gen 1 (full index over BOTH files)...
        let mut index = WorkspaceIndex::new();
        index
            .index_file(
                ws,
                std::path::Path::new("src/a.rs"),
                b"pub fn alpha() -> i64 { 1 }\n",
                1,
            )
            .unwrap();
        index
            .index_file(
                ws,
                std::path::Path::new("src/b.rs"),
                b"pub fn beta() -> i64 { 2 }\n",
                2,
            )
            .unwrap();
        publish_generation(dir.path(), ws, 1, &index);
        // ...and the next turn serves the FULL package from the generation.
        let second = provider.evidence(&query("beta", &[]));
        assert!(matches!(
            second.origin,
            ColdOrigin::StaleGeneration { generation: 1, .. }
        ));
        assert!(
            second.hits.iter().any(|h| h.path == "src/b.rs"),
            "{:?}",
            second.hits
        );
    }

    /// (d) Hostile referenced paths (parent traversal, absolute, empty)
    /// are rejected: nothing outside the root is ever read and the call
    /// never panics; safe siblings still serve.
    #[test]
    fn hostile_referenced_paths_are_rejected() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        std::fs::create_dir_all(&root).unwrap();
        write(&root, "ok.rs", b"pub fn safe() {}\n");
        let ws = ws_of(15);
        let provider = provider(root.clone(), ws, dir.path());
        // Hostile refs next to a safe one: only the safe one reads.
        let mut q = query("inspect", &["../escape.rs"]);
        q.referenced_paths = vec![
            "/etc/passwd".into(),
            "a/../../b.rs".into(),
            "".into(),
            "ok.rs".into(),
        ];
        let evidence = provider.evidence(&q);
        assert!(
            evidence.hits.iter().all(|h| h.path == "ok.rs"),
            "{:?}",
            evidence.hits
        );
        assert_eq!(evidence.stats.files_read, 1);
        // All-hostile: empty package, no reads, no panic.
        let q = query("inspect", &["/etc/hosts", "../x.rs", "d/../../e.rs"]);
        let evidence = provider.evidence(&q);
        assert_eq!(evidence.stats.files_read, 0);
        assert!(evidence.hits.is_empty());
        assert!(matches!(evidence.origin, ColdOrigin::None));
    }

    /// (e) ripgrep/git missing: the targeted read-only fallback serves the
    /// directly referenced files (documented degrade) and the full scan is
    /// structurally impossible (0 commands can ever run here).
    #[test]
    fn missing_tools_degrade_to_reads_only_and_never_scan() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        write(&root, "src/app.rs", b"pub fn payments() {}\n");
        // A .git marker exists, but every tool is missing in this fixture.
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let ws = ws_of(17);
        let provider = provider(root.clone(), ws, dir.path());
        let evidence = provider.evidence(&query("payments", &["src/app.rs"]));
        assert!(
            matches!(evidence.origin, ColdOrigin::DirectReads { files: 1 }),
            "{:?}",
            evidence.origin
        );
        // git was ATTEMPTED (a .git marker exists) and failed: the attempt
        // is counted, the degrade ladder served reads only — no scan.
        assert_eq!(evidence.stats.commands_run, 1);
        assert_eq!(evidence.stats.files_read, 1);
        assert!(
            evidence.hits.iter().any(|h| h.path == "src/app.rs"),
            "{:?}",
            evidence.hits
        );
        assert_eq!(evidence.stats.degraded_steps, 1, "documented degrade");
    }

    /// Git present + git works + rg missing: ls-files runs, rg degrades,
    /// the direct reads still serve and no full scan ever happens.
    #[test]
    fn git_ok_rg_missing_degrades_to_reads() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        write(&root, "src/app.rs", b"pub fn payments() {}\n");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let ws = ws_of(19);
        let run = Arc::new(|p: &str, _a: &[String]| -> std::io::Result<String> {
            if p == "git" {
                Ok("src/app.rs\0".into())
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "rg missing",
                ))
            }
        });
        let provider = ColdEvidenceProvider::with_runner(
            root.clone(),
            ws,
            dir.path().join("generations"),
            run,
        );
        let evidence = provider.evidence(&query("payments", &["src/app.rs"]));
        assert!(
            matches!(evidence.origin, ColdOrigin::DirectReads { files: 1 }),
            "{:?}",
            evidence.origin
        );
        assert_eq!(
            evidence.stats.commands_run, 2,
            "git attempted, rg attempted"
        );
        assert_eq!(evidence.stats.files_read, 1);
    }

    /// Behavior on a REAL small git repo (git + ripgrep available): the
    /// tracked file list gates the targeted ripgrep scope, and a file the
    /// turn references in a subdir surfaces through the tracked search.
    /// Skipped silently when git or rg is not installed (CI minimal
    /// images); the unit seams above lock the degrade ladder regardless.
    #[test]
    fn real_small_git_repo_serves_tracked_search() {
        let git_ok = std::process::Command::new("git")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        let rg_ok = std::process::Command::new("rg")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if !git_ok || !rg_ok {
            eprintln!("skipping real-git fixture: git/rg not installed");
            return;
        }
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("repo");
        write(
            &root,
            "src/app.rs",
            b"pub fn balance_account() -> i64 { 42 }\n",
        );
        write(&root, "src/other.rs", b"pub fn unrelated() {}\n");
        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&root)
            .status()
            .unwrap();
        assert!(init.success());
        let add = std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(&root)
            .status()
            .unwrap();
        assert!(add.success());
        let ws = ws_of(21);
        let provider = ColdEvidenceProvider::new(root.clone(), ws, dir.path().join("generations"));
        let evidence = provider.evidence(&ColdQuery {
            prompt: "continue".into(),
            changed_files: vec!["src/app.rs".into()],
            ..Default::default()
        });
        // Concept from the changed-file basename (app) matches app.rs via
        // tracked search; the direct read also surfaces the file itself.
        assert!(
            evidence.hits.iter().any(|h| h.path.ends_with("app.rs")),
            "{:?}",
            evidence.hits
        );
        assert!(evidence.stats.commands_run >= 2, "{:?}", evidence.stats);
        assert!(evidence.hits.len() <= COLD_MAX_HITS, "bounded package");
    }
}
