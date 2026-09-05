//! Lazy instructions + skills (audit: import rules WITHOUT dumping every
//! rule into every prompt; conditional activation; instruction-epoch
//! detection; skills that cost zero prompt tokens until loaded).
//!
//! Discovery is cheap and pure; only `active_for` produces output and only
//! `load_skill` reads full skill bodies.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// Deterministic precedence (higher wins when both are active). Faktor
/// native rules outrank every imported convention.
pub const PRECEDENCE: [(RuleSourceKind, u8); 9] = [
    (RuleSourceKind::FaktorNative, 100),
    (RuleSourceKind::AgentsMd, 90),
    (RuleSourceKind::ClaudeMd, 80),
    (RuleSourceKind::GeminiMd, 70),
    (RuleSourceKind::CopilotInstructions, 60),
    (RuleSourceKind::CursorRules, 50),
    (RuleSourceKind::WindsurfRules, 40),
    (RuleSourceKind::ContinueRules, 30),
    (RuleSourceKind::LegacyKiloRules, 10),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleSourceKind {
    AgentsMd,
    ClaudeMd,
    GeminiMd,
    CopilotInstructions,
    CursorRules,
    WindsurfRules,
    ContinueRules,
    FaktorNative,
    LegacyKiloRules,
}

impl RuleSourceKind {
    pub fn priority(self) -> u8 {
        PRECEDENCE
            .iter()
            .find(|(k, _)| *k == self)
            .map(|(_, p)| *p)
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Instruction {
    pub source: RuleSourceKind,
    pub path: String,
    pub scope: String,
    pub content: String,
    pub hash: u64,
    pub priority: u8,
    pub reason_loaded: String,
}

pub const MAX_RULE_BYTES: usize = 64 * 1024;
const MAX_WALK_DEPTH: usize = 16;

/// The convention directory (relative to root) each kind lives under.
fn convention_root_for(kind: RuleSourceKind) -> &'static str {
    match kind {
        RuleSourceKind::CursorRules => ".cursor/rules",
        RuleSourceKind::WindsurfRules => ".windsurf/rules",
        RuleSourceKind::ContinueRules => ".continue/rules",
        RuleSourceKind::FaktorNative => ".faktor/rules",
        RuleSourceKind::LegacyKiloRules => ".faktor/legacy",
        _ => "",
    }
}

fn hash_of(path: &Path, content: &str) -> u64 {
    let mut h = DefaultHasher::new();
    path.hash(&mut h);
    content.hash(&mut h);
    h.finish()
}

fn read_bounded(path: &Path, cap: usize) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(String::from_utf8_lossy(&bytes[..bytes.len().min(cap)]).into_owned())
}

/// Top-level well-known rule files.
fn top_level_candidates(root: &Path) -> Vec<(RuleSourceKind, PathBuf)> {
    let mut v = Vec::new();
    for (kind, names) in [
        (RuleSourceKind::FaktorNative, &["FAKTOR.md"][..]),
        (RuleSourceKind::AgentsMd, &["AGENTS.md"][..]),
        (RuleSourceKind::ClaudeMd, &["CLAUDE.md"][..]),
        (RuleSourceKind::GeminiMd, &["GEMINI.md"][..]),
        (
            RuleSourceKind::CopilotInstructions,
            &[".github/copilot-instructions.md"][..],
        ),
        (
            RuleSourceKind::WindsurfRules,
            &[".windsurfrules", ".windsurf/rules.md"][..],
        ),
    ] {
        for n in names {
            let p = root.join(n);
            if p.is_file() {
                v.push((kind, p));
            }
        }
    }
    v
}

/// Directory rule globs: every `*.md` under the dir with the dir path as
/// its activation scope.
fn dir_candidates(root: &Path) -> Vec<(RuleSourceKind, PathBuf)> {
    let mut v = Vec::new();
    for (kind, rel) in [
        (RuleSourceKind::CursorRules, ".cursor/rules"),
        (RuleSourceKind::WindsurfRules, ".windsurf/rules"),
        (RuleSourceKind::ContinueRules, ".continue/rules"),
        (RuleSourceKind::FaktorNative, ".faktor/rules"),
        (RuleSourceKind::LegacyKiloRules, ".faktor/legacy"),
    ] {
        let dir = root.join(rel);
        if !dir.is_dir() {
            continue;
        }
        // Depth is tracked PER DIRECTORY as (dir, depth-below-root) pairs
        // (audit 32): a hostile deep branch cuts itself off at
        // MAX_WALK_DEPTH without consuming the allowance of sibling
        // branches — a shallow file next to a 30-deep tree is still
        // discovered.
        let mut walk: VecDeque<(PathBuf, usize)> = VecDeque::from([(dir.clone(), 0usize)]);
        while let Some((d, depth)) = walk.pop_front() {
            if depth > MAX_WALK_DEPTH {
                continue;
            }
            let Ok(entries) = std::fs::read_dir(&d) else {
                continue;
            };
            let mut names: Vec<PathBuf> = Vec::new();
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk.push_back((p, depth + 1));
                } else if p
                    .extension()
                    .map(|x| x == "md" || x == "mdc")
                    .unwrap_or(false)
                {
                    names.push(p);
                }
            }
            names.sort();
            for p in names {
                v.push((kind, p));
            }
        }
    }
    v
}

/// Every rule file discoverable under `root` (bounded, deterministic).
pub fn discover_rule_files(root: &Path) -> Vec<(RuleSourceKind, PathBuf)> {
    let mut all = top_level_candidates(root);
    all.extend(dir_candidates(root));
    // Deterministic: by kind priority desc, then path asc.
    all.sort_by(|a, b| {
        b.0.priority()
            .cmp(&a.0.priority())
            .then_with(|| a.1.cmp(&b.1))
    });
    all
}

/// Loaded rule tree. `active_for` returns only rules whose scope/keywords
/// match the prompt or the touched files.
pub struct Instructions {
    rules: Vec<Instruction>,
    epoch: u64,
    root: PathBuf,
}

impl Instructions {
    pub fn load(root: &Path) -> Self {
        let rules = load_rules(root);
        let epoch = Self::compute_epoch(&rules);
        Self {
            rules,
            epoch,
            root: root.to_path_buf(),
        }
    }

    /// Strict epoch-pinned load: loads the LIVE tree and refuses — with a
    /// typed [`EnvSnapshotError::EpochMismatch`] — when the live
    /// instruction epoch differs from the pinned `expected_epoch`. There is
    /// never a silent drift: a caller that requires the environment of a
    /// snapshot must read the snapshot (audit 97), never fall back to
    /// whatever the filesystem holds now.
    pub fn load_at_epoch(root: &Path, expected_epoch: u64) -> Result<Self, EnvSnapshotError> {
        let live = Self::load(root);
        if live.epoch == expected_epoch {
            Ok(live)
        } else {
            Err(EnvSnapshotError::EpochMismatch {
                expected: expected_epoch,
                actual: live.epoch,
            })
        }
    }

    /// Build an epoch-pinned instruction tree from one immutable
    /// [`EnvSnapshot`] and its captured file contents. Reads ONLY the
    /// snapshot: the live filesystem is never consulted, so rules the
    /// parent changed after the snapshot was taken can never bleed into a
    /// child bound to it. Every entry is verified against the snapshot's
    /// recorded hashes (missing content or tampered bytes are loud typed
    /// errors, never a silent skip or a live fallback).
    pub fn from_snapshot(
        snap: &EnvSnapshot,
        content: &BTreeMap<String, String>,
    ) -> Result<Self, EnvSnapshotError> {
        // Structural validation FIRST: every recorded path must be a rule
        // file a discovery pass could produce (hostile rows are malformed
        // no matter what their hashes claim), then the content verification.
        for rel in snap.workspace_paths.keys() {
            let rel_path = Path::new(rel);
            kind_for_rel_path(rel_path).ok_or_else(|| {
                EnvSnapshotError::Malformed(format!(
                    "snapshot path {rel:?} is not a rule file any discovery pass could produce"
                ))
            })?;
        }
        verify_snapshot_content(snap, content)?;
        let root = snap.root.clone();
        let mut rules = Vec::new();
        for (rel, rec) in &snap.workspace_paths {
            let rel_path = Path::new(rel);
            let kind = kind_for_rel_path(rel_path).expect("validated above");
            let Some(text) = content.get(rel) else {
                return Err(EnvSnapshotError::Missing(format!(
                    "snapshot content for {rel:?}"
                )));
            };
            let abs = root.join(rel_path);
            rules.push(Instruction {
                source: kind,
                path: rel.clone(),
                scope: scope_of(&root, kind, &abs),
                content: text.clone(),
                hash: rec.rules_hash,
                priority: kind.priority(),
                reason_loaded: String::new(),
            });
        }
        Ok(Self {
            rules,
            epoch: snap.instruction_epoch,
            root,
        })
    }

    fn compute_epoch(rules: &[Instruction]) -> u64 {
        let mut h = DefaultHasher::new();
        for r in rules {
            r.path.hash(&mut h);
            r.hash.hash(&mut h);
        }
        h.finish()
    }

    /// True when any rule file changed since load (instruction epoch —
    /// stale rules must never silently govern a long task).
    pub fn reload_if_changed(&mut self) -> bool {
        let fresh = Self::load(&self.root);
        if fresh.epoch != self.epoch {
            *self = fresh;
            return true;
        }
        false
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Conditional activation: Faktor-native top-level + AGENTS.md always
    /// load; scoped rules activate when the prompt or a touched file
    /// matches their scope path or a `# Scope:` keyword directive.
    pub fn active_for(&self, prompt: &str, touched: &[String]) -> Vec<Instruction> {
        let mut out = Vec::new();
        for r in &self.rules {
            let top = r.scope.is_empty();
            let mut reason = None;
            if top
                && matches!(
                    r.source,
                    RuleSourceKind::FaktorNative | RuleSourceKind::AgentsMd
                )
            {
                reason = Some("always".to_string());
            } else {
                // Directory containment of a touched file activates
                // (rule's own dir path as scope).
                if touched.iter().any(|t| t.starts_with(&r.scope)) {
                    reason = Some(format!("scope:{}", r.scope));
                }
            }
            if reason.is_none() {
                // Keyword directives: leading "# Scope: a,b" lines.
                for line in r.content.lines().take(8) {
                    if let Some(kws) = line.trim_start().strip_prefix("# Scope:") {
                        for kw in kws.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                            if prompt.to_lowercase().contains(&kw.to_lowercase()) {
                                reason = Some(format!("keyword:{kw}"));
                                break;
                            }
                        }
                    }
                    if reason.is_some() {
                        break;
                    }
                }
            }
            if let Some(reason_loaded) = reason {
                let mut i = r.clone();
                i.reason_loaded = reason_loaded;
                out.push(i);
            }
        }
        out
    }
}

// -------------------------------------------------------- env snapshots (audit 97)

/// Hard cap on the number of rule files one env snapshot may capture. A
/// discovery that would exceed it is a typed [`EnvSnapshotError::Oversized`]
/// — snapshots never silently drop files (bounded everything).
pub const MAX_SNAPSHOT_PATHS: usize = 64;
/// Hard cap on the TOTAL captured rule bytes of one snapshot (per-file
/// reads are already bounded by [`MAX_RULE_BYTES`]). Exceeding it is a
/// typed [`EnvSnapshotError::Oversized`], never a silent truncation.
pub const MAX_SNAPSHOT_TOTAL_BYTES: usize = 512 * 1024;
/// Hard cap on one workspace-relative rule path inside a snapshot.
pub const MAX_SNAPSHOT_PATH_CHARS: usize = 1024;
/// Hard cap on one snapshot id (chars).
pub const MAX_SNAPSHOT_ID_CHARS: usize = 128;

/// Typed failures of the snapshot machinery (audit 97). Every variant is a
/// LOUD refusal — there is never a silent fallback to the live environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvSnapshotError {
    Oversized(String),
    Malformed(String),
    Missing(String),
    Tampered(String),
    EpochMismatch { expected: u64, actual: u64 },
}

impl fmt::Display for EnvSnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Oversized(m)
            | Self::Malformed(m)
            | Self::Missing(m)
            | Self::Tampered(m) => write!(f, "{m}"),
            Self::EpochMismatch { expected, actual } => write!(
                f,
                "live instruction epoch {actual} differs from the required epoch {expected}; the pinned snapshot must be read, never the live env"
            ),
        }
    }
}

impl std::error::Error for EnvSnapshotError {}

/// One captured rule file: `rules_hash` hashes (path, content) exactly like
/// the loaded [`Instruction`] hash (so the snapshot epoch equals the epoch
/// a live load of the same tree computes); `bytes_hash` identifies the
/// content bytes alone (the content-dedup identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EnvFileRecord {
    pub rules_hash: u64,
    pub bytes_hash: u64,
}

/// An IMMUTABLE, epoch-pinned capture of a directory's rule environment
/// (repo knowledge / AGENTS.md / instructions), taken once at child spawn.
/// A child bound to it reads ONLY these rules for the rest of its life —
/// later changes to the parent's environment never bleed into it. Paths are
/// workspace-relative (lossy strings, map keys) so the snapshot is portable
/// across roots; `root` is the capture root used to recompute the recorded
/// path-content hashes.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EnvSnapshot {
    pub snapshot_id: String,
    /// Deterministic over the ordered (path, rules_hash) pairs: identical
    /// to the epoch `Instructions::load(root)` computes over the same tree.
    pub instruction_epoch: u64,
    /// The root directory the snapshot was captured from.
    pub root: PathBuf,
    /// Workspace-relative rule path -> recorded hashes, sorted by path.
    pub workspace_paths: BTreeMap<String, EnvFileRecord>,
    pub taken_ms: i64,
}

impl EnvSnapshot {
    /// Capture the rule environment of `root` exactly as
    /// [`Instructions::load`] would see it (same discovery, same per-file
    /// read bound, same ordering — so the snapshot epoch matches a live
    /// load of an unchanged tree). Bounded: at most [`MAX_SNAPSHOT_PATHS`]
    /// rule files and at most [`MAX_SNAPSHOT_TOTAL_BYTES`] total captured
    /// bytes; beyond either the capture fails with a typed Oversized error
    /// and returns NOTHING (never a silently truncated snapshot). Returns
    /// the snapshot plus the captured file contents keyed by the same
    /// workspace-relative paths (contents are what a pinned read serves;
    /// they are stored separately so identical content is stored once).
    pub fn capture(
        root: &Path,
        snapshot_id: &str,
        taken_ms: i64,
    ) -> Result<CapturedEnv, EnvSnapshotError> {
        if !root.is_dir() {
            return Err(EnvSnapshotError::Malformed(format!(
                "snapshot root {:?} is not a directory",
                root
            )));
        }
        if snapshot_id.is_empty()
            || snapshot_id.chars().count() > MAX_SNAPSHOT_ID_CHARS
            || !snapshot_id.is_ascii()
            || snapshot_id.contains('/')
            || snapshot_id.contains('\\')
        {
            return Err(EnvSnapshotError::Malformed(format!(
                "snapshot id {snapshot_id:?} must be 1..={MAX_SNAPSHOT_ID_CHARS} ASCII characters without '/' or '\\'"
            )));
        }
        let discovered = discover_rule_files(root);
        let mut workspace_paths = BTreeMap::new();
        let mut content = BTreeMap::new();
        let mut total_bytes = 0usize;
        let mut epoch = DefaultHasher::new();
        for (_kind, path) in discovered {
            let Some(rel) = path.strip_prefix(root).ok() else {
                continue;
            };
            let rel_str = rel.to_string_lossy().into_owned();
            if rel_str.chars().count() > MAX_SNAPSHOT_PATH_CHARS {
                return Err(EnvSnapshotError::Oversized(format!(
                    "rule path {rel_str:?} exceeds {MAX_SNAPSHOT_PATH_CHARS} characters"
                )));
            }
            if workspace_paths.len() >= MAX_SNAPSHOT_PATHS {
                return Err(EnvSnapshotError::Oversized(format!(
                    "rule environment of {:?} has more than {MAX_SNAPSHOT_PATHS} rule files",
                    root
                )));
            }
            let Some(text) = read_bounded(&path, MAX_RULE_BYTES) else {
                // Identical skip semantics to Instructions::load: an
                // unreadable rule file is not part of the loaded env.
                continue;
            };
            if total_bytes.saturating_add(text.len()) > MAX_SNAPSHOT_TOTAL_BYTES {
                return Err(EnvSnapshotError::Oversized(format!(
                    "rule environment of {:?} exceeds {MAX_SNAPSHOT_TOTAL_BYTES} total bytes",
                    root
                )));
            }
            total_bytes += text.len();
            let rules_hash = hash_of(&path, &text);
            let bytes_hash = fnv1a64(text.as_bytes());
            // instruction_epoch must hash the same (rel path, rules hash)
            // pairs in the same order as compute_epoch over loaded rules.
            rel_str.hash(&mut epoch);
            rules_hash.hash(&mut epoch);
            workspace_paths.insert(
                rel_str.clone(),
                EnvFileRecord {
                    rules_hash,
                    bytes_hash,
                },
            );
            content.insert(rel_str, text);
        }
        Ok(CapturedEnv {
            snapshot: EnvSnapshot {
                snapshot_id: snapshot_id.to_string(),
                instruction_epoch: epoch.finish(),
                root: root.to_path_buf(),
                workspace_paths,
                taken_ms,
            },
            content,
        })
    }
}

/// A captured snapshot plus the file contents needed to serve a pinned
/// read. Contents are deliberately kept separate from the (small) snapshot
/// so a durable store can deduplicate unchanged content by hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedEnv {
    pub snapshot: EnvSnapshot,
    /// Workspace-relative path -> captured file content.
    pub content: BTreeMap<String, String>,
}

/// Verify that `content` satisfies EVERY entry of the snapshot: every
/// recorded path has content, the content bytes hash to the recorded
/// `bytes_hash`, and (path, content) hashes to the recorded `rules_hash`.
/// A missing or tampered entry is a loud typed error — pinned reads never
/// skip an entry and never fall back to the live filesystem.
pub fn verify_snapshot_content(
    snap: &EnvSnapshot,
    content: &BTreeMap<String, String>,
) -> Result<(), EnvSnapshotError> {
    for (rel, rec) in &snap.workspace_paths {
        let Some(text) = content.get(rel) else {
            return Err(EnvSnapshotError::Missing(format!(
                "content for rule path {rel:?}"
            )));
        };
        if fnv1a64(text.as_bytes()) != rec.bytes_hash {
            return Err(EnvSnapshotError::Tampered(format!(
                "content bytes of {rel:?} do not match the recorded bytes hash"
            )));
        }
        if hash_of(&snap.root.join(rel), text) != rec.rules_hash {
            return Err(EnvSnapshotError::Tampered(format!(
                "content of {rel:?} does not match the recorded (path, content) rules hash"
            )));
        }
    }
    Ok(())
}

/// Deterministic content-only hash (FNV-1a 64): stable across processes
/// and Rust versions, so content-dedup keys persist across restarts.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Load every rule of `root` in discovery order (the epoch input order).
fn load_rules(root: &Path) -> Vec<Instruction> {
    let mut rules = Vec::new();
    for (kind, path) in discover_rule_files(root) {
        let Some(content) = read_bounded(&path, MAX_RULE_BYTES) else {
            continue;
        };
        rules.push(Instruction {
            source: kind,
            path: path
                .strip_prefix(root)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| path.to_string_lossy().into_owned()),
            scope: scope_of(root, kind, &path),
            content: content.clone(),
            hash: hash_of(&path, &content),
            priority: kind.priority(),
            reason_loaded: String::new(),
        });
    }
    rules
}

/// The activation scope of one rule file (workspace-relative subdirectory
/// under its convention root; top-level rules carry an empty scope).
fn scope_of(root: &Path, kind: RuleSourceKind, path: &Path) -> String {
    let rel = path
        .parent()
        .and_then(|p| p.strip_prefix(root).ok())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    // Strip the convention root so activation uses the workspace-relative
    // subdir (frontend/App.tsx -> frontend).
    let convention = convention_root_for(kind);
    match rel.strip_prefix(convention) {
        Some("") => String::new(),
        Some(rest) => rest.trim_start_matches('/').to_string(),
        None => rel,
    }
}

/// The rule kind a workspace-relative path carries — exactly the
/// membership of [`discover_rule_files`]: top-level well-known names plus
/// `*.md`/`*.mdc` files under the convention directories. Any other path
/// is `None`: a pinned snapshot entry that maps to `None` is malformed
/// (hostile rows can never become rules).
pub fn kind_for_rel_path(rel: &Path) -> Option<RuleSourceKind> {
    let os = rel.as_os_str().to_str()?;
    for (kind, names) in [
        (RuleSourceKind::FaktorNative, &["FAKTOR.md"][..]),
        (RuleSourceKind::AgentsMd, &["AGENTS.md"][..]),
        (RuleSourceKind::ClaudeMd, &["CLAUDE.md"][..]),
        (RuleSourceKind::GeminiMd, &["GEMINI.md"][..]),
        (
            RuleSourceKind::CopilotInstructions,
            &[".github/copilot-instructions.md"][..],
        ),
        (
            RuleSourceKind::WindsurfRules,
            &[".windsurfrules", ".windsurf/rules.md"][..],
        ),
    ] {
        for n in names {
            if os == *n {
                return Some(kind);
            }
        }
    }
    let is_md = matches!(rel.extension().and_then(|e| e.to_str()), Some("md" | "mdc"));
    if is_md {
        for (kind, dir) in [
            (RuleSourceKind::CursorRules, ".cursor/rules"),
            (RuleSourceKind::WindsurfRules, ".windsurf/rules"),
            (RuleSourceKind::ContinueRules, ".continue/rules"),
            (RuleSourceKind::FaktorNative, ".faktor/rules"),
            (RuleSourceKind::LegacyKiloRules, ".faktor/legacy"),
        ] {
            if os == dir || os.starts_with(&format!("{dir}/")) {
                return Some(kind);
            }
        }
    }
    None
}

/// Skills: metadata only at discovery; bodies load on demand.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    pub path: String,
    pub summary: String,
    pub keywords: Vec<String>,
}

pub const MAX_SKILL_ENTRIES: usize = 2000;
const SKILL_SUMMARY_CAP: usize = 400;
const SKILL_BODY_CAP: usize = 64 * 1024;

pub struct SkillRegistry {
    entries: Vec<SkillMeta>,
}

impl SkillRegistry {
    pub fn discover(root: &Path) -> Self {
        let mut entries = Vec::new();
        for dir in [root.join(".faktor/skills"), root.join(".claude/skills")] {
            let Ok(rd) = std::fs::read_dir(&dir) else {
                continue;
            };
            let mut names: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            names.sort();
            for n in names {
                if entries.len() >= MAX_SKILL_ENTRIES {
                    break;
                }
                let meta_path = n.join("SKILL.md");
                let Some(raw) = read_bounded(&meta_path, SKILL_SUMMARY_CAP) else {
                    continue;
                };
                let name = n
                    .file_name()
                    .map(|f| f.to_string_lossy().into_owned())
                    .unwrap_or_default();
                // Frontmatter-lite summary: first heading/paragraph + any
                // keywords line. Never the full body.
                let mut description = String::new();
                let mut keywords = Vec::new();
                for line in raw.lines().take(6) {
                    if let Some(d) = line.strip_prefix("# ") {
                        if description.is_empty() {
                            description = d.trim().to_string();
                        }
                    }
                    if let Some(k) = line.strip_prefix("## Keywords:") {
                        keywords = k
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect();
                    }
                }
                entries.push(SkillMeta {
                    name,
                    description,
                    path: meta_path.to_string_lossy().into_owned(),
                    summary: raw,
                    keywords,
                });
            }
        }
        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn find(&self, query: &str) -> Vec<&SkillMeta> {
        let q = query.to_lowercase();
        let mut hits: Vec<(&SkillMeta, u8)> = Vec::new();
        for e in &self.entries {
            let mut score = 0u8;
            if e.name.to_lowercase().contains(&q) {
                score += 4;
            }
            if e.description.to_lowercase().contains(&q) {
                score += 2;
            }
            if e.keywords.iter().any(|k| q.contains(&k.to_lowercase())) {
                score += 3;
            }
            if score > 0 {
                hits.push((e, score));
            }
        }
        hits.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.name.cmp(&b.0.name)));
        hits.into_iter().map(|(e, _)| e).take(10).collect()
    }

    /// Full body ONLY on demand (bounded). None when absent/oversized.
    pub fn load_skill(&self, name: &str) -> Option<String> {
        let e = self.entries.iter().find(|e| e.name == name)?;
        read_bounded(Path::new(&e.path), SKILL_BODY_CAP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, content: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, content).unwrap();
    }

    #[test]
    fn precedence_agents_over_claude_conflict() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: no unsafe\n");
        write(d.path(), "CLAUDE.md", "always: use unsafe everywhere\n");
        let ins = Instructions::load(d.path());
        let active = ins.active_for("do it", &[]);
        assert_eq!(
            active.len(),
            1,
            "only AGENTS loads at top level: {active:?}"
        );
        assert!(active[0].content.contains("no unsafe"));
        assert_eq!(active[0].source, RuleSourceKind::AgentsMd);
    }

    #[test]
    fn faktor_native_outranks_agents() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "FAKTOR.md", "faktor rules\n");
        write(d.path(), "AGENTS.md", "agents rules\n");
        let ins = Instructions::load(d.path());
        let active = ins.active_for("x", &[]);
        let first = active
            .iter()
            .find(|i| i.content.contains("faktor rules"))
            .unwrap();
        assert_eq!(first.source, RuleSourceKind::FaktorNative);
        assert!(active.iter().any(|i| i.content.contains("agents rules")));
    }

    #[test]
    fn conditional_scope_activation() {
        let d = tempfile::tempdir().unwrap();
        // A top-level convention rule needs a keyword directive; a rule in
        // a SUBDIRECTORY activates when a touched file lives under it.
        write(
            d.path(),
            ".cursor/rules/backend.mdc",
            "# Scope: api\nbackend rules\n",
        );
        write(
            d.path(),
            ".cursor/rules/frontend/ux.mdc",
            "frontend style\n",
        );
        let ins = Instructions::load(d.path());
        let active = ins.active_for("fix the api server", &[]);
        assert!(
            active.iter().any(|i| i.path.contains("backend")),
            "keyword directive activates the top-level rule"
        );
        assert!(
            !active.iter().any(|i| i.path.contains("frontend")),
            "unrelated rules never load"
        );
        // Touching a file under the subdirectory scope activates it.
        let touched = ins.active_for("anything", &["frontend/App.tsx".into()]);
        assert!(
            touched.iter().any(|i| i.path.contains("frontend")),
            "subdirectory scope activates on contained touched files"
        );
    }

    #[test]
    fn keyword_directive_activation() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            ".cursor/rules/db.mdc",
            "# Scope: postgres, sql\nuse pools\n",
        );
        let ins = Instructions::load(d.path());
        assert!(ins
            .active_for("postgres connection", &[])
            .iter()
            .any(|i| i.path.contains("db")));
        assert!(!ins
            .active_for("frontend colors", &[])
            .iter()
            .any(|i| i.path.contains("db")));
        let a = ins.active_for("sql tuning", &[]);
        assert!(a[0].reason_loaded.contains("keyword:sql"));
    }

    #[test]
    fn epoch_flips_on_change() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "v1 rules\n");
        let mut ins = Instructions::load(d.path());
        let e0 = ins.epoch();
        assert!(!ins.reload_if_changed());
        write(d.path(), "AGENTS.md", "v2 rules (changed mid-task)\n");
        assert!(ins.reload_if_changed(), "stale epoch must be detected");
        assert_ne!(ins.epoch(), e0);
        let active = ins.active_for("x", &[]);
        assert!(active[0].content.contains("v2"));
    }

    #[test]
    fn legacy_import_has_lowest_priority_and_never_always_loads() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            ".faktor/legacy/kilo.md",
            "# Scope: import\nlegacy content\n",
        );
        let ins = Instructions::load(d.path());
        assert!(!ins
            .active_for("anything unrelated", &[])
            .iter()
            .any(|i| i.content.contains("legacy")));
        let a = ins.active_for("import config", &[]);
        assert!(a
            .iter()
            .any(|i| i.source == RuleSourceKind::LegacyKiloRules && i.priority == 10));
    }

    #[test]
    fn skills_cost_zero_prompt_tokens_until_loaded() {
        let d = tempfile::tempdir().unwrap();
        for i in 0..1000 {
            write(
                d.path(),
                &format!(".faktor/skills/skill-{i:04}/SKILL.md"),
                &format!(
                    "# Skill {i}\n## Keywords: kw-{i}\nfull body of skill {i} repeated to length\n"
                ),
            );
        }
        let reg = SkillRegistry::discover(d.path());
        assert_eq!(reg.len(), 1000);
        // Discovery output = metadata summaries only; total metadata bytes
        // stay far below full bodies (each body is 60+ bytes -> full would
        // be >= 60k; summaries capped at 400 each but only 6 lines read).
        let total_meta: usize = reg.entries.iter().map(|e| e.summary.len()).sum();
        assert!(total_meta < 1000 * 120, "summaries bounded");
        // Unrelated query: no hit, nothing loaded.
        assert!(reg.find("completely unrelated").is_empty());
        // Targeted load returns ONLY that skill's body, bounded.
        let body = reg.load_skill("skill-0042").unwrap();
        assert!(
            body.contains("full body of skill 42"),
            "body head: {:?}",
            &body[..body.len().min(120)]
        );
        assert!(body.len() <= SKILL_BODY_CAP);
        assert!(reg.load_skill("missing").is_none());
    }

    #[test]
    fn hostile_deep_walk_is_bounded() {
        let d = tempfile::tempdir().unwrap();
        let mut p = d.path().join(".cursor/rules");
        for _ in 0..40 {
            p = p.join("a");
        }
        std::fs::create_dir_all(&p).unwrap();
        std::fs::write(p.join("deep.md"), "deep\n").unwrap();
        let ins = Instructions::load(d.path());
        assert!(ins.rules.len() <= 1, "depth cap bounds the walk");
    }

    #[test]
    fn per_directory_depth_keeps_shallow_siblings_visible() {
        // Audit 32 regression: depth accounting must be PER DIRECTORY
        // BRANCH, never one global counter shared across sibling branches.
        // A shallow file at depth 5 next to TWO 30-deep hostile trees must
        // still be discovered (the old shared counter let the deep branches
        // consume the whole allowance and cut the shallow one too), while
        // each deep tree is still truncated on its own at MAX_WALK_DEPTH.
        let d = tempfile::tempdir().unwrap();
        for tree in ["deep-a", "deep-b"] {
            let mut p = d.path().join(".cursor/rules").join(tree);
            for _ in 0..30 {
                p = p.join("x");
            }
            std::fs::create_dir_all(&p).unwrap();
            std::fs::write(p.join("bottom.md"), "too deep\n").unwrap();
        }
        let mut shallow = d.path().join(".cursor/rules/shallow");
        for _ in 0..5 {
            shallow = shallow.join("y");
        }
        std::fs::create_dir_all(&shallow).unwrap();
        std::fs::write(shallow.join("near.md"), "shallow rules\n").unwrap();
        let ins = Instructions::load(d.path());
        let near: Vec<&Instruction> = ins
            .rules
            .iter()
            .filter(|r| r.path.contains("near"))
            .collect();
        let bottom: Vec<&Instruction> = ins
            .rules
            .iter()
            .filter(|r| r.path.contains("bottom"))
            .collect();
        assert_eq!(
            near.len(),
            1,
            "the depth-5 sibling file must be discovered: {}",
            ins.rules
                .iter()
                .map(|r| r.path.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert_eq!(
            bottom.len(),
            0,
            "both 30-deep branches must still be cut at MAX_WALK_DEPTH"
        );
    }

    // ------------------------------------------------- env snapshots (audit 97)

    #[test]
    fn snapshot_epoch_equals_live_load_epoch_of_the_same_tree() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: pinned rules v1\n");
        write(
            d.path(),
            ".cursor/rules/frontend/ux.mdc",
            "# Scope: ui\nfrontend style\n",
        );
        let live = Instructions::load(d.path());
        let captured = EnvSnapshot::capture(d.path(), "env-a", 7).unwrap();
        assert_eq!(captured.snapshot.instruction_epoch, live.epoch());
        assert_eq!(captured.snapshot.workspace_paths.len(), 2);
        assert!(captured.content["AGENTS.md"].contains("v1"));
    }

    #[test]
    fn pinned_reads_serve_spawn_time_rules_after_the_parent_changed_them() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: rule V1\n");
        let captured = EnvSnapshot::capture(d.path(), "env-a", 1).unwrap();
        // The parent's AGENTS.md changes AFTER the snapshot was taken.
        write(d.path(), "AGENTS.md", "always: rule V2\n");
        let pinned = Instructions::from_snapshot(&captured.snapshot, &captured.content).unwrap();
        assert_eq!(pinned.epoch(), captured.snapshot.instruction_epoch);
        let active = pinned.active_for("anything", &[]);
        assert!(
            active.iter().any(|i| i.content.contains("rule V1")),
            "the pinned tree must still serve the spawn-time rule: {active:?}"
        );
        assert!(
            !active.iter().any(|i| i.content.contains("rule V2")),
            "later parent changes must never bleed into the pinned tree"
        );
        // The live loader sees the new rule (its own epoch moved) — exactly
        // the drift a pinned read must refuse to take silently.
        let live = Instructions::load(d.path());
        assert!(live.active_for("x", &[])[0].content.contains("rule V2"));
        assert_ne!(live.epoch(), captured.snapshot.instruction_epoch);
    }

    #[test]
    fn two_snapshots_of_different_epochs_stay_consistent_independently() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: epoch-1 rules\n");
        let a = EnvSnapshot::capture(d.path(), "env-a", 1).unwrap();
        write(d.path(), "AGENTS.md", "always: epoch-2 rules\n");
        let b = EnvSnapshot::capture(d.path(), "env-b", 2).unwrap();
        assert_ne!(
            a.snapshot.instruction_epoch, b.snapshot.instruction_epoch,
            "different rules must flip the snapshot epoch"
        );
        let pa = Instructions::from_snapshot(&a.snapshot, &a.content).unwrap();
        let pb = Instructions::from_snapshot(&b.snapshot, &b.content).unwrap();
        let va = pa.active_for("x", &[]);
        let vb = pb.active_for("x", &[]);
        assert!(va[0].content.contains("epoch-1"));
        assert!(vb[0].content.contains("epoch-2"));
        assert_eq!(pa.epoch(), a.snapshot.instruction_epoch);
        assert_eq!(pb.epoch(), b.snapshot.instruction_epoch);
    }

    #[test]
    fn load_at_epoch_refuses_drift_loudly_never_silently() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: v1\n");
        let e0 = Instructions::load(d.path()).epoch();
        write(d.path(), "AGENTS.md", "always: v2\n");
        let err = Instructions::load_at_epoch(d.path(), e0)
            .err()
            .expect("epoch mismatch");
        assert!(matches!(
            err,
            EnvSnapshotError::EpochMismatch { expected, actual }
                if expected == e0 && actual != e0
        ));
        // An unchanged tree still loads at its epoch.
        let e1 = Instructions::load(d.path()).epoch();
        assert_eq!(
            e1,
            Instructions::load_at_epoch(d.path(), e1).unwrap().epoch()
        );
    }

    #[test]
    fn pinned_reads_refuse_missing_or_tampered_content() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: rule\n");
        let captured = EnvSnapshot::capture(d.path(), "env-a", 1).unwrap();
        // Missing content: loud, typed — never a skip.
        let mut partial = captured.content.clone();
        partial.remove("AGENTS.md");
        let err = Instructions::from_snapshot(&captured.snapshot, &partial)
            .err()
            .expect("missing content refused");
        assert!(matches!(err, EnvSnapshotError::Missing(_)), "{err:?}");
        // Tampered bytes: loud, typed — never a silent substitution.
        let mut tampered = captured.content.clone();
        tampered.insert("AGENTS.md".into(), "always: EVIL\n".into());
        let err = Instructions::from_snapshot(&captured.snapshot, &tampered)
            .err()
            .expect("tampered content refused");
        assert!(matches!(err, EnvSnapshotError::Tampered(_)), "{err:?}");
        // An extra content entry shadows nothing: the snapshot's path set
        // is authoritative.
        let mut extra = captured.content.clone();
        extra.insert("nope.md".into(), "never a rule\n".into());
        let pinned = Instructions::from_snapshot(&captured.snapshot, &extra).unwrap();
        assert_eq!(pinned.active_for("x", &[]).len(), 1);
    }

    #[test]
    fn hostile_snapshot_paths_are_malformed_never_rules() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: ok\n");
        let mut captured = EnvSnapshot::capture(d.path(), "env-a", 1).unwrap();
        assert_eq!(
            kind_for_rel_path(std::path::Path::new("AGENTS.md")),
            Some(RuleSourceKind::AgentsMd)
        );
        assert_eq!(
            kind_for_rel_path(std::path::Path::new(".cursor/rules/x.mdc")),
            Some(RuleSourceKind::CursorRules)
        );
        assert_eq!(kind_for_rel_path(std::path::Path::new("evil/x.md")), None);
        // A captured path set that could never come from discovery is a
        // malformed snapshot.
        captured.snapshot.workspace_paths.insert(
            "evil/x.md".into(),
            EnvFileRecord {
                rules_hash: 1,
                bytes_hash: 1,
            },
        );
        captured.content.insert("evil/x.md".into(), "x".into());
        let err = Instructions::from_snapshot(&captured.snapshot, &captured.content)
            .err()
            .expect("malformed path refused");
        assert!(matches!(err, EnvSnapshotError::Malformed(_)), "{err:?}");
    }

    #[test]
    fn capture_is_bounded_with_typed_oversized_never_truncation() {
        let d = tempfile::tempdir().unwrap();
        // Path cap: MAX_SNAPSHOT_PATHS + 1 rule files.
        for i in 0..(MAX_SNAPSHOT_PATHS + 1) {
            write(
                d.path(),
                &format!(".cursor/rules/f{i:04}.mdc"),
                "# Scope: k{i}\nrule body\n",
            );
        }
        let err = EnvSnapshot::capture(d.path(), "env-a", 1).unwrap_err();
        assert!(matches!(err, EnvSnapshotError::Oversized(_)), "{err:?}");
        assert!(err.to_string().contains("rule files"), "{err}");
        // Total-byte cap: fewer files, but each read saturates at
        // MAX_RULE_BYTES, so 10 x 64 KiB blows the total bound.
        let d2 = tempfile::tempdir().unwrap();
        let body = "b".repeat(MAX_RULE_BYTES);
        for i in 0..10 {
            write(d2.path(), &format!(".cursor/rules/g{i:02}.mdc"), &body);
        }
        let err2 = EnvSnapshot::capture(d2.path(), "env-a", 1).unwrap_err();
        assert!(matches!(err2, EnvSnapshotError::Oversized(_)), "{err2:?}");
        assert!(err2.to_string().contains("total bytes"), "{err2}");
        // Hostile ids and roots refuse loudly.
        assert!(EnvSnapshot::capture(d.path(), "", 1).is_err());
        assert!(EnvSnapshot::capture(d.path(), "a/b", 1).is_err());
        assert!(EnvSnapshot::capture(d.path(), "caf\u{e9}", 1).is_err());
        assert!(EnvSnapshot::capture(&d.path().join("missing"), "env-a", 1).is_err());
    }

    #[test]
    fn snapshot_hashes_are_stable_across_copies_and_serde_shaped() {
        // The snapshot types must be serde-ready (the orchestrator persists
        // them as JSON rows); the compile-time bound is the contract.
        fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}
        assert_serde::<EnvSnapshot>();
        assert_serde::<EnvFileRecord>();
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: durable rules\n");
        let captured = EnvSnapshot::capture(d.path(), "env-a", 42).unwrap();
        // A second capture of the same unchanged tree carries identical
        // hashes — the dedup identity of unchanged envs.
        let again = EnvSnapshot::capture(d.path(), "env-b", 43).unwrap();
        assert_eq!(
            again.snapshot.instruction_epoch,
            captured.snapshot.instruction_epoch
        );
        assert_eq!(
            again.snapshot.workspace_paths,
            captured.snapshot.workspace_paths
        );
        assert_eq!(again.content, captured.content);
        let pinned = Instructions::from_snapshot(&captured.snapshot, &captured.content).unwrap();
        assert!(pinned.active_for("x", &[])[0]
            .content
            .contains("durable rules"));
    }
}
