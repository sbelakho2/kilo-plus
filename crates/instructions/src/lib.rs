//! Lazy instructions + skills (audit: import rules WITHOUT dumping every
//! rule into every prompt; conditional activation; instruction-epoch
//! detection; skills that cost zero prompt tokens until loaded).
//!
//! Discovery is cheap and pure; only `active_for` produces output and only
//! `load_skill` reads full skill bodies.
//!
//! Per-workspace resolution (P0-32): [`InstructionResolver`] turns a durable
//! workspace id into the loaded instruction set of that workspace's root —
//! the root ALWAYS comes from a [`WorkspaceRootProvider`] (the daemon
//! session store), never from the process CWD and never from a static
//! config default. File hashes and instruction epochs (P0-33) are BLAKE3
//! digests, durable across processes and Rust versions — never the
//! unspecified `DefaultHasher`. Authority rule files beyond
//! [`MAX_RULE_BYTES`] (P0-34) are a loud typed error, never a silent
//! truncation; lower-priority optional imports that exceed the cap are
//! skipped whole with a surfaced [`RuleSkip`] entry — no partial text ever
//! reaches a prompt.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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

const HEX: &[u8; 16] = b"0123456789abcdef";

fn hex32(bytes: &[u8; 32]) -> String {
    let mut out = String::with_capacity(64);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn dehex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, pair) in s.as_bytes().as_chunks::<2>().0.iter().enumerate() {
        let hi = HEX.iter().position(|c| *c == pair[0])? as u8;
        let lo = HEX.iter().position(|c| *c == pair[1])? as u8;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

macro_rules! digest_newtype {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name([u8; 32]);

        impl $name {
            #[inline]
            pub fn as_bytes(&self) -> &[u8; 32] {
                &self.0
            }

            /// Little-endian projection of the first 8 digest bytes. Used
            /// ONLY at seams whose durable row shapes predate the 256-bit
            /// types (env-snapshot rows and env-content keys are `u64`
            /// columns today); full-strength equality comparisons always
            /// use the 256-bit value.
            #[inline]
            pub fn as_u64(self) -> u64 {
                u64::from_le_bytes(self.0[..8].try_into().expect("8 bytes"))
            }
        }

        impl From<blake3::Hash> for $name {
            fn from(h: blake3::Hash) -> Self {
                Self(*h.as_bytes())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", hex32(&self.0))
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(&hex32(&self.0))
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let raw = <String as serde::Deserialize>::deserialize(d)?;
                dehex32(&raw)
                    .map(Self)
                    .ok_or_else(|| serde::de::Error::custom("expected 64 lowercase hex chars"))
            }
        }
    };
}

digest_newtype!(InstructionHash, "BLAKE3-256 digest of one (rule path, rule content) pair — the durable file identity of a loaded rule.");
digest_newtype!(InstructionEpoch, "BLAKE3-256 digest over the ordered (path, hash) pairs of a loaded rule tree — the durable instruction epoch.");

/// Typed load failures of the rule loader (P0-34). Authority rule files
/// (top-level AGENTS.md / FAKTOR.md / CLAUDE.md) beyond
/// [`MAX_RULE_BYTES`] — or unreadable — are a LOUD error; the loader never
/// half-reads an authority file into a prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RulesLoadError {
    Oversized(String),
    Unreadable(String),
    EpochMismatch {
        expected: InstructionEpoch,
        actual: InstructionEpoch,
    },
}

impl fmt::Display for RulesLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Oversized(m) | Self::Unreadable(m) => write!(f, "{m}"),
            Self::EpochMismatch { expected, actual } => write!(
                f,
                "live instruction epoch {actual} differs from the required epoch {expected}; the pinned snapshot must be read, never the live env"
            ),
        }
    }
}

impl std::error::Error for RulesLoadError {}

/// A surfaced skip of one rule file that exceeded a bound or could not be
/// read: the file NEVER contributes partial text to the loaded set — it is
/// either loaded whole or absent whole, and its absence is recorded here.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RuleSkip {
    /// Workspace-relative rule path.
    pub path: String,
    /// On-disk size when known (oversized files); `None` for unreadable ones.
    pub bytes: Option<u64>,
    /// Human reason: `oversized: <bytes> bytes > <cap>` or `unreadable: <err>`.
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Instruction {
    pub source: RuleSourceKind,
    pub path: String,
    pub scope: String,
    pub content: String,
    pub hash: InstructionHash,
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

/// Deterministic BLAKE3 digest of (path, content) — the durable file
/// identity (P0-33). Stable across processes and Rust versions.
fn hash_of(path: &Path, content: &str) -> InstructionHash {
    let mut h = blake3::Hasher::new();
    h.update(path.to_string_lossy().as_bytes());
    h.update(b"\0");
    h.update(content.as_bytes());
    InstructionHash::from(h.finalize())
}

/// Content-only digest projection (the `bytes_hash` of env records).
fn content_hash_proj(content: &[u8]) -> u64 {
    blake3::hash(content).as_bytes()[..8]
        .try_into()
        .map(u64::from_le_bytes)
        .expect("8 bytes")
}

/// Bounded read outcome of ONE rule file. Oversized is detected by reading
/// at most `cap + 1` bytes — a hostile multi-GiB file never enters RAM —
/// and is reported with its size, never truncated.
enum RuleFileRead {
    Missing,
    Unreadable(String),
    Oversized(u64),
    Full(String),
}

fn read_rule_file(path: &Path, cap: usize) -> RuleFileRead {
    let file = match std::fs::File::open(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return RuleFileRead::Missing,
        Err(e) => return RuleFileRead::Unreadable(e.to_string()),
        Ok(f) => f,
    };
    let mut buf = Vec::with_capacity(cap + 1);
    // The reported size of an oversized file is its metadata length (the
    // read below stops at cap + 1 bytes; the on-disk size is the honest
    // number a skip/error entry must carry).
    let disk_len = file.metadata().map(|m| m.len()).ok();
    match file.take(cap as u64 + 1).read_to_end(&mut buf) {
        Err(e) => RuleFileRead::Unreadable(e.to_string()),
        Ok(_) if buf.len() > cap => {
            let size = disk_len.map_or(buf.len() as u64, |l| l.max(buf.len() as u64));
            RuleFileRead::Oversized(size)
        }
        Ok(_) => RuleFileRead::Full(String::from_utf8_lossy(&buf).into_owned()),
    }
}

/// Authority rule files: top-level AGENTS.md / FAKTOR.md / CLAUDE.md at the
/// repo root (P0-34). These feed the prompt unconditionally or near it, so
/// an oversized or unreadable one is a loud error — never silently omitted
/// or half-read. Everything else (scoped/directory rules, GEMINI.md,
/// copilot instructions, legacy imports, ...) is a lower-priority import:
/// oversize/unreadable files are skipped WHOLE with a surfaced entry.
fn is_authority_policy_file(root: &Path, path: &Path) -> bool {
    ["AGENTS.md", "FAKTOR.md", "CLAUDE.md"]
        .iter()
        .any(|n| path == root.join(n))
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
#[derive(Debug)]
pub struct Instructions {
    rules: Vec<Instruction>,
    skipped: Vec<RuleSkip>,
    epoch: InstructionEpoch,
    root: PathBuf,
}

impl Instructions {
    /// Load the LIVE rule tree of `root` (P0-34): an authority file
    /// (top-level AGENTS.md / FAKTOR.md / CLAUDE.md) beyond
    /// [`MAX_RULE_BYTES`] — or unreadable — is a typed error, NEVER a
    /// silent truncation or omission. Optional oversized imports are
    /// skipped whole and surfaced via [`Instructions::skipped`].
    pub fn load(root: &Path) -> Result<Self, RulesLoadError> {
        let (rules, skipped) = load_rules(root)?;
        let epoch = Self::compute_epoch(&rules);
        Ok(Self {
            rules,
            skipped,
            epoch,
            root: root.to_path_buf(),
        })
    }

    /// Strict epoch-pinned load: loads the LIVE tree and refuses — with a
    /// typed [`RulesLoadError::EpochMismatch`] — when the live
    /// instruction epoch differs from the pinned `expected_epoch`. There is
    /// never a silent drift: a caller that requires the environment of a
    /// snapshot must read the snapshot (audit 97), never fall back to
    /// whatever the filesystem holds now.
    pub fn load_at_epoch(
        root: &Path,
        expected_epoch: InstructionEpoch,
    ) -> Result<Self, RulesLoadError> {
        let live = Self::load(root)?;
        if live.epoch == expected_epoch {
            Ok(live)
        } else {
            Err(RulesLoadError::EpochMismatch {
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
    /// errors, never a silent skip or a live fallback). The epoch is
    /// recomputed over the re-verified (path, hash) pairs in discovery
    /// order, so it equals the digest a live load of the same tree
    /// computes.
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
        for rel in snap.workspace_paths.keys() {
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
                hash: hash_of(&abs, text),
                priority: kind.priority(),
                reason_loaded: String::new(),
            });
        }
        // Discovery order (kind priority desc, path asc) is the epoch
        // input order of a live load of the same tree.
        rules.sort_by(|a, b| {
            b.source
                .priority()
                .cmp(&a.source.priority())
                .then_with(|| a.path.cmp(&b.path))
        });
        let epoch = Self::compute_epoch(&rules);
        Ok(Self {
            rules,
            skipped: snap.skipped.clone(),
            epoch,
            root,
        })
    }

    /// BLAKE3 over the ordered (path, hash) pairs — durable, deterministic.
    fn compute_epoch(rules: &[Instruction]) -> InstructionEpoch {
        let mut h = blake3::Hasher::new();
        for r in rules {
            h.update(r.path.as_bytes());
            h.update(r.hash.as_bytes());
        }
        InstructionEpoch::from(h.finalize())
    }

    /// True when any rule file changed since load (instruction epoch —
    /// stale rules must never silently govern a long task). A tree that
    /// became unloadable (hostile oversized authority file) is a typed
    /// error — never a silent "unchanged".
    pub fn reload_if_changed(&mut self) -> Result<bool, RulesLoadError> {
        let fresh = Self::load(&self.root)?;
        if fresh.epoch != self.epoch {
            *self = fresh;
            return Ok(true);
        }
        Ok(false)
    }

    pub fn epoch(&self) -> InstructionEpoch {
        self.epoch
    }

    /// Surfaced whole-file skips (P0-34): optional rule files that were
    /// never read (oversized or unreadable) are listed here with their size
    /// and reason — they are never half-loaded.
    pub fn skipped(&self) -> &[RuleSkip] {
        &self.skipped
    }

    /// True when the tree holds at least one rule file. An empty tree's
    /// epoch is a vacuous stamp; epoch-pinning callers may treat an empty
    /// tree like "no rules loaded".
    pub fn has_rules(&self) -> bool {
        !self.rules.is_empty()
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
///
/// `EpochMismatch` keeps its historic `u64` payloads: orchestrator code
/// (outside this crate) pattern-matches this enum exhaustively and formats
/// the pair. New epoch-pinned loaders use
/// [`RulesLoadError::EpochMismatch`] with the 256-bit [`InstructionEpoch`].
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

/// One captured rule file. `rules_hash` is the 64-bit little-endian
/// projection of the BLAKE3 digest of (path, content) — the SAME digest
/// [`Instruction::hash`] carries — and `bytes_hash` the projection of the
/// content-only BLAKE3 digest. Both projections exist because the durable
/// env rows and env-content keys this crate feeds (orchestrator side) are
/// `u64`-shaped today; the projections are stable across processes and Rust
/// versions (P0-33).
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
    /// 64-bit projection of the BLAKE3 instruction-epoch digest over the
    /// ordered (path, rules_hash) pairs — equal to
    /// `Instructions::load(root).epoch().as_u64()` over an unchanged tree.
    pub instruction_epoch: u64,
    /// The root directory the snapshot was captured from.
    pub root: PathBuf,
    /// Workspace-relative rule path -> recorded hashes, sorted by path.
    pub workspace_paths: BTreeMap<String, EnvFileRecord>,
    /// Surfaced whole-file skips during capture (optional oversized /
    /// unreadable rule files; never partial text).
    #[serde(default)]
    pub skipped: Vec<RuleSkip>,
    pub taken_ms: i64,
}

impl EnvSnapshot {
    /// Capture the rule environment of `root` exactly as
    /// [`Instructions::load`] would see it (same discovery, same per-file
    /// read bound, same ordering — so the snapshot epoch matches a live
    /// load of an unchanged tree). Bounded: at most [`MAX_SNAPSHOT_PATHS`]
    /// rule files and at most [`MAX_SNAPSHOT_TOTAL_BYTES`] total captured
    /// bytes; beyond either the capture fails with a typed Oversized error
    /// and returns NOTHING (never a silently truncated snapshot). An
    /// authority rule file (top-level AGENTS.md / FAKTOR.md / CLAUDE.md)
    /// beyond [`MAX_RULE_BYTES`] also fails loudly; an oversized optional
    /// import is skipped whole and surfaced in `snapshot.skipped`. Returns
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
        let mut skipped = Vec::new();
        let mut total_bytes = 0usize;
        let mut epoch = blake3::Hasher::new();
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
            let authority = is_authority_policy_file(root, &path);
            match read_rule_file(&path, MAX_RULE_BYTES) {
                RuleFileRead::Missing => {}
                RuleFileRead::Unreadable(err) if authority => {
                    return Err(EnvSnapshotError::Malformed(format!(
                        "authority rule file {} is unreadable ({err}); refusing a capture that silently omits it",
                        path.display()
                    )));
                }
                RuleFileRead::Oversized(len) if authority => {
                    return Err(EnvSnapshotError::Oversized(format!(
                        "authority rule file {} is {len} bytes — over the {MAX_RULE_BYTES} rule bound; refusing a capture that half-reads it",
                        path.display()
                    )));
                }
                RuleFileRead::Unreadable(err) => {
                    skipped.push(RuleSkip {
                        path: rel_str.clone(),
                        bytes: None,
                        reason: format!("unreadable: {err}"),
                    });
                }
                RuleFileRead::Oversized(len) => {
                    skipped.push(RuleSkip {
                        path: rel_str.clone(),
                        bytes: Some(len),
                        reason: format!("oversized: {len} bytes > {MAX_RULE_BYTES}"),
                    });
                }
                RuleFileRead::Full(text) => {
                    if total_bytes.saturating_add(text.len()) > MAX_SNAPSHOT_TOTAL_BYTES {
                        return Err(EnvSnapshotError::Oversized(format!(
                            "rule environment of {:?} exceeds {MAX_SNAPSHOT_TOTAL_BYTES} total bytes",
                            root
                        )));
                    }
                    total_bytes += text.len();
                    let rules_hash = hash_of(&path, &text);
                    // instruction_epoch must digest the same (rel path,
                    // 32-byte rules hash) pairs in the same order as
                    // compute_epoch over loaded rules.
                    epoch.update(rel_str.as_bytes());
                    epoch.update(rules_hash.as_bytes());
                    workspace_paths.insert(
                        rel_str.clone(),
                        EnvFileRecord {
                            rules_hash: rules_hash.as_u64(),
                            bytes_hash: content_hash_proj(text.as_bytes()),
                        },
                    );
                    content.insert(rel_str, text);
                }
            }
        }
        let digest = epoch.finalize();
        Ok(CapturedEnv {
            snapshot: EnvSnapshot {
                snapshot_id: snapshot_id.to_string(),
                instruction_epoch: InstructionEpoch::from(digest).as_u64(),
                root: root.to_path_buf(),
                workspace_paths,
                skipped,
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
        if content_hash_proj(text.as_bytes()) != rec.bytes_hash {
            return Err(EnvSnapshotError::Tampered(format!(
                "content bytes of {rel:?} do not match the recorded bytes hash"
            )));
        }
        if hash_of(&snap.root.join(rel), text).as_u64() != rec.rules_hash {
            return Err(EnvSnapshotError::Tampered(format!(
                "content of {rel:?} does not match the recorded (path, content) rules hash"
            )));
        }
    }
    Ok(())
}

/// Load every rule of `root` in discovery order (the epoch input order).
/// Authority files that are oversized or unreadable FAIL the whole load
/// (P0-34); optional imports that are oversized or unreadable are skipped
/// whole with a surfaced [`RuleSkip`] — partial text never reaches rules.
fn load_rules(root: &Path) -> Result<(Vec<Instruction>, Vec<RuleSkip>), RulesLoadError> {
    let mut rules = Vec::new();
    let mut skipped = Vec::new();
    for (kind, path) in discover_rule_files(root) {
        let authority = is_authority_policy_file(root, &path);
        match read_rule_file(&path, MAX_RULE_BYTES) {
            RuleFileRead::Missing => {}
            RuleFileRead::Unreadable(err) if authority => {
                return Err(RulesLoadError::Unreadable(format!(
                    "authority rule file {} could not be read ({err}); refusing to load rules that silently omit it",
                    path.display()
                )));
            }
            RuleFileRead::Oversized(len) if authority => {
                return Err(RulesLoadError::Oversized(format!(
                    "authority rule file {} is {len} bytes — over the {MAX_RULE_BYTES} rule bound; refusing to half-load it",
                    path.display()
                )));
            }
            RuleFileRead::Unreadable(err) => {
                skipped.push(RuleSkip {
                    path: rel_of(root, &path),
                    bytes: None,
                    reason: format!("unreadable: {err}"),
                });
            }
            RuleFileRead::Oversized(len) => {
                skipped.push(RuleSkip {
                    path: rel_of(root, &path),
                    bytes: Some(len),
                    reason: format!("oversized: {len} bytes > {MAX_RULE_BYTES}"),
                });
            }
            RuleFileRead::Full(content) => {
                rules.push(Instruction {
                    source: kind,
                    path: rel_of(root, &path),
                    scope: scope_of(root, kind, &path),
                    content: content.clone(),
                    hash: hash_of(&path, &content),
                    priority: kind.priority(),
                    reason_loaded: String::new(),
                });
            }
        }
    }
    Ok((rules, skipped))
}

fn rel_of(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| path.to_string_lossy().into_owned())
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

// ------------------------------------------------- per-workspace resolver (P0-32)

/// The durable workspace-root source of the resolver. Implemented in the
/// daemon (cli) over `SessionManager` — and in tests over their own session
/// managers — NEVER over the process CWD or a static config root.
///
/// The trait lives HERE (not in faktor-core, which this crate cannot depend
/// on, and not in faktor-session, which must not depend on this crate):
/// callers implement it for their own local type, so there is no dependency
/// cycle.
pub trait WorkspaceRootProvider: Send + Sync {
    /// The durable root directory of `workspace_id` (raw `WorkspaceId` from
    /// the daemon workspace table). `None` for an unknown workspace or a
    /// workspace without a root — the caller resolves that to
    /// [`LoadedInstructions::Empty`], never an error.
    fn workspace_root(&self, workspace_id: u64) -> Option<PathBuf>;
}

/// Default bound on cached loaded instruction sets per resolver.
pub const DEFAULT_RESOLVER_CACHE_ENTRIES: usize = 32;

/// Provider that never resolves a root: every resolution is Empty. Used by
/// code paths (and tests) that must not serve repository rules — the exact
/// behavioral equivalent of the historic optional loader being absent.
struct NoRoots;

impl WorkspaceRootProvider for NoRoots {
    fn workspace_root(&self, _workspace_id: u64) -> Option<PathBuf> {
        None
    }
}

/// A resolver over [`NoRoots`]: every `resolve` returns
/// [`LoadedInstructions::Empty`]. Drop-in replacement for the historic
/// "no instructions loader wired" state.
pub fn no_roots_resolver() -> Arc<InstructionResolver> {
    Arc::new(InstructionResolver::new(
        Arc::new(NoRoots),
        DEFAULT_RESOLVER_CACHE_ENTRIES,
    ))
}

/// LRU over loaded rule trees keyed by (root, epoch). A pinned-epoch
/// request is served from this cache even after the live tree changed; once
/// evicted it refuses loudly (the durable snapshot is the only other source
/// of old trees — this crate has none). Never unbounded.
struct RuleCache {
    entries: HashMap<(PathBuf, InstructionEpoch), Arc<Instructions>>,
    order: VecDeque<(PathBuf, InstructionEpoch)>,
    cap: usize,
}

impl RuleCache {
    fn new(cap: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            cap: cap.max(1),
        }
    }

    fn get(&mut self, key: &(PathBuf, InstructionEpoch)) -> Option<Arc<Instructions>> {
        if !self.entries.contains_key(key) {
            return None;
        }
        self.order.retain(|k| k != key);
        self.order.push_back(key.clone());
        self.entries.get(key).cloned()
    }

    fn put(&mut self, key: (PathBuf, InstructionEpoch), value: Arc<Instructions>) {
        if self.entries.contains_key(&key) {
            return;
        }
        if self.entries.len() >= self.cap {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, value);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Per-workspace instruction resolver (P0-32): resolves a workspace id to
/// the loaded instruction set of its DURABLE root, with a bounded LRU cache
/// over (root, epoch) entries.
pub struct InstructionResolver {
    provider: Arc<dyn WorkspaceRootProvider>,
    cache: Mutex<RuleCache>,
}

impl InstructionResolver {
    /// `cap` bounds the cache; `0` degenerates to 1 (never unbounded).
    pub fn new(provider: Arc<dyn WorkspaceRootProvider>, cap: usize) -> Self {
        Self {
            provider,
            cache: Mutex::new(RuleCache::new(cap)),
        }
    }

    /// Resolve `workspace_id` to its loaded instruction set.
    ///
    /// - No durable root (unknown workspace / rootless session):
    ///   [`LoadedInstructions::Empty`] — documented, never an error.
    /// - A hostile tree (authority rule file oversized/unreadable): typed
    ///   error, never a silent truncation.
    /// - `pinned_epoch: Some(e)`: the request wants the OLD tree of epoch
    ///   `e`. Served from the cache when still present; otherwise the live
    ///   tree is loaded and served ONLY when its epoch still equals `e`;
    ///   any other outcome is a loud [`RulesLoadError::EpochMismatch`] —
    ///   never silently serving today's rules as yesterday's.
    pub fn resolve(
        &self,
        workspace_id: u64,
        pinned_epoch: Option<InstructionEpoch>,
    ) -> Result<LoadedInstructions, RulesLoadError> {
        let Some(root) = self.provider.workspace_root(workspace_id) else {
            return Ok(LoadedInstructions::Empty);
        };
        match pinned_epoch {
            Some(epoch) => {
                let key = (root.clone(), epoch);
                if let Some(cached) = self.cache_get(&key) {
                    return Ok(LoadedInstructions::Loaded(cached));
                }
                let live = Arc::new(Instructions::load(&root)?);
                if live.epoch() == epoch {
                    self.cache_put(key, live.clone());
                    Ok(LoadedInstructions::Loaded(live))
                } else {
                    Err(RulesLoadError::EpochMismatch {
                        expected: epoch,
                        actual: live.epoch(),
                    })
                }
            }
            None => {
                let loaded = Arc::new(Instructions::load(&root)?);
                let key = (root, loaded.epoch());
                self.cache_put(key, loaded.clone());
                Ok(LoadedInstructions::Loaded(loaded))
            }
        }
    }

    fn cache_get(&self, key: &(PathBuf, InstructionEpoch)) -> Option<Arc<Instructions>> {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        cache.get(key)
    }

    fn cache_put(&self, key: (PathBuf, InstructionEpoch), value: Arc<Instructions>) {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        cache.put(key, value);
    }

    /// Test/observability probe: current number of cached rule trees
    /// (always <= the resolver cap).
    pub fn cache_len(&self) -> usize {
        let cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        cache.len()
    }

    /// The resolver cap (cache bound).
    pub fn cache_cap(&self) -> usize {
        let cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        cache.cap
    }
}

/// Result of one [`InstructionResolver::resolve`]: either the loaded rule
/// tree of the workspace's durable root, or [`LoadedInstructions::Empty`]
/// for sessions whose workspace carries no durable root.
#[derive(Debug, Clone)]
pub enum LoadedInstructions {
    Empty,
    Loaded(Arc<Instructions>),
}

impl LoadedInstructions {
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    pub fn instructions(&self) -> Option<&Instructions> {
        match self {
            Self::Empty => None,
            Self::Loaded(ins) => Some(ins.as_ref()),
        }
    }

    pub fn epoch(&self) -> Option<InstructionEpoch> {
        self.instructions().map(Instructions::epoch)
    }

    pub fn active_for(&self, prompt: &str, touched: &[String]) -> Vec<Instruction> {
        self.instructions()
            .map(|ins| ins.active_for(prompt, touched))
            .unwrap_or_default()
    }

    /// True when the loaded tree holds at least one rule file (`Empty` is
    /// false).
    pub fn has_rules(&self) -> bool {
        self.instructions()
            .map(Instructions::has_rules)
            .unwrap_or(false)
    }

    /// Surfaced whole-file skips of the loaded set (empty for
    /// [`LoadedInstructions::Empty`]).
    pub fn skipped(&self) -> &[RuleSkip] {
        self.instructions()
            .map(Instructions::skipped)
            .unwrap_or(&[])
    }
}

// ---------------------------------------------------------------- skills

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

/// Legacy bounded read for NON-rule payloads (skill metadata summaries and
/// on-demand skill bodies): skills are never injected into a prompt
/// wholesale, and their read is strictly best-effort.
fn read_bounded(path: &Path, cap: usize) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(String::from_utf8_lossy(&bytes[..bytes.len().min(cap)]).into_owned())
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
        let ins = Instructions::load(d.path()).unwrap();
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
        let ins = Instructions::load(d.path()).unwrap();
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
        let ins = Instructions::load(d.path()).unwrap();
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
        let ins = Instructions::load(d.path()).unwrap();
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
        let mut ins = Instructions::load(d.path()).unwrap();
        let e0 = ins.epoch();
        assert!(!ins.reload_if_changed().unwrap());
        write(d.path(), "AGENTS.md", "v2 rules (changed mid-task)\n");
        assert!(
            ins.reload_if_changed().unwrap(),
            "stale epoch must be detected"
        );
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
        let ins = Instructions::load(d.path()).unwrap();
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
        let ins = Instructions::load(d.path()).unwrap();
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
        let ins = Instructions::load(d.path()).unwrap();
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

    // ------------------------------------------------- hashes + epochs (P0-33)

    #[test]
    fn hashes_and_epochs_are_blake3_durable_and_content_keyed() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: durable rules\n");
        write(
            d.path(),
            ".cursor/rules/frontend/ux.mdc",
            "# Scope: ui\nfrontend style\n",
        );
        // Two loads of identical content -> identical hashes AND epochs.
        let a = Instructions::load(d.path()).unwrap();
        let b = Instructions::load(d.path()).unwrap();
        assert_eq!(a.epoch(), b.epoch());
        assert_eq!(a.rules.len(), 2);
        for (x, y) in a.rules.iter().zip(&b.rules) {
            assert_eq!(x.hash, y.hash, "identical content -> identical hash");
            assert_eq!(x.content, y.content);
        }
        // Different content -> different hash and epoch.
        write(d.path(), "AGENTS.md", "always: changed rules\n");
        let c = Instructions::load(d.path()).unwrap();
        assert_ne!(c.epoch(), a.epoch());
        assert_ne!(
            c.rules.iter().find(|r| r.path == "AGENTS.md").unwrap().hash,
            a.rules.iter().find(|r| r.path == "AGENTS.md").unwrap().hash
        );
        // Sequential loads after a rewrite agree (process-independent
        // algorithm: the digest is a pure function of bytes).
        let d2 = Instructions::load(d.path()).unwrap();
        assert_eq!(d2.epoch(), c.epoch());
    }

    #[test]
    fn digest_newtypes_serde_roundtrip_as_hex() {
        // Compile-time serde bounds (no JSON crate lives in this crate;
        // the serde glue is the hex codec below).
        fn assert_serde<T: serde::Serialize + serde::de::DeserializeOwned>() {}
        assert_serde::<InstructionHash>();
        assert_serde::<InstructionEpoch>();
        assert_serde::<RuleSkip>();
        let h = hash_of(Path::new("AGENTS.md"), "rules\n");
        // hex32/dehex32 are exact inverses; the wire format is lowercase
        // hex of the 32 digest bytes.
        let enc = hex32(h.as_bytes());
        assert_eq!(enc.len(), 64);
        assert_eq!(InstructionHash(dehex32(&enc).unwrap()), h);
        assert_eq!(h.to_string(), enc);
        assert!(dehex32(&enc[..63]).is_none(), "short hex rejected");
        assert!(dehex32(&"z".repeat(64)).is_none(), "non-hex rejected");
        assert!(dehex32(&enc.to_uppercase()).is_none(), "uppercase rejected");
        // Projection is deterministic and lossy-by-design (seam-only).
        assert_eq!(
            h.as_u64(),
            u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
        );
        // The empty tree's epoch is deterministic: BLAKE3 of nothing.
        let empty = Instructions::load(Path::new("/nonexistent-rule-root-xyz"))
            .unwrap()
            .epoch();
        assert_eq!(empty, InstructionEpoch::from(blake3::hash(b"")));
    }

    // ------------------------------------------- oversized authority rules (P0-34)

    #[test]
    fn oversized_authority_agents_md_fails_loading_loudly() {
        // (a) A 100 KiB AGENTS.md whose critical rule starts at byte 70 KiB:
        // the load MUST fail loudly (typed Oversized) instead of silently
        // omitting the rule.
        let d = tempfile::tempdir().unwrap();
        let mut body = vec![b'x'; 100 * 1024];
        body[70 * 1024..70 * 1024 + 22].copy_from_slice(b"CRITICAL RULE AT 70KIB");
        let text = String::from_utf8_lossy(&body);
        write(d.path(), "AGENTS.md", &text);
        let err = Instructions::load(d.path()).unwrap_err();
        assert!(
            matches!(err, RulesLoadError::Oversized(_)),
            "authority oversize must be a typed Oversized error: {err:?}"
        );
        assert!(err.to_string().contains("AGENTS.md"), "{err}");
        assert!(err.to_string().contains("rule bound"), "{err}");
        // The snapshot capture refuses the same tree with its typed error.
        let err2 = EnvSnapshot::capture(d.path(), "env-a", 1).unwrap_err();
        assert!(matches!(err2, EnvSnapshotError::Oversized(_)), "{err2:?}");
        assert!(err2.to_string().contains("AGENTS.md"), "{err2}");
        // FAKTOR.md and CLAUDE.md are authority too.
        for name in ["FAKTOR.md", "CLAUDE.md"] {
            let d2 = tempfile::tempdir().unwrap();
            write(d2.path(), name, &"y".repeat(MAX_RULE_BYTES + 1));
            let err3 = Instructions::load(d2.path()).unwrap_err();
            assert!(
                matches!(err3, RulesLoadError::Oversized(_)),
                "{name}: {err3:?}"
            );
            assert!(err3.to_string().contains(name), "{err3}");
        }
    }

    #[test]
    fn oversized_optional_import_is_surfaced_skip_never_partial() {
        // (b) An optional imported file over the cap: load SUCCEEDS, the
        // set records a surfaced-skip entry, and no partial text reaches
        // the rules.
        let d = tempfile::tempdir().unwrap();
        let mut body = vec![b'z'; MAX_RULE_BYTES + 8192];
        body[0..36].copy_from_slice(b"# Scope: hostile\nTAIL MARKER AT 70K\n");
        body[MAX_RULE_BYTES..MAX_RULE_BYTES + 23].copy_from_slice(b"SECRET RULE BEYOND CAP\n");
        let text = String::from_utf8_lossy(&body);
        write(d.path(), ".cursor/rules/huge.mdc", &text);
        write(d.path(), "AGENTS.md", "always: sane rules\n");
        let ins = Instructions::load(d.path()).unwrap();
        // The set records the surfaced skip with the file and its size.
        let skip = ins
            .skipped()
            .iter()
            .find(|s| s.path == ".cursor/rules/huge.mdc")
            .expect("surfaced skip entry must exist");
        assert_eq!(skip.bytes, Some(text.len() as u64));
        assert!(skip.reason.contains("oversized"), "{skip:?}");
        // No partial text reaches the rules: no rule from that file at all,
        // and nothing beyond the cap anywhere.
        assert!(
            !ins.rules.iter().any(|r| r.path.contains("huge")),
            "the oversized optional file must not be half-loaded"
        );
        assert!(
            !ins.active_for("hostile", &[])
                .iter()
                .any(|i| i.content.contains("SECRET RULE BEYOND CAP")),
            "no partial text may reach activation"
        );
        // The snapshot capture behaves identically and surfaces the skip on
        // the durable snapshot.
        let captured = EnvSnapshot::capture(d.path(), "env-a", 1).unwrap();
        let cskip = captured
            .snapshot
            .skipped
            .iter()
            .find(|s| s.path == ".cursor/rules/huge.mdc")
            .expect("capture must surface the same skip");
        assert_eq!(cskip.bytes, skip.bytes);
        assert!(!captured
            .snapshot
            .workspace_paths
            .contains_key(".cursor/rules/huge.mdc"));
        assert!(!captured.content.contains_key(".cursor/rules/huge.mdc"));
        // A pinned read from the snapshot surfaces the skip too.
        let pinned = Instructions::from_snapshot(&captured.snapshot, &captured.content).unwrap();
        assert!(pinned
            .skipped()
            .iter()
            .any(|s| s.path == ".cursor/rules/huge.mdc"));
        assert!(!pinned
            .active_for("hostile", &[])
            .iter()
            .any(|i| i.content.contains("SECRET RULE BEYOND CAP")));
    }

    #[test]
    fn rule_at_exact_cap_loads_fully() {
        // (c) A root AGENTS.md exactly at the cap loads FULLY — the marker
        // written at the very last bytes is present; the cap is not off by
        // one in either direction.
        let d = tempfile::tempdir().unwrap();
        let mut body = vec![b'r'; MAX_RULE_BYTES];
        body[MAX_RULE_BYTES - 23..].copy_from_slice(b"FINAL BYTES LOAD WHOLE\n");
        let text = String::from_utf8_lossy(&body);
        write(d.path(), "AGENTS.md", &text);
        let ins = Instructions::load(d.path()).unwrap();
        let agent = ins
            .rules
            .iter()
            .find(|r| r.path == "AGENTS.md")
            .expect("at-cap AGENTS.md loads");
        assert_eq!(
            agent.content.len(),
            MAX_RULE_BYTES,
            "loaded fully, not truncated"
        );
        assert!(agent.content.ends_with("FINAL BYTES LOAD WHOLE\n"));
        assert!(ins.skipped().is_empty());
    }

    // --------------------------------------- env snapshots (audit 97)

    #[test]
    fn snapshot_epoch_equals_live_load_epoch_of_the_same_tree() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: pinned rules v1\n");
        write(
            d.path(),
            ".cursor/rules/frontend/ux.mdc",
            "# Scope: ui\nfrontend style\n",
        );
        let live = Instructions::load(d.path()).unwrap();
        let captured = EnvSnapshot::capture(d.path(), "env-a", 7).unwrap();
        assert_eq!(captured.snapshot.instruction_epoch, live.epoch().as_u64());
        assert_eq!(captured.snapshot.workspace_paths.len(), 2);
        assert!(captured.content["AGENTS.md"].contains("v1"));
        assert!(captured.snapshot.skipped.is_empty());
    }

    #[test]
    fn pinned_reads_serve_spawn_time_rules_after_the_parent_changed_them() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: rule V1\n");
        let captured = EnvSnapshot::capture(d.path(), "env-a", 1).unwrap();
        // The parent's AGENTS.md changes AFTER the snapshot was taken.
        write(d.path(), "AGENTS.md", "always: rule V2\n");
        let pinned = Instructions::from_snapshot(&captured.snapshot, &captured.content).unwrap();
        assert_eq!(pinned.epoch().as_u64(), captured.snapshot.instruction_epoch);
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
        let live = Instructions::load(d.path()).unwrap();
        assert!(live.active_for("x", &[])[0].content.contains("rule V2"));
        assert_ne!(live.epoch().as_u64(), captured.snapshot.instruction_epoch);
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
        assert_eq!(pa.epoch().as_u64(), a.snapshot.instruction_epoch);
        assert_eq!(pb.epoch().as_u64(), b.snapshot.instruction_epoch);
    }

    #[test]
    fn load_at_epoch_refuses_drift_loudly_never_silently() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: v1\n");
        let e0 = Instructions::load(d.path()).unwrap().epoch();
        write(d.path(), "AGENTS.md", "always: v2\n");
        let err = Instructions::load_at_epoch(d.path(), e0).expect_err("epoch mismatch");
        assert!(matches!(
            err,
            RulesLoadError::EpochMismatch { expected, actual }
                if expected == e0 && actual != e0
        ));
        // An unchanged tree still loads at its epoch.
        let e1 = Instructions::load(d.path()).unwrap().epoch();
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
            .expect_err("missing content refused");
        assert!(matches!(err, EnvSnapshotError::Missing(_)), "{err:?}");
        // Tampered bytes: loud, typed — never a silent substitution.
        let mut tampered = captured.content.clone();
        tampered.insert("AGENTS.md".into(), "always: EVIL\n".into());
        let err = Instructions::from_snapshot(&captured.snapshot, &tampered)
            .expect_err("tampered content refused");
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
            .expect_err("malformed path refused");
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
        // Total-byte cap: fewer files, each read at the per-file bound, so
        // 10 x 64 KiB blows the total bound.
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

    // ------------------------------------------------- per-workspace resolver (P0-32)

    /// Fake durable-root provider: id -> root, mirroring the daemon
    /// workspace table (no CWD, no config defaults anywhere).
    struct MapRoots(HashMap<u64, PathBuf>);

    impl WorkspaceRootProvider for MapRoots {
        fn workspace_root(&self, workspace_id: u64) -> Option<PathBuf> {
            self.0.get(&workspace_id).cloned()
        }
    }

    fn resolver_with(roots: HashMap<u64, PathBuf>, cap: usize) -> InstructionResolver {
        InstructionResolver::new(Arc::new(MapRoots(roots)), cap)
    }

    #[test]
    fn resolver_uses_durable_roots_and_returns_empty_for_rootless_sessions() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: durable-root rules\n");
        let resolver = resolver_with(HashMap::from([(7, d.path().to_path_buf())]), 8);
        // Unknown / rootless workspace ids: Empty — documented, never an
        // error and never the process CWD.
        assert!(resolver.resolve(1, None).unwrap().is_empty());
        assert!(resolver.resolve(999, None).unwrap().is_empty());
        // The durable root is served with its content.
        let loaded = resolver.resolve(7, None).unwrap();
        let active = loaded.active_for("anything", &[]);
        assert!(active
            .iter()
            .any(|i| i.content.contains("durable-root rules")));
        assert!(loaded.epoch().is_some());
        // A hostile tree at the durable root is a typed error.
        let hostile = tempfile::tempdir().unwrap();
        write(hostile.path(), "AGENTS.md", &"h".repeat(MAX_RULE_BYTES + 1));
        let r2 = resolver_with(HashMap::from([(8, hostile.path().to_path_buf())]), 8);
        assert!(matches!(
            r2.resolve(8, None),
            Err(RulesLoadError::Oversized(_))
        ));
    }

    #[test]
    fn resolver_pinned_epoch_serves_the_old_tree_after_a_rewrite() {
        // (d) Resolver cache semantics: after AGENTS.md is replaced, a
        // re-resolve returns the NEW content with a NEW epoch, while a
        // request pinned to the OLD epoch still sees the OLD content (the
        // tree the old env snapshot was taken from).
        let d = tempfile::tempdir().unwrap();
        let roots = HashMap::from([(1, d.path().to_path_buf())]);
        let resolver = resolver_with(roots, 8);
        write(d.path(), "AGENTS.md", "always: rule V1\n");
        let v1 = resolver.resolve(1, None).unwrap();
        let e1 = v1.epoch().unwrap();
        assert!(v1.active_for("x", &[])[0].content.contains("V1"));
        write(d.path(), "AGENTS.md", "always: rule V2\n");
        let v2 = resolver.resolve(1, None).unwrap();
        let e2 = v2.epoch().unwrap();
        assert_ne!(e1, e2, "a rewrite must move the durable epoch");
        assert!(v2.active_for("x", &[])[0].content.contains("V2"));
        // Pinned to the OLD epoch: the resolver serves the cached OLD tree,
        // exactly the spawn-time content an env snapshot of epoch e1 holds.
        let pinned_old = resolver.resolve(1, Some(e1)).unwrap();
        assert_eq!(pinned_old.epoch(), Some(e1));
        assert!(
            pinned_old.active_for("x", &[])[0].content.contains("V1"),
            "an old env snapshot must still see the old epoch content"
        );
        // A fresh load still matches epoch e2 (no cross-contamination).
        assert_eq!(resolver.resolve(1, Some(e2)).unwrap().epoch(), Some(e2));
    }

    #[test]
    fn resolver_refuses_pinned_epochs_it_cannot_serve_after_eviction() {
        // Once the old tree is evicted from the bounded cache, a pinned
        // request for the evicted epoch cannot reconstruct yesterday's
        // content from the live filesystem — it refuses loudly instead of
        // silently serving today's rules as yesterday's.
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "AGENTS.md", "always: rule V1\n");
        // cap 1: resolving the rewritten tree evicts the (root, e1) entry.
        let resolver = resolver_with(HashMap::from([(1, d.path().to_path_buf())]), 1);
        let e1 = resolver.resolve(1, None).unwrap().epoch().unwrap();
        write(d.path(), "AGENTS.md", "always: rule V2\n");
        assert!(resolver.resolve(1, None).is_ok(), "new epoch loads");
        assert_eq!(resolver.cache_len(), 1, "only the newest epoch is cached");
        let err = resolver.resolve(1, Some(e1)).unwrap_err();
        assert!(
            matches!(
                err,
                RulesLoadError::EpochMismatch { expected, actual }
                    if expected == e1 && actual != e1
            ),
            "{err:?}"
        );
    }

    #[test]
    fn resolver_cache_stays_bounded_under_many_roots() {
        // (e) Cache boundedness: 500 distinct roots must keep the cache
        // under its cap, evicting LRU entries (oldest first).
        let base = tempfile::tempdir().unwrap();
        let mut roots = HashMap::new();
        for i in 0..500u64 {
            let sub = base.path().join(format!("root-{i}"));
            std::fs::create_dir_all(&sub).unwrap();
            std::fs::write(sub.join("AGENTS.md"), format!("always: rules of {i}\n")).unwrap();
            roots.insert(i, sub);
        }
        let resolver = resolver_with(roots, 16);
        // Resolve every root; afterwards the cache must still be under its
        // cap, and the resolver never grew without bound.
        for i in 0..500u64 {
            let loaded = resolver.resolve(i, None).unwrap();
            assert!(loaded.active_for("x", &[])[0]
                .content
                .contains(&format!("rules of {i}")));
            assert!(
                resolver.cache_len() <= resolver.cache_cap(),
                "never unbounded"
            );
        }
        assert_eq!(resolver.cache_len(), 16, "full LRU under cap");
        // The most recent roots are cached (LRU): a re-resolve of an old
        // evicted root re-loads from disk with identical content (idempotent
        // resolution — content equality, not cache presence).
        let again = resolver.resolve(0, None).unwrap();
        assert!(again.active_for("x", &[])[0].content.contains("rules of 0"));
        assert!(resolver.cache_len() <= resolver.cache_cap());
    }
}
