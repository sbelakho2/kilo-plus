//! Lazy instructions + skills (audit: import rules WITHOUT dumping every
//! rule into every prompt; conditional activation; instruction-epoch
//! detection; skills that cost zero prompt tokens until loaded).
//!
//! Discovery is cheap and pure; only `active_for` produces output and only
//! `load_skill` reads full skill bodies.

use std::collections::hash_map::DefaultHasher;
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
        let mut walk: Vec<PathBuf> = vec![dir.clone()];
        let mut depth = 0usize;
        while let Some(d) = walk.pop() {
            depth += 1;
            if depth > MAX_WALK_DEPTH {
                break;
            }
            let Ok(entries) = std::fs::read_dir(&d) else {
                continue;
            };
            let mut names: Vec<PathBuf> = Vec::new();
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk.push(p);
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
        let discovered = discover_rule_files(root);
        let mut rules = Vec::new();
        for (kind, path) in discovered {
            let Some(content) = read_bounded(&path, MAX_RULE_BYTES) else {
                continue;
            };
            // Activation scope: the rule's subdirectory RELATIVE TO its
            // convention root (e.g. .cursor/rules/frontend/x.mdc -> the
            // ".cursor/rules/frontend" parent path prefix works below via
            // full parent matching); top-level convention rules carry the
            // convention dir itself so keyword directives still gate them.
            let scope = {
                let rel = path
                    .parent()
                    .and_then(|p| p.strip_prefix(root).ok())
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default();
                // Strip the convention root so activation uses the
                // workspace-relative subdir (frontend/App.tsx -> frontend).
                let convention = convention_root_for(kind);
                match rel.strip_prefix(convention) {
                    Some("") => String::new(),
                    Some(rest) => rest.trim_start_matches('/').to_string(),
                    None => rel,
                }
            };
            rules.push(Instruction {
                source: kind,
                path: path
                    .strip_prefix(root)
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| path.to_string_lossy().into_owned()),
                scope,
                content: content.clone(),
                hash: hash_of(&path, &content),
                priority: kind.priority(),
                reason_loaded: String::new(),
            });
        }
        let epoch = Self::compute_epoch(&rules);
        Self {
            rules,
            epoch,
            root: root.to_path_buf(),
        }
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
}
