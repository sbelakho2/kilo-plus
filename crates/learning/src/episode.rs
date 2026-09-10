//! Failure episodes: the ONLY input the miner accepts (audit items 65-67).
//!
//! An episode is typed around an *observed* failure and a *verified*
//! recovery. `eventual_verified_success` is an `Option<VerificationRecordId>`
//! — a durable verification record id — so there is deliberately no API
//! shape that accepts "the model said fixed" as a success signal.
//!
//! Normalization and boundedness are structural:
//!
//! - every descriptor string is whitespace-normalized (and workspace paths
//!   get `/` separators, no `./` or duplicate separators) before hashing;
//! - every field and every list is explicitly bounded, and constructors
//!   reject oversized input instead of truncating it silently;
//! - fingerprints are BLAKE3 over length-prefixed, domain-separated parts,
//!   so field-boundary smuggling cannot make two different descriptors share
//!   a digest;
//! - the project scope (workspace + project key) is part of the environment
//!   identity, so identical symbol names in another project can never match
//!   a pattern mined here.

use std::fmt;

use faktor_core::{id::VerificationRecordId, FileHash, WorkspaceId};
use faktor_evidence::types::EvidenceId;
use serde::{Deserialize, Serialize};

use crate::{learning_id, LearningError};

learning_id!(
    /// Identifies one recorded failure episode. Serialized as a plain u64;
    /// zero is never a valid episode id.
    EpisodeId
);

/// Maximum UTF-8 bytes of one normalized descriptor field.
pub const MAX_FIELD_BYTES: usize = 1024;
/// Maximum UTF-8 bytes of a failure message.
pub const MAX_MESSAGE_BYTES: usize = 2048;
/// Maximum UTF-8 bytes of a project key.
pub const MAX_PROJECT_KEY_BYTES: usize = 256;
/// Maximum UTF-8 bytes of one environment component (platform/toolchain).
pub const MAX_ENVIRONMENT_BYTES: usize = 256;
/// Maximum actions in one recovery chain.
pub const MAX_RECOVERY_ACTIONS: usize = 32;
/// Maximum changed assumptions carried by one episode.
pub const MAX_ASSUMPTIONS: usize = 64;
/// Maximum evidence ids carried by one episode (and by one learning).
pub const MAX_EVIDENCE: usize = 256;
/// Domain separation prefix for every fingerprint and key in this crate.
pub const FINGERPRINT_DOMAIN_V1: &[u8] = b"faktor-learning/fingerprint/v1";

/// Collapse every whitespace run to one space and trim both ends. Never
/// lowercases: symbol and path case is preserved so case-sensitive targets
/// stay distinct.
pub(crate) fn normalize_ws(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut pending_space = false;
    for c in raw.chars() {
        if c.is_whitespace() {
            pending_space = !out.is_empty();
        } else {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(c);
        }
    }
    out
}

/// Normalize a workspace-relative target: whitespace-collapsed, `\` becomes
/// `/`, `.` segments and duplicate separators dropped. Absolute paths keep
/// their leading `/` (documented as caller responsibility to pass
/// workspace-relative targets).
pub(crate) fn normalize_path(raw: &str) -> String {
    let flattened = normalize_ws(raw).replace('\\', "/");
    let absolute = flattened.starts_with('/');
    let joined = flattened
        .split('/')
        .filter(|segment| !segment.is_empty() && *segment != ".")
        .collect::<Vec<_>>()
        .join("/");
    if absolute && !joined.is_empty() {
        format!("/{joined}")
    } else {
        joined
    }
}

/// Reject `text` above `max` bytes; never truncate silently.
pub(crate) fn bounded(text: String, max: usize, what: &str) -> Result<String, LearningError> {
    if text.len() > max {
        return Err(LearningError::Oversized {
            what: what.to_string(),
            max,
            actual: text.len(),
        });
    }
    Ok(text)
}

/// Hash length-prefixed parts with a crate domain tag. Length prefixes make
/// the encoding unambiguous: `["ab","c"]` and `["a","bc"]` hash differently.
pub(crate) fn hash_parts(parts: &[Vec<u8>]) -> FileHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(FINGERPRINT_DOMAIN_V1);
    hasher.update(&(parts.len() as u64).to_be_bytes());
    for part in parts {
        hasher.update(&(part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    FileHash::from(*hasher.finalize().as_bytes())
}

/// A task class is a free-form normalized label (for example `bugfix` or
/// `migration`); it is not the router's [`TaskClass`] enum because learning
/// classes are project vocabulary, not routing tiers.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct TaskClass(String);

impl TaskClass {
    /// Normalize and bound a task class. Empty input is malformed.
    pub fn new(raw: &str) -> Result<Self, LearningError> {
        let normalized = normalize_ws(raw);
        if normalized.is_empty() {
            return Err(LearningError::Malformed("task class is empty".to_string()));
        }
        bounded(normalized, MAX_FIELD_BYTES, "task class").map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for TaskClass {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        TaskClass::new(&raw).map_err(serde::de::Error::custom)
    }
}

/// The project/workspace a learning belongs to. Scoping is a first-class
/// data field, not a caller convention: hashing the scope into every
/// pattern key makes cross-project matches impossible, and the store keys
/// pages by the same value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct ProjectScope {
    /// The workspace root identity.
    pub workspace_id: WorkspaceId,
    /// Normalized project key inside the workspace (package/crate/project).
    pub project_key: String,
}

impl ProjectScope {
    /// Construct a scope with a normalized, bounded, non-empty project key.
    pub fn new(workspace_id: WorkspaceId, project_key: &str) -> Result<Self, LearningError> {
        let project_key = normalize_ws(project_key);
        if project_key.is_empty() {
            return Err(LearningError::Malformed("project key is empty".to_string()));
        }
        let project_key = bounded(project_key, MAX_PROJECT_KEY_BYTES, "project key")?;
        Ok(Self {
            workspace_id,
            project_key,
        })
    }

    /// Stable scope digest, domain-separated from every other digest.
    pub fn digest(&self) -> FileHash {
        hash_parts(&[
            b"project-scope".to_vec(),
            self.workspace_id.raw().to_be_bytes().to_vec(),
            self.project_key.as_bytes().to_vec(),
        ])
    }
}

impl<'de> Deserialize<'de> for ProjectScope {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            workspace_id: WorkspaceId,
            project_key: String,
        }
        let raw = Raw::deserialize(deserializer)?;
        ProjectScope::new(raw.workspace_id, &raw.project_key).map_err(serde::de::Error::custom)
    }
}

/// The environment an episode was observed in. The project scope is part of
/// the environment identity; `source_hash` is the workspace-wide source
/// revision the failure was verified against (the invalidation anchor).
///
/// [`EnvironmentFingerprint::pattern_digest`] deliberately EXCLUDES
/// `source_hash`: episodes that verify the same advice across revisions
/// accumulate into one pattern, while the latest recorded revision becomes
/// the pattern's invalidation anchor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EnvironmentFingerprint {
    /// Project/workspace scope (part of the identity, never a label only).
    pub project: ProjectScope,
    /// Normalized platform label (for example `linux` or `macos`).
    pub platform: String,
    /// Normalized toolchain label (for example `rustc-1.98`).
    pub toolchain: String,
    /// Source revision the episode was verified against, when known.
    pub source_hash: Option<FileHash>,
}

impl EnvironmentFingerprint {
    /// Normalize and bound platform/toolchain labels.
    pub fn new(
        project: ProjectScope,
        platform: &str,
        toolchain: &str,
        source_hash: Option<FileHash>,
    ) -> Result<Self, LearningError> {
        let platform = bounded(
            normalize_ws(platform),
            MAX_ENVIRONMENT_BYTES,
            "environment platform",
        )?;
        let toolchain = bounded(
            normalize_ws(toolchain),
            MAX_ENVIRONMENT_BYTES,
            "environment toolchain",
        )?;
        Ok(Self {
            project,
            platform,
            toolchain,
            source_hash,
        })
    }

    /// Pattern identity: project + platform + toolchain. Excludes the source
    /// revision so repeated verification across revisions accumulates.
    pub fn pattern_digest(&self) -> FileHash {
        hash_parts(&[
            b"environment-pattern".to_vec(),
            self.project.digest().bytes().to_vec(),
            self.platform.as_bytes().to_vec(),
            self.toolchain.as_bytes().to_vec(),
        ])
    }

    /// Full episode identity, including the source revision.
    pub fn digest(&self) -> FileHash {
        let mut parts = vec![
            b"environment".to_vec(),
            self.project.digest().bytes().to_vec(),
            self.platform.as_bytes().to_vec(),
            self.toolchain.as_bytes().to_vec(),
        ];
        match self.source_hash {
            Some(hash) => {
                parts.push(vec![1]);
                parts.push(hash.bytes().to_vec());
            }
            None => parts.push(vec![0]),
        }
        hash_parts(&parts)
    }
}

impl<'de> Deserialize<'de> for EnvironmentFingerprint {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            project: ProjectScope,
            platform: String,
            toolchain: String,
            source_hash: Option<FileHash>,
        }
        let raw = Raw::deserialize(deserializer)?;
        EnvironmentFingerprint::new(raw.project, &raw.platform, &raw.toolchain, raw.source_hash)
            .map_err(serde::de::Error::custom)
    }
}

/// One attempted action, described in project-normalized terms. Descriptors
/// are transient input; only [`ActionFingerprint`] is persisted and matched.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionDescriptor {
    /// Tool/verb (`edit`, `shell`, ...), lowercased.
    pub tool: String,
    /// Workspace-relative target (path or artifact id).
    pub target: String,
    /// Symbol the action touched, when known.
    pub symbol: Option<String>,
    /// Normalized summary of the action detail (arguments/commit shape).
    pub detail: String,
}

impl ActionDescriptor {
    pub fn new(
        tool: &str,
        target: &str,
        symbol: Option<&str>,
        detail: &str,
    ) -> Result<Self, LearningError> {
        let tool = bounded(
            normalize_ws(tool).to_lowercase(),
            MAX_FIELD_BYTES,
            "action tool",
        )?;
        if tool.is_empty() {
            return Err(LearningError::Malformed("action tool is empty".to_string()));
        }
        let target = bounded(normalize_path(target), MAX_FIELD_BYTES, "action target")?;
        let symbol = match symbol {
            Some(symbol) => {
                let symbol = normalize_ws(symbol);
                if symbol.is_empty() {
                    None
                } else {
                    Some(bounded(symbol, MAX_FIELD_BYTES, "action symbol")?)
                }
            }
            None => None,
        };
        let detail = bounded(normalize_ws(detail), MAX_FIELD_BYTES, "action detail")?;
        Ok(Self {
            tool,
            target,
            symbol,
            detail,
        })
    }

    /// Stable BLAKE3 digest of the normalized descriptor.
    pub fn fingerprint(&self) -> FileHash {
        let mut parts = vec![
            b"action-descriptor".to_vec(),
            self.tool.as_bytes().to_vec(),
            self.target.as_bytes().to_vec(),
            vec![u8::from(self.symbol.is_some())],
        ];
        if let Some(symbol) = &self.symbol {
            parts.push(symbol.as_bytes().to_vec());
        }
        parts.push(self.detail.as_bytes().to_vec());
        hash_parts(&parts)
    }
}

/// Stable digest of one attempted action (see [`ActionDescriptor`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ActionFingerprint(FileHash);

impl ActionFingerprint {
    /// Fingerprint a normalized descriptor.
    pub fn of(descriptor: &ActionDescriptor) -> Self {
        Self(descriptor.fingerprint())
    }

    /// Wrap an already-computed stable digest.
    pub const fn from_digest(digest: FileHash) -> Self {
        Self(digest)
    }

    pub const fn digest(self) -> FileHash {
        self.0
    }
}

impl fmt::Display for ActionFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.to_hex())
    }
}

/// One observed failure, described in project-normalized terms. The message
/// is DATA (never instructions) and is normalized to a single line before
/// hashing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureDescriptor {
    /// Failure kind (`compile_error`, `test_failure`, ...), lowercased.
    pub kind: String,
    /// Optional machine code (`E0308`, test name, exit code).
    pub code: Option<String>,
    /// Normalized single-line message (data only).
    pub message: String,
}

impl FailureDescriptor {
    pub fn new(kind: &str, code: Option<&str>, message: &str) -> Result<Self, LearningError> {
        let kind = bounded(
            normalize_ws(kind).to_lowercase(),
            MAX_FIELD_BYTES,
            "failure kind",
        )?;
        if kind.is_empty() {
            return Err(LearningError::Malformed(
                "failure kind is empty".to_string(),
            ));
        }
        let code = match code {
            Some(code) => {
                let code = normalize_ws(code);
                if code.is_empty() {
                    None
                } else {
                    Some(bounded(code, MAX_FIELD_BYTES, "failure code")?)
                }
            }
            None => None,
        };
        let message = bounded(normalize_ws(message), MAX_MESSAGE_BYTES, "failure message")?;
        Ok(Self {
            kind,
            code,
            message,
        })
    }

    /// Stable BLAKE3 digest of the normalized descriptor.
    pub fn fingerprint(&self) -> FileHash {
        let mut parts = vec![
            b"failure-descriptor".to_vec(),
            self.kind.as_bytes().to_vec(),
            vec![u8::from(self.code.is_some())],
        ];
        if let Some(code) = &self.code {
            parts.push(code.as_bytes().to_vec());
        }
        parts.push(self.message.as_bytes().to_vec());
        hash_parts(&parts)
    }
}

/// Stable digest of one observed failure (see [`FailureDescriptor`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FailureFingerprint(FileHash);

impl FailureFingerprint {
    /// Fingerprint a normalized descriptor.
    pub fn of(descriptor: &FailureDescriptor) -> Self {
        Self(descriptor.fingerprint())
    }

    /// Wrap an already-computed stable digest.
    pub const fn from_digest(digest: FileHash) -> Self {
        Self(digest)
    }

    pub const fn digest(self) -> FileHash {
        self.0
    }
}

impl fmt::Display for FailureFingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.to_hex())
    }
}

/// One assumption the recovery changed. Advice carries these as DATA.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssumptionDelta {
    /// Normalized assumption key.
    pub assumption: String,
    /// What was believed before recovery.
    pub previous: String,
    /// What is believed after verified recovery.
    pub updated: String,
}

impl AssumptionDelta {
    pub fn new(assumption: &str, previous: &str, updated: &str) -> Result<Self, LearningError> {
        let assumption = normalize_ws(assumption);
        if assumption.is_empty() {
            return Err(LearningError::Malformed(
                "assumption key is empty".to_string(),
            ));
        }
        Ok(Self {
            assumption: bounded(assumption, MAX_FIELD_BYTES, "assumption key")?,
            previous: bounded(
                normalize_ws(previous),
                MAX_FIELD_BYTES,
                "assumption previous",
            )?,
            updated: bounded(normalize_ws(updated), MAX_FIELD_BYTES, "assumption updated")?,
        })
    }
}

/// One durable failure/recovery episode. See the module docs for the
/// verified-success contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureEpisode {
    pub id: EpisodeId,
    pub task_class: TaskClass,
    pub environment_fingerprint: EnvironmentFingerprint,
    pub attempted_action: ActionFingerprint,
    pub failure: FailureFingerprint,
    pub recovery_actions: Vec<ActionFingerprint>,
    /// The durable verification record that certifies the eventual success.
    /// `None` means no verification was recorded and the episode can never
    /// produce a learning.
    pub eventual_verified_success: Option<VerificationRecordId>,
    pub changed_assumptions: Vec<AssumptionDelta>,
    pub evidence: Vec<EvidenceId>,
}

impl FailureEpisode {
    pub fn new(
        id: EpisodeId,
        task_class: TaskClass,
        environment_fingerprint: EnvironmentFingerprint,
        attempted_action: ActionFingerprint,
        failure: FailureFingerprint,
    ) -> Self {
        Self {
            id,
            task_class,
            environment_fingerprint,
            attempted_action,
            failure,
            recovery_actions: Vec::new(),
            eventual_verified_success: None,
            changed_assumptions: Vec::new(),
            evidence: Vec::new(),
        }
    }

    /// Attach the recovery chain. An empty or oversized chain is refused.
    pub fn with_recovery_actions(
        mut self,
        recovery_actions: Vec<ActionFingerprint>,
    ) -> Result<Self, LearningError> {
        if recovery_actions.is_empty() {
            return Err(LearningError::Refused(
                "a failure episode without a recovery chain cannot be mined".to_string(),
            ));
        }
        if recovery_actions.len() > MAX_RECOVERY_ACTIONS {
            return Err(LearningError::Oversized {
                what: "recovery chain".to_string(),
                max: MAX_RECOVERY_ACTIONS,
                actual: recovery_actions.len(),
            });
        }
        self.recovery_actions = recovery_actions;
        Ok(self)
    }

    /// Record the durable verification record. This is the only success
    /// signal the API accepts.
    pub fn verified(mut self, record: VerificationRecordId) -> Self {
        self.eventual_verified_success = Some(record);
        self
    }

    /// Attach changed assumptions, bounded.
    pub fn with_changed_assumptions(
        mut self,
        changed_assumptions: Vec<AssumptionDelta>,
    ) -> Result<Self, LearningError> {
        if changed_assumptions.len() > MAX_ASSUMPTIONS {
            return Err(LearningError::Oversized {
                what: "changed assumptions".to_string(),
                max: MAX_ASSUMPTIONS,
                actual: changed_assumptions.len(),
            });
        }
        self.changed_assumptions = changed_assumptions;
        Ok(self)
    }

    /// Attach evidence ids, bounded.
    pub fn with_evidence(mut self, evidence: Vec<EvidenceId>) -> Result<Self, LearningError> {
        if evidence.len() > MAX_EVIDENCE {
            return Err(LearningError::Oversized {
                what: "episode evidence".to_string(),
                max: MAX_EVIDENCE,
                actual: evidence.len(),
            });
        }
        self.evidence = evidence;
        Ok(self)
    }

    pub fn project(&self) -> &ProjectScope {
        &self.environment_fingerprint.project
    }

    pub fn is_verified_success(&self) -> bool {
        self.eventual_verified_success.is_some()
    }

    /// The learning pattern identity this episode belongs to: project scope,
    /// task class, environment pattern (revision-independent), attempted
    /// action, failure, and recovery chain. Source revisions deliberately do
    /// NOT split the pattern.
    pub fn pattern_digest(&self) -> FileHash {
        pattern_key_of(
            self.project(),
            &self.task_class,
            self.environment_fingerprint.pattern_digest(),
            &self.attempted_action,
            &self.failure,
            &self.recovery_actions,
        )
    }

    /// Identity of ONE observed event, used to dedupe duplicate episode
    /// records before counting. Deliberately excludes evidence ids and
    /// changed-assumption text (payload, not identity) and INCLUDES the
    /// verification record, so repeated verified events with distinct
    /// records count separately while replays of the same record do not.
    pub fn dedupe_digest(&self) -> FileHash {
        let mut parts = vec![
            b"episode-dedupe".to_vec(),
            self.pattern_digest().bytes().to_vec(),
        ];
        match self.eventual_verified_success {
            Some(record) => {
                parts.push(vec![1]);
                parts.push(record.raw().to_be_bytes().to_vec());
            }
            None => parts.push(vec![0]),
        }
        hash_parts(&parts)
    }
}

/// Shared pattern identity so an episode and a [`LearningPattern`]
/// (crate::miner::LearningPattern) compute the same digest.
pub(crate) fn pattern_key_of(
    project: &ProjectScope,
    task_class: &TaskClass,
    environment_pattern: FileHash,
    attempted_action: &ActionFingerprint,
    failure: &FailureFingerprint,
    recovery_actions: &[ActionFingerprint],
) -> FileHash {
    let mut parts = vec![
        b"learning-pattern".to_vec(),
        project.digest().bytes().to_vec(),
        task_class.as_str().as_bytes().to_vec(),
        environment_pattern.bytes().to_vec(),
        attempted_action.digest().bytes().to_vec(),
        failure.digest().bytes().to_vec(),
        (recovery_actions.len() as u64).to_be_bytes().to_vec(),
    ];
    for action in recovery_actions {
        parts.push(action.digest().bytes().to_vec());
    }
    hash_parts(&parts)
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    pub(crate) fn project(workspace: u64, key: &str) -> ProjectScope {
        ProjectScope::new(WorkspaceId::new(workspace), key).unwrap()
    }

    pub(crate) fn environment(
        project: ProjectScope,
        source_hash: Option<FileHash>,
    ) -> EnvironmentFingerprint {
        EnvironmentFingerprint::new(project, "linux", "rustc-1.98", source_hash).unwrap()
    }

    pub(crate) fn action(
        tool: &str,
        target: &str,
        symbol: Option<&str>,
        detail: &str,
    ) -> ActionFingerprint {
        ActionFingerprint::of(&ActionDescriptor::new(tool, target, symbol, detail).unwrap())
    }

    pub(crate) fn failure(kind: &str, code: Option<&str>, message: &str) -> FailureFingerprint {
        FailureFingerprint::of(&FailureDescriptor::new(kind, code, message).unwrap())
    }

    pub(crate) fn assumption(assumption: &str, previous: &str, updated: &str) -> AssumptionDelta {
        AssumptionDelta::new(assumption, previous, updated).unwrap()
    }

    pub(crate) fn episode(
        id: u64,
        project: ProjectScope,
        verification: Option<VerificationRecordId>,
    ) -> FailureEpisode {
        let mut episode = FailureEpisode::new(
            EpisodeId::new(id),
            TaskClass::new("bugfix").unwrap(),
            environment(project, None),
            action("edit", "src/lib.rs", Some("parse"), "replace body"),
            failure("test_failure", None, "assertion failed"),
        )
        .with_recovery_actions(vec![action(
            "edit",
            "src/lib.rs",
            Some("parse"),
            "fix guard",
        )])
        .unwrap();
        if let Some(record) = verification {
            episode = episode.verified(record);
        }
        episode
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use faktor_core::id::VerificationRecordId;

    #[test]
    fn fingerprints_are_stable_across_spelling_and_whitespace() {
        let a = action("Edit", "src/lib.rs", Some("parse"), "replace   body");
        let b = action("edit", "./src/lib.rs", Some("parse"), " replace body ");
        let c = action("edit", "src\\lib.rs", Some("  parse "), "replace\nbody");
        assert_eq!(a, b);
        assert_eq!(a, c);

        let d = failure("Test_Failure", None, "assertion  failed");
        let e = failure("test_failure", None, " assertion failed ");
        assert_eq!(d, e);
        // Kind is case-folded, the message content still matters.
        assert_ne!(
            failure("test_failure", None, "assertion failed"),
            failure("test_failure", None, "assertion failed elsewhere")
        );
    }

    #[test]
    fn different_symbols_yield_different_fingerprints() {
        let a = action("edit", "src/lib.rs", Some("parse"), "fix");
        let b = action("edit", "src/lib.rs", Some("render"), "fix");
        let c = action("edit", "src/lib.rs", None, "fix");
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test]
    fn field_boundary_smuggling_cannot_collide() {
        // Length prefixes make ["ab","c"] and ["a","bc"] distinct even
        // though a naive concatenation would merge them.
        let one = hash_parts(&[b"ab".to_vec(), b"c".to_vec()]);
        let two = hash_parts(&[b"a".to_vec(), b"bc".to_vec()]);
        assert_ne!(one, two);
    }

    #[test]
    fn project_scope_digest_is_workspace_and_key_sensitive() {
        let a = project(1, "alpha").digest();
        let b = project(2, "alpha").digest();
        let c = project(1, "beta").digest();
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(a, project(1, " alpha ").digest());
    }

    #[test]
    fn constructors_reject_malformed_and_oversized_input() {
        assert!(matches!(
            TaskClass::new("   "),
            Err(LearningError::Malformed(_))
        ));
        assert!(matches!(
            ProjectScope::new(WorkspaceId::new(1), ""),
            Err(LearningError::Malformed(_))
        ));
        assert!(matches!(
            ActionDescriptor::new(" ", "src/lib.rs", None, "x"),
            Err(LearningError::Malformed(_))
        ));
        assert!(matches!(
            FailureDescriptor::new("", None, "x"),
            Err(LearningError::Malformed(_))
        ));
        assert!(matches!(
            AssumptionDelta::new("", "a", "b"),
            Err(LearningError::Malformed(_))
        ));

        let huge = "x".repeat(MAX_FIELD_BYTES + 1);
        assert!(matches!(
            TaskClass::new(&huge),
            Err(LearningError::Oversized { .. })
        ));
        assert!(matches!(
            ActionDescriptor::new("edit", &huge, None, "x"),
            Err(LearningError::Oversized { .. })
        ));
        assert!(matches!(
            FailureDescriptor::new("test_failure", None, &"m".repeat(MAX_MESSAGE_BYTES + 1)),
            Err(LearningError::Oversized { .. })
        ));
    }

    #[test]
    fn episode_builder_refuses_empty_and_oversized_recovery_chains() {
        let base = FailureEpisode::new(
            EpisodeId::new(1),
            TaskClass::new("bugfix").unwrap(),
            environment(project(1, "alpha"), None),
            action("edit", "src/lib.rs", None, "x"),
            failure("test_failure", None, "boom"),
        );
        assert!(matches!(
            base.clone().with_recovery_actions(Vec::new()),
            Err(LearningError::Refused(_))
        ));
        let chain = vec![action("edit", "src/lib.rs", None, "x"); MAX_RECOVERY_ACTIONS + 1];
        assert!(matches!(
            base.with_recovery_actions(chain),
            Err(LearningError::Oversized { .. })
        ));
    }

    #[test]
    fn episode_id_zero_is_rejected_on_deserialize() {
        assert!(serde_json::from_str::<EpisodeId>("0").is_err());
        assert!(serde_json::from_str::<EpisodeId>("-1").is_err());
        assert!(serde_json::from_str::<EpisodeId>("12.5").is_err());
        assert_eq!(
            serde_json::from_str::<EpisodeId>("12").unwrap(),
            EpisodeId::new(12)
        );
    }

    #[test]
    fn repeated_verified_events_are_distinct_samples_but_replays_dedupe() {
        let project = project(1, "alpha");
        let one = episode(1, project.clone(), Some(VerificationRecordId::new(9)));
        let replay = FailureEpisode {
            evidence: vec![EvidenceId(77)],
            changed_assumptions: vec![assumption("guard", "old", "new")],
            ..one.clone()
        };
        assert_eq!(one.dedupe_digest(), replay.dedupe_digest());

        let other_record = episode(2, project, Some(VerificationRecordId::new(10)));
        assert_ne!(one.dedupe_digest(), other_record.dedupe_digest());
    }

    #[test]
    fn pattern_digest_ignores_source_revision_but_includes_project() {
        let scope = project(1, "alpha");
        let mut a = episode(1, scope.clone(), Some(VerificationRecordId::new(1)));
        a.environment_fingerprint.source_hash = Some(FileHash::from([1; 32]));
        let mut b = episode(2, scope, Some(VerificationRecordId::new(2)));
        b.environment_fingerprint.source_hash = Some(FileHash::from([2; 32]));
        assert_eq!(a.pattern_digest(), b.pattern_digest());
        assert_ne!(
            a.environment_fingerprint.digest(),
            b.environment_fingerprint.digest()
        );
    }
}
