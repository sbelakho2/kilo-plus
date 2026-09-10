//! The miner (audit items 65-67/92/106): verified failure/recovery episodes
//! become project learnings.
//!
//! Eligibility is deliberately narrow:
//!
//! - the episode must carry a non-empty recovery chain;
//! - the episode must carry a durable verification record
//!   (`eventual_verified_success`). Zero verified supporting episodes means
//!   zero learnings — a model claiming a fix is not an input;
//! - duplicate episode records are deduped by fingerprint before counting,
//!   so replaying the ledger cannot inflate confidence.
//!
//! Confidence is a documented Beta-Binomial posterior mean with a
//! conservative `Beta(1, 3)` prior:
//!
//! ```text
//! confidence_ppm = 1e6 * (PRIOR_ALPHA + verified) / (PRIOR_ALPHA + PRIOR_BETA + verified + unverified)
//! ```
//!
//! One verified clean sample sits at 400k ppm and two at 500k ppm — both
//! below [`CONFIDENCE_TRUSTED_PPM`] (600k); four clean samples reach 625k.
//! Unverified same-pattern attempts count in the denominator, so they shrink
//! confidence instead of disappearing.
//!
//! Invalidation is chosen deterministically:
//!
//! 1. any verified supporting source revision -> [`InvalidationRule::SourceHashChanged`]
//!    (the greatest recorded digest is the anchor);
//! 2. else any evidence id -> [`InvalidationRule::EvidenceStale`] (greatest id);
//! 3. else -> [`InvalidationRule::TaskEnd`].
//!
//! [`InvalidationRule::Never`] is never emitted by the miner; it is an
//! explicit caller opt-out for hand-authored learnings.

use std::collections::{HashMap, HashSet};

use faktor_core::hash::FileHash;
use faktor_evidence::{
    provenance::{assert_not_instruction_authority, is_instruction_authority},
    types::{EvidenceId, ProvenanceSet, ProvenanceSource},
};
use serde::{Deserialize, Serialize};

use crate::episode::{
    pattern_key_of, ActionFingerprint, AssumptionDelta, EpisodeId, FailureEpisode,
    FailureFingerprint, ProjectScope, TaskClass, MAX_ASSUMPTIONS, MAX_EVIDENCE,
    MAX_RECOVERY_ACTIONS,
};
use crate::LearningError;

/// Beta prior pseudo-count for verified successes.
pub const PRIOR_ALPHA: u64 = 1;
/// Beta prior pseudo-count for unverified attempts.
pub const PRIOR_BETA: u64 = 3;
/// The documented trust bar in ppm: no learning below it is treated as
/// established advice. One and two clean verified samples never clear it.
pub const CONFIDENCE_TRUSTED_PPM: u32 = 600_000;
/// Maximum source hashes retained per learning.
const MAX_SOURCE_HASHES: usize = 64;

/// Conservative Beta-Binomial posterior mean of the verified-success rate,
/// in ppm. See the module docs for the exact formula and calibration.
pub fn bayesian_confidence_ppm(successes: u64, failures: u64) -> u32 {
    let attempts = successes.saturating_add(failures);
    let denominator = u128::from(PRIOR_ALPHA + PRIOR_BETA) + u128::from(attempts);
    let numerator = u128::from(PRIOR_ALPHA) + u128::from(successes);
    let ppm = (numerator * 1_000_000 + denominator / 2) / denominator;
    u32::try_from(ppm).unwrap_or(1_000_000).min(1_000_000)
}

/// The matchable identity of a learning: project scope, task class,
/// revision-independent environment pattern, attempted action, failure, and
/// the verified recovery chain. Two patterns with identical symbol names in
/// different projects are different patterns because the project scope is
/// hashed into the digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearningPattern {
    pub project: ProjectScope,
    /// Environment PATTERN digest (project + platform + toolchain; excludes
    /// the source revision).
    pub environment: FileHash,
    pub task_class: TaskClass,
    pub attempted_action: ActionFingerprint,
    pub failure: FailureFingerprint,
    pub recovery_actions: Vec<ActionFingerprint>,
}

impl LearningPattern {
    /// Stable pattern identity, computed exactly like
    /// [`FailureEpisode::pattern_digest`](crate::episode::FailureEpisode::pattern_digest).
    pub fn digest(&self) -> FileHash {
        pattern_key_of(
            &self.project,
            &self.task_class,
            self.environment,
            &self.attempted_action,
            &self.failure,
            &self.recovery_actions,
        )
    }

    /// True only when the episode belongs to this exact pattern, including
    /// the project scope. Cross-project matches are impossible.
    pub fn matches_episode(&self, episode: &FailureEpisode) -> bool {
        self.project == *episode.project()
            && self.task_class == episode.task_class
            && self.environment == episode.environment_fingerprint.pattern_digest()
            && self.attempted_action == episode.attempted_action
            && self.failure == episode.failure
            && self.recovery_actions == episode.recovery_actions
    }
}

/// When a learning stops being valid. The miner emits the first applicable
/// rule; `Never` is an explicit caller opt-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InvalidationRule {
    /// Valid until explicitly removed. Never emitted by [`mine`].
    Never,
    /// Invalid when the currently observed source revision differs from the
    /// recorded supporting revision. Unknown current revision (no way to
    /// prove a change) does NOT invalidate.
    SourceHashChanged(FileHash),
    /// Invalid when the recorded supporting evidence id is stale.
    EvidenceStale(EvidenceId),
    /// Invalid once the producing task has ended.
    TaskEnd,
}

/// The world state invalidation rules are evaluated against. Scoping to the
/// project is the caller's job (`LearningService` methods take a
/// [`ProjectScope`](crate::episode::ProjectScope)).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InvalidationContext {
    /// Currently observed source revision, when known.
    pub current_source_hash: Option<FileHash>,
    /// Evidence ids the evidence layer has declared stale.
    pub stale_evidence: Vec<EvidenceId>,
    /// Whether the producing task has ended.
    pub task_ended: bool,
}

impl InvalidationContext {
    /// Context with a changed source revision.
    pub fn source_changed_to(current_source_hash: FileHash) -> Self {
        Self {
            current_source_hash: Some(current_source_hash),
            ..Self::default()
        }
    }

    /// Context for a task that has ended.
    pub fn task_ended() -> Self {
        Self {
            task_ended: true,
            ..Self::default()
        }
    }

    /// Mark one evidence id stale.
    pub fn with_stale_evidence(mut self, id: EvidenceId) -> Self {
        self.stale_evidence.push(id);
        self
    }

    /// True when `id` is in the stale set.
    pub fn is_evidence_stale(&self, id: EvidenceId) -> bool {
        self.stale_evidence.contains(&id)
    }
}

impl InvalidationRule {
    /// Evaluate this rule against the world. `SourceHashChanged` with no
    /// current revision is NOT invalidated: a change that cannot be proven
    /// is not asserted.
    pub fn is_invalidated(self, context: &InvalidationContext) -> bool {
        match self {
            InvalidationRule::Never => false,
            InvalidationRule::SourceHashChanged(recorded) => context
                .current_source_hash
                .is_some_and(|current| current != recorded),
            InvalidationRule::EvidenceStale(id) => context.is_evidence_stale(id),
            InvalidationRule::TaskEnd => context.task_ended,
        }
    }

    /// Short human-readable reason tag (data, for diagnostics only).
    pub const fn reason(self) -> &'static str {
        match self {
            InvalidationRule::Never => "never",
            InvalidationRule::SourceHashChanged(_) => "source_hash_changed",
            InvalidationRule::EvidenceStale(_) => "evidence_stale",
            InvalidationRule::TaskEnd => "task_end",
        }
    }
}

/// Structured advice produced from a verified recovery chain. The advice is
/// DATA: [`StructuredAdvice::is_instruction_authority`] can only ever be
/// true if a caller hand-builds a `UserPolicy` provenance set, and
/// [`StructuredAdvice::assert_data_only`] refuses exactly that before
/// rendering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredAdvice {
    /// One-line normalized summary.
    pub summary: String,
    /// One data line per recovery step (action fingerprints).
    pub recovery_steps: Vec<String>,
    /// Assumptions the verified recovery changed.
    pub changed_assumptions: Vec<AssumptionDelta>,
    /// Where the advice came from. The miner always uses
    /// `[Tool, Repository, Verification]` — never `UserPolicy`.
    pub provenance: ProvenanceSet,
}

impl StructuredAdvice {
    /// Construct data-only advice with the miner's provenance set.
    pub fn data_only(
        summary: String,
        recovery_steps: Vec<String>,
        changed_assumptions: Vec<AssumptionDelta>,
    ) -> Self {
        Self {
            summary,
            recovery_steps,
            changed_assumptions,
            provenance: ProvenanceSet::new([
                ProvenanceSource::Tool,
                ProvenanceSource::Repository,
                ProvenanceSource::Verification,
            ]),
        }
    }

    /// True only for `UserPolicy` provenance (the evidence rule).
    pub fn is_instruction_authority(&self) -> bool {
        is_instruction_authority(&self.provenance)
    }

    /// Guard for every DATA render path: refuses advice that carries
    /// instruction authority instead of laundering it through the data
    /// channel.
    pub fn assert_data_only(&self) -> Result<(), LearningError> {
        assert_not_instruction_authority(&self.provenance)
            .map_err(|error| LearningError::Refused(error.to_string()))
    }
}

/// One mined project learning. Learnings are DATA: they can be rendered for
/// context, never executed as instructions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectLearning {
    pub pattern: LearningPattern,
    pub advice: StructuredAdvice,
    /// Distinct deduped attempts behind this pattern (verified + unverified).
    pub sample_count: u32,
    /// Conservative verified-success confidence in ppm (0..=1_000_000).
    pub confidence_ppm: u32,
    /// The verified supporting episodes (sorted by id).
    pub supporting_episodes: Vec<EpisodeId>,
    pub invalidation: InvalidationRule,
}

impl ProjectLearning {
    pub fn pattern_digest(&self) -> FileHash {
        self.pattern.digest()
    }

    pub fn project(&self) -> &ProjectScope {
        &self.pattern.project
    }

    pub fn is_invalidated(&self, context: &InvalidationContext) -> bool {
        self.invalidation.is_invalidated(context)
    }

    /// Whether the confidence clears the documented trust bar.
    pub fn is_trusted(&self) -> bool {
        self.confidence_ppm >= CONFIDENCE_TRUSTED_PPM
    }
}

struct Group {
    pattern: LearningPattern,
    verified: Vec<EpisodeId>,
    unverified: u32,
    evidence: Vec<EvidenceId>,
    source_hashes: Vec<FileHash>,
    canonical: Option<(EpisodeId, Vec<AssumptionDelta>)>,
}

impl Group {
    fn new(pattern: LearningPattern) -> Self {
        Self {
            pattern,
            verified: Vec::new(),
            unverified: 0,
            evidence: Vec::new(),
            source_hashes: Vec::new(),
            canonical: None,
        }
    }

    fn record_verified(&mut self, episode: &FailureEpisode) {
        self.verified.push(episode.id);
        if self.source_hashes.len() < MAX_SOURCE_HASHES {
            if let Some(hash) = episode.environment_fingerprint.source_hash {
                if !self.source_hashes.contains(&hash) {
                    self.source_hashes.push(hash);
                }
            }
        }
        for id in &episode.evidence {
            if self.evidence.len() >= MAX_EVIDENCE {
                break;
            }
            if !self.evidence.contains(id) {
                self.evidence.push(*id);
            }
        }
        let replace = match &self.canonical {
            None => true,
            Some((existing_id, _)) => episode.id < *existing_id,
        };
        if replace {
            let assumptions = episode
                .changed_assumptions
                .iter()
                .take(MAX_ASSUMPTIONS)
                .cloned()
                .collect();
            self.canonical = Some((episode.id, assumptions));
        }
    }

    fn record_unverified(&mut self) {
        self.unverified = self.unverified.saturating_add(1);
    }

    fn finish(self) -> Option<ProjectLearning> {
        if self.verified.is_empty() {
            return None;
        }
        let successes = u64::try_from(self.verified.len()).unwrap_or(u64::MAX);
        let failures = u64::from(self.unverified);
        let confidence_ppm = bayesian_confidence_ppm(successes, failures);
        let sample_count = u32::try_from(successes.saturating_add(failures)).unwrap_or(u32::MAX);
        let mut supporting_episodes = self.verified;
        supporting_episodes.sort();
        let assumptions = self
            .canonical
            .map(|(_, assumptions)| assumptions)
            .unwrap_or_default();
        let advice = build_advice(&self.pattern, &assumptions, successes);
        let invalidation = pick_invalidation(&self.source_hashes, &self.evidence);
        Some(ProjectLearning {
            pattern: self.pattern,
            advice,
            sample_count,
            confidence_ppm,
            supporting_episodes,
            invalidation,
        })
    }
}

fn pick_invalidation(source_hashes: &[FileHash], evidence: &[EvidenceId]) -> InvalidationRule {
    if let Some(hash) = source_hashes.iter().max_by_key(|hash| hash.to_hex()) {
        return InvalidationRule::SourceHashChanged(*hash);
    }
    if let Some(id) = evidence.iter().max_by_key(|id| id.0) {
        return InvalidationRule::EvidenceStale(*id);
    }
    InvalidationRule::TaskEnd
}

fn short_digest(hash: FileHash) -> String {
    hash.to_hex().chars().take(16).collect()
}

fn build_advice(
    pattern: &LearningPattern,
    assumptions: &[AssumptionDelta],
    successes: u64,
) -> StructuredAdvice {
    let failure = short_digest(pattern.failure.digest());
    let action = short_digest(pattern.attempted_action.digest());
    let summary = format!(
        "verified recovery for {} task: failure {failure} after action {action} cleared by {} recovery step(s) in {successes} verified episode(s)",
        pattern.task_class.as_str(),
        pattern.recovery_actions.len()
    );
    let recovery_steps = pattern
        .recovery_actions
        .iter()
        .enumerate()
        .map(|(index, action)| {
            format!(
                "step {}: action {}",
                index + 1,
                short_digest(action.digest())
            )
        })
        .collect();
    StructuredAdvice::data_only(summary, recovery_steps, assumptions.to_vec())
}

/// Mine learnings from episodes. The function is total: ineligible episodes
/// (empty recovery chain, oversized chain, no recovery chain, no verified
/// success) are skipped, duplicates are deduped by fingerprint, and the
/// output is sorted by pattern digest so results do not depend on hash-map
/// iteration order.
pub fn mine(episodes: &[FailureEpisode]) -> Vec<ProjectLearning> {
    let mut seen: HashSet<FileHash> = HashSet::new();
    let mut groups: HashMap<FileHash, Group> = HashMap::new();
    for episode in episodes {
        if episode.recovery_actions.is_empty()
            || episode.recovery_actions.len() > MAX_RECOVERY_ACTIONS
        {
            continue;
        }
        if !seen.insert(episode.dedupe_digest()) {
            continue;
        }
        let key = episode.pattern_digest();
        let group = groups.entry(key).or_insert_with(|| {
            Group::new(LearningPattern {
                project: episode.project().clone(),
                environment: episode.environment_fingerprint.pattern_digest(),
                task_class: episode.task_class.clone(),
                attempted_action: episode.attempted_action,
                failure: episode.failure,
                recovery_actions: episode.recovery_actions.clone(),
            })
        });
        if episode.eventual_verified_success.is_some() {
            group.record_verified(episode);
        } else {
            group.record_unverified();
        }
    }
    let mut learnings: Vec<ProjectLearning> =
        groups.into_values().filter_map(Group::finish).collect();
    learnings.sort_by_key(|learning| learning.pattern_digest().to_hex());
    learnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::test_support::{action, assumption, episode, project};
    use faktor_core::id::VerificationRecordId;

    fn verified(id: u64, workspace: u64, key: &str, record: u64) -> FailureEpisode {
        episode(
            id,
            project(workspace, key),
            Some(VerificationRecordId::new(record)),
        )
    }

    #[test]
    fn no_verified_success_never_produces_a_learning() {
        let unverified = episode(1, project(1, "alpha"), None);
        assert!(mine(&[unverified]).is_empty());

        // The API has no "model_said_fixed" input: even with a recovery
        // chain and evidence, absence of a verification record means no
        // learning.
        let mut with_evidence = episode(2, project(1, "alpha"), None);
        with_evidence.evidence = vec![EvidenceId(5)];
        assert!(mine(&[with_evidence]).is_empty());
    }

    #[test]
    fn duplicate_episode_records_dedupe_by_fingerprint() {
        let mut first = verified(1, 1, "alpha", 7);
        first.evidence = vec![EvidenceId(1)];
        first.changed_assumptions = vec![assumption("guard", "old", "new")];
        let mut second = first.clone();
        second.id = EpisodeId::new(2);
        second.evidence = vec![EvidenceId(2), EvidenceId(3)];
        second.changed_assumptions = vec![assumption("guard", "old", "newer")];
        let mut third = first.clone();
        third.id = EpisodeId::new(3);
        third.environment_fingerprint.platform = "macos".to_string();

        // Third differs in environment, so it is a different pattern; the
        // first two are replays of one event and collapse to one sample.
        let macos_pattern = third.environment_fingerprint.pattern_digest();
        let learnings = mine(&[first, second, third]);
        let alpha = learnings
            .iter()
            .find(|learning| learning.pattern.environment != macos_pattern)
            .expect("linux pattern mined");
        assert_eq!(alpha.sample_count, 1);
        assert_eq!(alpha.confidence_ppm, 400_000);
        assert_eq!(alpha.supporting_episodes, vec![EpisodeId::new(1)]);
    }

    #[test]
    fn confidence_is_conservative_and_rises_only_with_verified_history() {
        // 1 clean verified sample -> 400k, 2 -> 500k; both below the bar.
        assert_eq!(bayesian_confidence_ppm(1, 0), 400_000);
        assert_eq!(bayesian_confidence_ppm(2, 0), 500_000);
        assert!(bayesian_confidence_ppm(1, 0) < CONFIDENCE_TRUSTED_PPM);
        assert!(bayesian_confidence_ppm(2, 0) < CONFIDENCE_TRUSTED_PPM);
        // Monotone in verified samples; unverified same-pattern attempts
        // shrink confidence and are counted in sample_count.
        let one = mine(&[verified(1, 1, "alpha", 1)]);
        let two = mine(&[verified(1, 1, "alpha", 1), verified(2, 1, "alpha", 2)]);
        let four = mine(&[
            verified(1, 1, "alpha", 1),
            verified(2, 1, "alpha", 2),
            verified(3, 1, "alpha", 3),
            verified(4, 1, "alpha", 4),
        ]);
        assert!(one[0].confidence_ppm < two[0].confidence_ppm);
        assert!(two[0].confidence_ppm < four[0].confidence_ppm);
        assert!(four[0].is_trusted());
        assert!(two[0].confidence_ppm < CONFIDENCE_TRUSTED_PPM);

        let with_unverified = mine(&[
            verified(1, 1, "alpha", 1),
            episode(2, project(1, "alpha"), None),
        ]);
        assert_eq!(with_unverified[0].sample_count, 2);
        assert_eq!(with_unverified[0].confidence_ppm, 333_333);
        assert!(with_unverified[0].confidence_ppm < one[0].confidence_ppm);
    }

    #[test]
    fn supporting_source_hash_change_invalidates() {
        let first = FileHash::from([1; 32]);
        let second = FileHash::from([2; 32]);
        let mut episode = verified(1, 1, "alpha", 1);
        episode.environment_fingerprint.source_hash = Some(first);
        let learnings = mine(&[episode]);
        assert_eq!(
            learnings[0].invalidation,
            InvalidationRule::SourceHashChanged(first)
        );
        assert!(!learnings[0].is_invalidated(&InvalidationContext::default()));
        assert!(!learnings[0].is_invalidated(&InvalidationContext::source_changed_to(first)));
        assert!(learnings[0].is_invalidated(&InvalidationContext::source_changed_to(second)));
        // Unknown current revision cannot prove a change: still valid.
        assert!(!learnings[0].is_invalidated(&InvalidationContext {
            current_source_hash: None,
            ..InvalidationContext::default()
        }));
    }

    #[test]
    fn evidence_staleness_invalidates_when_no_source_hash_exists() {
        let mut episode = verified(1, 1, "alpha", 1);
        episode.evidence = vec![EvidenceId(4), EvidenceId(9)];
        let learnings = mine(&[episode]);
        assert_eq!(
            learnings[0].invalidation,
            InvalidationRule::EvidenceStale(EvidenceId(9))
        );
        assert!(!learnings[0]
            .is_invalidated(&InvalidationContext::default().with_stale_evidence(EvidenceId(4))));
        assert!(learnings[0]
            .is_invalidated(&InvalidationContext::default().with_stale_evidence(EvidenceId(9))));
    }

    #[test]
    fn task_end_and_never_rules_evaluate_structurally() {
        let learnings = mine(&[verified(1, 1, "alpha", 1)]);
        assert_eq!(learnings[0].invalidation, InvalidationRule::TaskEnd);
        assert!(learnings[0].is_invalidated(&InvalidationContext::task_ended()));
        assert!(!learnings[0].is_invalidated(&InvalidationContext::default()));
        assert!(!InvalidationRule::Never.is_invalidated(&InvalidationContext::task_ended()));
    }

    #[test]
    fn project_isolation_prevents_identical_symbol_names_cross_matching() {
        let alpha = verified(1, 1, "alpha", 1);
        let same_key_other_workspace = verified(2, 2, "alpha", 2);
        let same_workspace_other_project = verified(3, 1, "beta", 3);
        let learnings = mine(&[
            alpha.clone(),
            same_key_other_workspace.clone(),
            same_workspace_other_project.clone(),
        ]);
        assert_eq!(
            learnings.len(),
            3,
            "identical symbols in other projects stay distinct"
        );
        let digests: HashSet<String> = learnings
            .iter()
            .map(|learning| learning.pattern_digest().to_hex())
            .collect();
        assert_eq!(digests.len(), 3);

        let alpha_learning = learnings
            .iter()
            .find(|learning| learning.pattern.project == *alpha.project())
            .unwrap();
        assert!(alpha_learning.pattern.matches_episode(&alpha));
        assert!(!alpha_learning
            .pattern
            .matches_episode(&same_key_other_workspace));
        assert!(!alpha_learning
            .pattern
            .matches_episode(&same_workspace_other_project));
    }

    #[test]
    fn mined_advice_is_data_only_and_policy_provenance_is_refused() {
        let learnings = mine(&[verified(1, 1, "alpha", 1)]);
        let advice = &learnings[0].advice;
        assert!(!advice.is_instruction_authority());
        assert!(advice.assert_data_only().is_ok());
        assert!(advice.recovery_steps.len() == 1);

        let mut hostile = advice.clone();
        hostile.provenance = ProvenanceSet::new([ProvenanceSource::UserPolicy]);
        assert!(hostile.is_instruction_authority());
        assert!(matches!(
            hostile.assert_data_only(),
            Err(LearningError::Refused(_))
        ));
    }

    #[test]
    fn mining_is_deterministic_and_ignores_ineligible_shapes() {
        let empty_recovery = FailureEpisode {
            recovery_actions: Vec::new(),
            ..verified(1, 1, "alpha", 1)
        };
        let mut oversized = verified(2, 1, "beta", 2);
        oversized.recovery_actions =
            vec![action("edit", "src/lib.rs", None, "x"); MAX_RECOVERY_ACTIONS + 1];
        let learnings = mine(&[empty_recovery, oversized]);
        assert!(learnings.is_empty());

        let episodes = vec![verified(2, 1, "alpha", 2), verified(1, 1, "alpha", 1)];
        let forward = mine(&episodes);
        let reversed = mine(&[episodes[1].clone(), episodes[0].clone()]);
        assert_eq!(forward, reversed);
    }
}
