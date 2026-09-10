//! The persistence seam (audit items 92/106): [`LearningStore`] is the only
//! way learnings are persisted. [`MemoryLearningStore`] is the bounded
//! in-process implementation; [`SessionLearningStore`] is the durable
//! adapter over the session's typed ledger rows.
//!
//! **No schema is owned here and no migration is added.** The durable
//! adapter maps the trait onto `learning_record` entries of the session
//! typed ledger (`faktor-session`), one bounded JSON payload per row:
//!
//! ```text
//! upsert             -> append a `learning` row; the pattern's FIRST row
//!                       seq is its stable LearningId (re-mined content
//!                       dedupes byte-identically, changed content appends
//!                       a newer row that wins on reopen)
//! get/by_pattern     -> in-memory index built from the decoded rows
//! page_for_scope     -> live learnings of the scope, newest (highest
//!                       first-seq) first — the trait's ordering contract
//! remove_invalidated -> append a `removed` tombstone and drop the live row
//! len/len_for_scope  -> live corpus counts (bounded by `capacity`)
//! episodes           -> append `episode` rows; pending failures drive the
//!                       recovery miner and are consumed by tombstones
//! ```
//!
//! The adapter hook is exactly this trait: `LearningService` is generic
//! over it, so the daemon wires the durable implementation without any
//! other crate knowing about learning persistence. Mutation methods return
//! [`LearningError`] so adapter I/O failures and corrupt rows surface
//! loudly instead of being swallowed.

use std::collections::{BTreeMap, HashMap};

use faktor_core::hash::FileHash;

use crate::episode::{FailureEpisode, ProjectScope};
use crate::miner::{InvalidationContext, ProjectLearning};
use crate::{learning_id, LearningError};

learning_id!(
    /// Identifies one stored learning. Monotonic insertion order inside
    /// [`MemoryLearningStore`] defines "newest first"; a durable adapter
    /// must preserve that ordering contract with its row order.
    LearningId
);

/// Default bound of the in-process store.
pub const DEFAULT_MEMORY_CAPACITY: usize = 4096;

/// The persistence seam for project learnings.
///
/// Contracts:
///
/// - `upsert` dedupes by `(project scope, pattern digest)`: re-storing the
///   same pattern replaces the row and returns the SAME [`LearningId`];
///   `page_for_scope` returns live learnings newest-first;
/// - pages are bounded by `limit`, and `offset` is a plain newest-first
///   skip, so rendering never loads the corpus;
/// - every method is scoped by [`ProjectScope`]; there is no cross-project
///   read or write path;
/// - `remove_invalidated` evaluates [`ProjectLearning::is_invalidated`] in
///   the given scope only.
pub trait LearningStore {
    /// Insert or replace one learning by pattern digest.
    fn upsert(&mut self, learning: ProjectLearning) -> Result<LearningId, LearningError>;

    /// One learning by id, scoped to its project.
    fn get(&self, scope: &ProjectScope, id: LearningId) -> Option<&ProjectLearning>;

    /// One learning by pattern digest, scoped to its project.
    fn by_pattern(
        &self,
        scope: &ProjectScope,
        pattern_digest: FileHash,
    ) -> Option<&ProjectLearning>;

    /// Newest-first page of one project's learnings.
    fn page_for_scope(
        &self,
        scope: &ProjectScope,
        offset: usize,
        limit: usize,
    ) -> Vec<&ProjectLearning>;

    /// Remove every learning in `scope` invalidated by `context`.
    fn remove_invalidated(
        &mut self,
        scope: &ProjectScope,
        context: &InvalidationContext,
    ) -> Result<usize, LearningError>;

    /// Total stored learnings across all scopes.
    fn len(&self) -> usize;

    /// True when no learning is stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Stored learnings in one scope.
    fn len_for_scope(&self, scope: &ProjectScope) -> usize;

    /// Every stored learning, across scopes, for the corpus-level failure
    /// prior (`LearningService::omission_risk_index`). Order is
    /// unspecified. Bounded by the store's own capacity — a store is
    /// required to return at most its configured bound, never to walk an
    /// unbounded corpus. The default is EMPTY: an adapter that cannot
    /// enumerate cheaply stays neutral (every omission risk `1.0`), which
    /// fails closed to the prior-less byte parity instead of a wrong risk.
    fn all(&self) -> Vec<&ProjectLearning> {
        Vec::new()
    }
}

#[derive(Debug, Default)]
struct ScopePages {
    pages: BTreeMap<LearningId, ProjectLearning>,
    index: HashMap<FileHash, LearningId>,
}

/// Bounded in-process store. Insertion order is recency; when full the
/// globally oldest learning is evicted first, so memory stays under
/// `capacity` entries no matter how many patterns are ingested.
#[derive(Debug)]
pub struct MemoryLearningStore {
    capacity: usize,
    next_id: u64,
    count: usize,
    scopes: HashMap<ProjectScope, ScopePages>,
}

impl MemoryLearningStore {
    /// Store bounded to [`DEFAULT_MEMORY_CAPACITY`].
    pub fn new() -> Self {
        Self::bounded(DEFAULT_MEMORY_CAPACITY)
    }

    /// Store bounded to `capacity` entries (floored at 1).
    pub fn bounded(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            next_id: 1,
            count: 0,
            scopes: HashMap::new(),
        }
    }

    /// Configured entry bound.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn evict_oldest(&mut self) -> bool {
        let mut victim: Option<(LearningId, ProjectScope)> = None;
        for (scope, pages) in &self.scopes {
            if let Some((id, _)) = pages.pages.iter().next() {
                let better = match &victim {
                    None => true,
                    Some((best, _)) => id < best,
                };
                if better {
                    victim = Some((*id, scope.clone()));
                }
            }
        }
        let Some((id, scope)) = victim else {
            return false;
        };
        if let Some(pages) = self.scopes.get_mut(&scope) {
            if let Some(removed) = pages.pages.remove(&id) {
                pages.index.remove(&removed.pattern_digest());
            }
            if pages.pages.is_empty() {
                self.scopes.remove(&scope);
            }
        }
        self.count = self.count.saturating_sub(1);
        true
    }
}

impl Default for MemoryLearningStore {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningStore for MemoryLearningStore {
    fn upsert(&mut self, learning: ProjectLearning) -> Result<LearningId, LearningError> {
        let scope = learning.pattern.project.clone();
        let digest = learning.pattern_digest();
        if let Some(id) = self
            .scopes
            .get(&scope)
            .and_then(|pages| pages.index.get(&digest))
            .copied()
        {
            if let Some(pages) = self.scopes.get_mut(&scope) {
                if let Some(slot) = pages.pages.get_mut(&id) {
                    *slot = learning;
                }
            }
            return Ok(id);
        }
        if self.count >= self.capacity {
            self.evict_oldest();
        }
        let id = LearningId::new(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        let pages = self.scopes.entry(scope).or_default();
        pages.index.insert(digest, id);
        let replaced = pages.pages.insert(id, learning).is_some();
        if !replaced {
            self.count = self.count.saturating_add(1);
        }
        Ok(id)
    }

    fn get(&self, scope: &ProjectScope, id: LearningId) -> Option<&ProjectLearning> {
        self.scopes.get(scope)?.pages.get(&id)
    }

    fn by_pattern(
        &self,
        scope: &ProjectScope,
        pattern_digest: FileHash,
    ) -> Option<&ProjectLearning> {
        let pages = self.scopes.get(scope)?;
        let id = pages.index.get(&pattern_digest)?;
        pages.pages.get(id)
    }

    fn page_for_scope(
        &self,
        scope: &ProjectScope,
        offset: usize,
        limit: usize,
    ) -> Vec<&ProjectLearning> {
        let Some(pages) = self.scopes.get(scope) else {
            return Vec::new();
        };
        pages
            .pages
            .values()
            .rev()
            .skip(offset)
            .take(limit)
            .collect()
    }

    fn remove_invalidated(
        &mut self,
        scope: &ProjectScope,
        context: &InvalidationContext,
    ) -> Result<usize, LearningError> {
        let Some(pages) = self.scopes.get_mut(scope) else {
            return Ok(0);
        };
        let victims: Vec<LearningId> = pages
            .pages
            .iter()
            .filter(|(_, learning)| learning.is_invalidated(context))
            .map(|(id, _)| *id)
            .collect();
        let removed = victims.len();
        for id in victims {
            if let Some(learning) = pages.pages.remove(&id) {
                pages.index.remove(&learning.pattern_digest());
            }
        }
        let empty = pages.pages.is_empty();
        if empty {
            self.scopes.remove(scope);
        }
        self.count = self.count.saturating_sub(removed);
        Ok(removed)
    }

    fn len(&self) -> usize {
        self.count
    }

    fn len_for_scope(&self, scope: &ProjectScope) -> usize {
        self.scopes.get(scope).map_or(0, |pages| pages.pages.len())
    }

    fn all(&self) -> Vec<&ProjectLearning> {
        let mut out: Vec<&ProjectLearning> = Vec::with_capacity(self.count);
        for pages in self.scopes.values() {
            out.extend(pages.pages.values());
        }
        // Deterministic order regardless of hash-map iteration order; the
        // digest is the store's stable identity for a learning.
        out.sort_by_cached_key(|learning| learning.pattern_digest().to_hex());
        out
    }
}

// ------------------------------------------------------------ durable adapter

/// Bound of the in-memory episode window of [`SessionLearningStore`]:
/// unverified episodes beyond this are dropped from the cache while their
/// durable rows stay in the ledger (a reopen re-reads the newest rows).
pub const MAX_SESSION_EPISODES: usize = 256;

/// The corpus key of one learning: project scope + pattern digest. Scoping
/// is part of the identity, so cross-project matches stay impossible.
type PatternKey = (ProjectScope, FileHash);

/// One removal tombstone appended by the durable adapter:
///
/// - `Learning`: an invalidated `(scope, pattern)` stays removed;
/// - `Episode`: a pending failure episode is CONSUMED by a verified
///   recovery and can never be paired with a second, later success.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "tombstone", rename_all = "snake_case")]
enum DurableTombstone {
    Learning {
        scope: ProjectScope,
        pattern_digest: FileHash,
    },
    Episode {
        /// The consumed episode's id (diagnostics; the recovered episode
        /// reuses it, so consumption is keyed by dedupe digest).
        episode_id: u64,
        dedupe_digest: FileHash,
    },
}

/// Durable [`LearningStore`] over the session's typed ledger rows (audits
/// 65-67/92): failure episodes, stored learnings and removal tombstones are
/// `learning_record` entries appended through
/// [`faktor_session::SessionHandle::ledger_learning_record`] and read back
/// with `ledger_learning_records`.
///
/// Contracts (additive; no schema owned here and no migration):
///
/// - `open` re-reads the session's whole learning corpus and decodes every
///   row STRICTLY: a corrupt payload is a loud [`LearningError::Store`],
///   never a silent drop;
/// - `upsert` keeps the pattern's FIRST durable seq as its [`LearningId`],
///   so ids are stable across restarts and idempotent re-mining of
///   byte-identical content appends no duplicate row;
/// - the live corpus is bounded by `capacity`: the oldest id is evicted
///   from memory first (the durable row remains and a reopen deterministically
///   evicts the same way), exactly like [`MemoryLearningStore`];
/// - episodes are recorded from durable verification records: an unverified
///   (failed-attempt) episode is invisible to the miner ("no verified
///   success" never mints a learning) and the recovered copy carries the
///   same failure identity plus the recovery chain and the durable
///   verification record id.
#[derive(Debug)]
pub struct SessionLearningStore {
    handle: faktor_session::SessionHandle,
    capacity: usize,
    pages: BTreeMap<LearningId, ProjectLearning>,
    index: HashMap<PatternKey, LearningId>,
    /// First durable seq per pattern: the stable [`LearningId`] of a
    /// pattern, preserved across upserts and reopens.
    first_ids: HashMap<PatternKey, LearningId>,
    /// Durable episodes `(seq, episode)`, oldest first.
    episodes: Vec<(i64, FailureEpisode)>,
    /// Dedupe digests of episodes consumed by a verified recovery (durable
    /// tombstones); consumed failures are never pending again.
    consumed_episodes: std::collections::HashSet<FileHash>,
}

impl SessionLearningStore {
    /// Open the durable learning corpus of `handle`'s session, bounded to
    /// `capacity` live learnings (floored at 1). Every durable row is
    /// strictly decoded; the first corrupt row refuses the open loudly.
    pub fn open(
        handle: faktor_session::SessionHandle,
        capacity: usize,
    ) -> Result<Self, LearningError> {
        let rows = handle
            .ledger_learning_records()
            .map_err(|e| LearningError::Store(format!("learning ledger read failed: {e}")))?;
        let mut store = Self {
            handle,
            capacity: capacity.max(1),
            pages: BTreeMap::new(),
            index: HashMap::new(),
            first_ids: HashMap::new(),
            episodes: Vec::new(),
            consumed_episodes: std::collections::HashSet::new(),
        };
        for row in rows {
            let seq_id = u64::try_from(row.seq)
                .ok()
                .filter(|seq| *seq != 0)
                .map(LearningId::new)
                .ok_or_else(|| {
                    LearningError::Store(format!(
                        "learning ledger row has a non-positive seq {}",
                        row.seq
                    ))
                })?;
            match row.record.as_str() {
                faktor_session::LEARNING_RECORD_LEARNING => {
                    let learning: ProjectLearning =
                        serde_json::from_str(&row.payload).map_err(|e| {
                            LearningError::Store(format!(
                                "learning ledger row {} is corrupt (undecodable learning): {e}",
                                row.seq
                            ))
                        })?;
                    store.adopt(learning, seq_id);
                }
                faktor_session::LEARNING_RECORD_EPISODE => {
                    let episode: FailureEpisode =
                        serde_json::from_str(&row.payload).map_err(|e| {
                            LearningError::Store(format!(
                                "learning ledger row {} is corrupt (undecodable episode): {e}",
                                row.seq
                            ))
                        })?;
                    store.episodes.push((row.seq, episode));
                }
                faktor_session::LEARNING_RECORD_REMOVED => {
                    let tombstone: DurableTombstone =
                        serde_json::from_str(&row.payload).map_err(|e| {
                            LearningError::Store(format!(
                                "learning ledger row {} is corrupt (undecodable removal): {e}",
                                row.seq
                            ))
                        })?;
                    match tombstone {
                        DurableTombstone::Learning {
                            scope,
                            pattern_digest,
                        } => store.remove_pattern(&scope, pattern_digest),
                        DurableTombstone::Episode {
                            episode_id: _,
                            dedupe_digest,
                        } => store.consume_episode(dedupe_digest),
                    }
                }
                other => {
                    return Err(LearningError::Store(format!(
                        "learning ledger row {} has unknown record kind {other:?}",
                        row.seq
                    )))
                }
            }
        }
        store.evict_over_capacity();
        store.trim_episodes();
        Ok(store)
    }

    /// The durable session handle this store appends through.
    pub fn handle(&self) -> &faktor_session::SessionHandle {
        &self.handle
    }

    /// Configured live-corpus bound.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Record one failure episode durably. A byte-identical replay (same
    /// [`FailureEpisode::dedupe_digest`]) is a no-op. No learning is minted
    /// here: an unverified, recovery-less episode is invisible to the miner,
    /// which is exactly the "failed alone must not learn" contract.
    pub fn record_episode(&mut self, episode: &FailureEpisode) -> Result<(), LearningError> {
        let dedupe = episode.dedupe_digest();
        if self.consumed_episodes.contains(&dedupe) {
            return Ok(());
        }
        if self
            .episodes
            .iter()
            .any(|(_, e)| e.dedupe_digest() == dedupe)
        {
            return Ok(());
        }
        let payload = serde_json::to_string(episode)
            .map_err(|e| LearningError::Store(format!("episode serialization: {e}")))?;
        let seq = self.append_record(faktor_session::LEARNING_RECORD_EPISODE, &payload)?;
        self.episodes.push((seq, episode.clone()));
        self.trim_episodes();
        Ok(())
    }

    /// Consume every pending (unverified) episode with durable tombstones:
    /// the confirmed recovery that mined a learning is the ONE success those
    /// failures pair with, and a later verified completion must never
    /// re-pair them. Returns how many episodes were consumed.
    pub fn consume_pending_episodes(&mut self) -> Result<usize, LearningError> {
        let pending: Vec<(i64, u64, FileHash)> = self
            .episodes
            .iter()
            .filter(|(_, episode)| !episode.is_verified_success())
            .map(|(seq, episode)| (*seq, episode.id.raw(), episode.dedupe_digest()))
            .collect();
        let mut consumed = 0usize;
        for (seq, episode_id, dedupe_digest) in pending {
            let payload = serde_json::to_string(&DurableTombstone::Episode {
                episode_id,
                dedupe_digest,
            })
            .map_err(|e| LearningError::Store(format!("tombstone serialization: {e}")))?;
            self.append_record(faktor_session::LEARNING_RECORD_REMOVED, &payload)?;
            self.consumed_episodes.insert(dedupe_digest);
            self.episodes.retain(|(row_seq, _)| *row_seq != seq);
            consumed += 1;
        }
        Ok(consumed)
    }

    /// Every durable failure episode of this session, oldest first.
    pub fn episodes(&self) -> Vec<&FailureEpisode> {
        self.episodes.iter().map(|(_, episode)| episode).collect()
    }

    /// The unverified episodes (failed attempts with no recorded recovery
    /// yet), oldest first.
    pub fn pending_episodes(&self) -> Vec<&FailureEpisode> {
        self.episodes
            .iter()
            .filter(|(_, episode)| !episode.is_verified_success())
            .map(|(_, episode)| episode)
            .collect()
    }

    /// The newest unverified episode, when one exists.
    pub fn latest_pending(&self) -> Option<&FailureEpisode> {
        self.episodes
            .iter()
            .rev()
            .find(|(_, episode)| !episode.is_verified_success())
            .map(|(_, episode)| episode)
    }

    fn adopt(&mut self, learning: ProjectLearning, seq_id: LearningId) {
        let key = (learning.pattern.project.clone(), learning.pattern_digest());
        let stable = *self.first_ids.entry(key.clone()).or_insert(seq_id);
        self.index.insert(key, stable);
        self.pages.insert(stable, learning);
    }

    fn remove_pattern(&mut self, scope: &ProjectScope, digest: FileHash) {
        let key = (scope.clone(), digest);
        if let Some(id) = self.index.remove(&key) {
            self.pages.remove(&id);
        }
        self.first_ids.remove(&key);
    }

    fn consume_episode(&mut self, dedupe_digest: FileHash) {
        self.consumed_episodes.insert(dedupe_digest);
        self.episodes
            .retain(|(_, episode)| episode.dedupe_digest() != dedupe_digest);
    }

    fn evict_over_capacity(&mut self) {
        while self.pages.len() > self.capacity {
            let Some((&victim, _)) = self.pages.iter().next() else {
                break;
            };
            let Some(learning) = self.pages.remove(&victim) else {
                break;
            };
            let key = (learning.pattern.project.clone(), learning.pattern_digest());
            self.index.remove(&key);
            self.first_ids.remove(&key);
        }
    }

    fn trim_episodes(&mut self) {
        let excess = self.episodes.len().saturating_sub(MAX_SESSION_EPISODES);
        if excess > 0 {
            self.episodes.drain(..excess);
        }
    }

    fn append_record(&self, record: &str, payload: &str) -> Result<i64, LearningError> {
        if payload.len() > faktor_session::MAX_LEARNING_RECORD_PAYLOAD {
            return Err(LearningError::Oversized {
                what: format!("{record} record payload"),
                max: faktor_session::MAX_LEARNING_RECORD_PAYLOAD,
                actual: payload.len(),
            });
        }
        self.handle
            .ledger_learning_record(record, payload)
            .map_err(|e| LearningError::Store(format!("learning ledger append failed: {e}")))?
            .ok_or_else(|| LearningError::Store("learning ledger append returned no seq".into()))
    }
}

impl LearningStore for SessionLearningStore {
    fn upsert(&mut self, learning: ProjectLearning) -> Result<LearningId, LearningError> {
        let key = (learning.pattern.project.clone(), learning.pattern_digest());
        let payload = serde_json::to_string(&learning)
            .map_err(|e| LearningError::Store(format!("learning serialization: {e}")))?;
        if let Some(id) = self.index.get(&key).copied() {
            let unchanged = self
                .pages
                .get(&id)
                .and_then(|existing| serde_json::to_string(existing).ok())
                .as_deref()
                == Some(payload.as_str());
            if unchanged {
                return Ok(id);
            }
        }
        let seq = self.append_record(faktor_session::LEARNING_RECORD_LEARNING, &payload)?;
        let seq_id = LearningId::new(u64::try_from(seq).map_err(|_| {
            LearningError::Store(format!("learning ledger row has invalid seq {seq}"))
        })?);
        let stable = match self.first_ids.get(&key).copied() {
            Some(id) => id,
            None => {
                self.first_ids.insert(key.clone(), seq_id);
                seq_id
            }
        };
        self.index.insert(key, stable);
        self.pages.insert(stable, learning);
        self.evict_over_capacity();
        Ok(stable)
    }

    fn get(&self, scope: &ProjectScope, id: LearningId) -> Option<&ProjectLearning> {
        self.pages.get(&id).filter(|l| &l.pattern.project == scope)
    }

    fn by_pattern(
        &self,
        scope: &ProjectScope,
        pattern_digest: FileHash,
    ) -> Option<&ProjectLearning> {
        let id = self.index.get(&(scope.clone(), pattern_digest))?;
        self.pages.get(id)
    }

    fn page_for_scope(
        &self,
        scope: &ProjectScope,
        offset: usize,
        limit: usize,
    ) -> Vec<&ProjectLearning> {
        self.pages
            .values()
            .rev()
            .filter(|l| &l.pattern.project == scope)
            .skip(offset)
            .take(limit)
            .collect()
    }

    fn remove_invalidated(
        &mut self,
        scope: &ProjectScope,
        context: &InvalidationContext,
    ) -> Result<usize, LearningError> {
        let victims: Vec<LearningId> = self
            .pages
            .iter()
            .filter(|(_, learning)| {
                &learning.pattern.project == scope && learning.is_invalidated(context)
            })
            .map(|(id, _)| *id)
            .collect();
        let mut removed = 0usize;
        for id in victims {
            let Some(learning) = self.pages.remove(&id) else {
                continue;
            };
            let digest = learning.pattern_digest();
            let tombstone = DurableTombstone::Learning {
                scope: learning.pattern.project.clone(),
                pattern_digest: digest,
            };
            let payload = serde_json::to_string(&tombstone)
                .map_err(|e| LearningError::Store(format!("removal serialization: {e}")))?;
            self.append_record(faktor_session::LEARNING_RECORD_REMOVED, &payload)?;
            let key = (learning.pattern.project, digest);
            self.index.remove(&key);
            self.first_ids.remove(&key);
            removed += 1;
        }
        Ok(removed)
    }

    fn len(&self) -> usize {
        self.pages.len()
    }

    fn len_for_scope(&self, scope: &ProjectScope) -> usize {
        self.pages
            .values()
            .filter(|learning| &learning.pattern.project == scope)
            .count()
    }

    fn all(&self) -> Vec<&ProjectLearning> {
        let mut out: Vec<&ProjectLearning> = self.pages.values().collect();
        out.sort_by_cached_key(|learning| learning.pattern_digest().to_hex());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::test_support::{action, failure, project};
    use crate::episode::{EpisodeId, TaskClass};
    use crate::miner::{InvalidationRule, LearningPattern, ProjectLearning, StructuredAdvice};
    use crate::service::{LearningService, OMISSION_RISK_NEUTRAL};

    fn learning(workspace: u64, key: &str, message: &str, summary: &str) -> ProjectLearning {
        let scope = project(workspace, key);
        let pattern = LearningPattern {
            environment: crate::episode::test_support::environment(scope.clone(), None)
                .pattern_digest(),
            project: scope,
            task_class: TaskClass::new("bugfix").unwrap(),
            attempted_action: action("edit", "src/lib.rs", None, "x"),
            failure: failure("test_failure", None, message),
            recovery_actions: vec![action("edit", "src/lib.rs", None, "y")],
        };
        ProjectLearning {
            pattern,
            advice: StructuredAdvice::data_only(summary.to_string(), Vec::new(), Vec::new()),
            sample_count: 1,
            confidence_ppm: 400_000,
            supporting_episodes: vec![EpisodeId::new(1)],
            invalidation: InvalidationRule::Never,
        }
    }

    #[test]
    fn upsert_dedupes_by_pattern_and_pages_newest_first() {
        let mut store = MemoryLearningStore::new();
        let first = learning(1, "alpha", "one", "old");
        let digest = first.pattern_digest();
        let first_id = store.upsert(first).unwrap();
        let second_id = store.upsert(learning(1, "alpha", "two", "new")).unwrap();
        assert_ne!(first_id, second_id);
        assert_eq!(store.len(), 2);

        let page: Vec<&ProjectLearning> = store.page_for_scope(&project(1, "alpha"), 0, 16);
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].advice.summary, "new", "newest first");
        assert_eq!(page[1].advice.summary, "old");

        // Re-upserting the same pattern replaces instead of duplicating and
        // keeps the original id (idempotent re-mining must not reorder).
        let replacement = learning(1, "alpha", "one", "replacement");
        assert_eq!(replacement.pattern_digest(), digest);
        assert_eq!(store.upsert(replacement).unwrap(), first_id);
        assert_eq!(store.len(), 2);
        assert_eq!(
            store
                .by_pattern(&project(1, "alpha"), digest)
                .unwrap()
                .advice
                .summary,
            "replacement"
        );
    }

    #[test]
    fn store_is_bounded_and_evicts_oldest() {
        let mut store = MemoryLearningStore::bounded(2);
        store.upsert(learning(1, "alpha", "one", "1")).unwrap();
        store.upsert(learning(1, "alpha", "two", "2")).unwrap();
        store.upsert(learning(1, "alpha", "three", "3")).unwrap();
        assert_eq!(store.len(), 2);
        assert!(
            store
                .by_pattern(
                    &project(1, "alpha"),
                    learning(1, "alpha", "one", "1").pattern_digest()
                )
                .is_none(),
            "oldest evicted"
        );
        assert_eq!(store.capacity(), 2);

        // Capacity floors at 1: an adversarial zero-capacity store cannot
        // become unbounded.
        let mut tiny = MemoryLearningStore::bounded(0);
        tiny.upsert(learning(1, "alpha", "one", "1")).unwrap();
        tiny.upsert(learning(1, "alpha", "two", "2")).unwrap();
        assert_eq!(tiny.len(), 1);
    }

    #[test]
    fn scoping_never_crosses_projects_even_with_identical_message() {
        let mut store = MemoryLearningStore::new();
        let alpha = learning(1, "alpha", "same", "alpha advice");
        let digest = alpha.pattern_digest();
        store.upsert(alpha).unwrap();
        store
            .upsert(learning(2, "alpha", "same", "other workspace"))
            .unwrap();
        store
            .upsert(learning(1, "beta", "same", "other project"))
            .unwrap();

        assert_eq!(store.len(), 3);
        let alpha_page = store.page_for_scope(&project(1, "alpha"), 0, 16);
        assert_eq!(alpha_page.len(), 1);
        assert_eq!(alpha_page[0].advice.summary, "alpha advice");
        // Querying another project with alpha's digest must miss.
        assert!(store.by_pattern(&project(2, "alpha"), digest).is_none());
        assert!(store.by_pattern(&project(1, "beta"), digest).is_none());
        assert_eq!(store.len_for_scope(&project(1, "beta")), 1);
        assert_eq!(store.len_for_scope(&project(9, "missing")), 0);
    }

    #[test]
    fn remove_invalidated_is_scoped_and_rule_driven() {
        let mut store = MemoryLearningStore::new();
        let mut task_end = learning(1, "alpha", "one", "task end");
        task_end.invalidation = InvalidationRule::TaskEnd;
        task_end.pattern.failure = failure("test_failure", None, "task-end-pattern");
        store.upsert(task_end).unwrap();
        let mut other_scope = learning(2, "alpha", "one", "other scope");
        other_scope.invalidation = InvalidationRule::TaskEnd;
        other_scope.pattern.failure = failure("test_failure", None, "other-scope-pattern");
        store.upsert(other_scope).unwrap();

        let ctx = InvalidationContext::task_ended();
        assert_eq!(
            store
                .remove_invalidated(&project(1, "alpha"), &ctx)
                .unwrap(),
            1
        );
        assert_eq!(store.len(), 1, "other project untouched");
        assert_eq!(store.len_for_scope(&project(2, "alpha")), 1);
        assert_eq!(store.len_for_scope(&project(1, "alpha")), 0);
        assert_eq!(
            store
                .remove_invalidated(&project(1, "alpha"), &ctx)
                .unwrap(),
            0
        );
    }

    // ------------------------------------------------ durable session adapter

    fn durable_session() -> (
        std::sync::Arc<faktor_session::SessionManager>,
        faktor_session::SessionHandle,
        tempfile::TempDir,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let manager = faktor_session::SessionManager::open(
            dir.path().join("store"),
            dir.path().join("cas"),
            true,
        )
        .unwrap();
        let ws = manager.create_workspace("/workspace").unwrap();
        let handle = manager.create_session(ws, "learning", "fake", "m").unwrap();
        (manager, handle, dir)
    }

    /// Durable round trip: upsert dedupes by pattern with a STABLE id,
    /// pages newest-first, and a reopen of the same session re-reads the
    /// corpus (ids included) from the typed ledger rows.
    #[test]
    fn session_store_persists_learnings_and_reuses_ids_across_reopen() {
        let (_manager, handle, _dir) = durable_session();
        let scope = project(1, "alpha");
        let mut store =
            SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        let first = learning(1, "alpha", "one", "older");
        let digest = first.pattern_digest();
        let first_id = store.upsert(first).unwrap();
        let second = learning(1, "alpha", "two", "newer");
        let second_id = store.upsert(second).unwrap();
        assert_ne!(first_id, second_id);

        // Idempotent replacement keeps the same id and appends no duplicate.
        let mut replaced = learning(1, "alpha", "one", "replaced");
        replaced.advice = StructuredAdvice::data_only("replaced".into(), Vec::new(), Vec::new());
        assert_eq!(replaced.pattern_digest(), digest);
        assert_eq!(store.upsert(replaced).unwrap(), first_id);
        assert_eq!(store.len(), 2);
        let page = store.page_for_scope(&scope, 0, 16);
        assert_eq!(page.len(), 2);
        assert_eq!(page[0].advice.summary, "newer", "newest-first by first seq");

        // Reopen: everything (including the stable ids) is re-read from the
        // durable learning rows.
        let reopened = SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        assert_eq!(reopened.len(), 2);
        assert_eq!(
            reopened.get(&scope, first_id).unwrap().advice.summary,
            "replaced"
        );
        assert_eq!(
            reopened.by_pattern(&scope, digest).unwrap().advice.summary,
            "replaced"
        );
        assert!(reopened.by_pattern(&project(2, "alpha"), digest).is_none());
        assert_eq!(
            reopened.page_for_scope(&scope, 0, 16)[0]
                .pattern_digest()
                .to_hex(),
            reopened.page_for_scope(&scope, 0, 16)[0]
                .pattern_digest()
                .to_hex(),
            "deterministic reopen"
        );
        assert_eq!(reopened.all().len(), 2);
    }

    /// Corpus bound: a capacity-N durable store evicts the oldest id from
    /// memory (the durable row stays), and a reopen evicts identically.
    #[test]
    fn session_store_is_capacity_bounded_and_a_reopen_matches() {
        let (_manager, handle, _dir) = durable_session();
        let mut store = SessionLearningStore::open(handle.clone(), 2).unwrap();
        let oldest = learning(1, "alpha", "one", "1");
        let oldest_digest = oldest.pattern_digest();
        store.upsert(oldest).unwrap();
        store.upsert(learning(1, "alpha", "two", "2")).unwrap();
        store.upsert(learning(1, "alpha", "three", "3")).unwrap();
        assert_eq!(store.len(), 2);
        assert!(store
            .by_pattern(&project(1, "alpha"), oldest_digest)
            .is_none());
        let reopened = SessionLearningStore::open(handle.clone(), 2).unwrap();
        assert_eq!(reopened.len(), 2);
        assert!(reopened
            .by_pattern(&project(1, "alpha"), oldest_digest)
            .is_none());
    }

    /// Oversized payloads are refused loudly (typed `Oversized`) BEFORE any
    /// row is journaled — the durable adapter never writes a partial corpus.
    #[test]
    fn session_store_refuses_oversized_payloads_without_journaling() {
        let (_manager, handle, _dir) = durable_session();
        let mut store =
            SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        let huge = "x".repeat(faktor_session::MAX_LEARNING_RECORD_PAYLOAD + 1);
        let mut oversized = learning(1, "alpha", "one", "tiny");
        oversized.advice = StructuredAdvice::data_only(huge, Vec::new(), Vec::new());
        let err = store.upsert(oversized).unwrap_err();
        assert!(matches!(err, LearningError::Oversized { .. }), "{err}");
        assert_eq!(store.len(), 0);
        let reopened = SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        assert_eq!(reopened.len(), 0, "nothing was journaled");
    }

    /// A corrupt durable row is LOUD: both a semantically invalid learning
    /// payload and a structurally unknown record kind refuse the open.
    #[test]
    fn session_store_fails_loud_on_corrupt_ledger_rows() {
        use faktor_session::ledger::LEDGER_ENTRY_SCHEMA_V;
        use faktor_session::LEARNING_RECORD_LEARNING;

        let (_manager, handle, _dir) = durable_session();
        handle
            .ledger_learning_record(LEARNING_RECORD_LEARNING, "not json")
            .unwrap();
        let err = SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap_err();
        assert!(
            matches!(err, LearningError::Store(ref m) if m.contains("corrupt")),
            "{err}"
        );

        // Structurally hostile row written through the raw store (unknown
        // record kind): the session decode refuses it loudly too.
        let (other_manager, other_handle, _other_dir) = durable_session();
        other_manager
            .store()
            .append_ledger_entry(
                other_handle.id(),
                faktor_session::ENTRY_LEARNING_RECORD,
                LEDGER_ENTRY_SCHEMA_V,
                serde_json::json!({
                    "kind": "learning_record",
                    "record": "bogus",
                    "payload": "{}",
                }),
            )
            .unwrap();
        assert!(SessionLearningStore::open(other_handle.clone(), DEFAULT_MEMORY_CAPACITY).is_err());
    }

    /// The failed-alone contract end-to-end: an unverified episode mines no
    /// learning (and is skipped by the miner even if handed in); once the
    /// recovery chain + durable verification record are attached the mining
    /// persists a learning, the corpus index keys BOTH digests, and the
    /// whole corpus survives a reopen.
    #[test]
    fn unverified_episode_never_mines_but_a_recovered_one_does_and_survives_reopen() {
        use crate::episode::test_support::{action, episode};
        use faktor_core::id::VerificationRecordId;

        let (_manager, handle, _dir) = durable_session();
        let scope = project(1, "alpha");
        let unverified = episode(1, project(1, "alpha"), None);
        let mut store =
            SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        store.record_episode(&unverified).unwrap();
        assert_eq!(store.pending_episodes().len(), 1);
        assert_eq!(store.latest_pending().unwrap().id, unverified.id);
        // Replaying the same episode appends nothing.
        store.record_episode(&unverified).unwrap();
        assert_eq!(store.episodes().len(), 1);

        let service = LearningService::new(store);
        assert_eq!(
            service.len(),
            0,
            "an unverified episode alone mints no learning"
        );

        // The recovered episode reuses the durable FAILED identity and adds
        // the recovery chain plus the durable verification record id.
        let recovered = unverified
            .clone()
            .with_recovery_actions(vec![action("edit", "src/lib.rs", Some("parse"), "fix")])
            .unwrap()
            .verified(VerificationRecordId::new(7));
        let store = SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        assert_eq!(store.latest_pending().unwrap().id, unverified.id);
        let mut service = LearningService::new(store);
        let ids = service
            .mine_and_store(std::slice::from_ref(&recovered))
            .unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(service.len(), 1);
        let stored = service.page(&scope, 0, 1)[0].clone();
        let index = service.omission_risk_index();
        assert!(index.contains_key(&stored.pattern_digest()));
        assert!(index.contains_key(&stored.pattern.failure.digest()));
        assert!(index[&stored.pattern_digest()] > OMISSION_RISK_NEUTRAL);

        // Durability: a fresh adapter over the same session sees the corpus.
        drop(service);
        let mut reopened =
            SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        assert_eq!(reopened.len(), 1);
        assert!(reopened
            .by_pattern(&scope, stored.pattern_digest())
            .is_some());
        assert_eq!(
            reopened.pending_episodes().len(),
            1,
            "an un-consumed failure stays pending"
        );

        // Consuming the pending failure is durable: a later reopen never
        // pairs it with another verified success.
        assert_eq!(reopened.consume_pending_episodes().unwrap(), 1);
        assert!(reopened.pending_episodes().is_empty());
        reopened.record_episode(&unverified).unwrap();
        assert!(
            reopened.pending_episodes().is_empty(),
            "a consumed episode cannot be re-recorded"
        );
        let again = SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        assert!(again.pending_episodes().is_empty());
    }

    /// A removal tombstone is durable: an invalidated learning stays gone
    /// after a reopen, and the same pattern may later be re-learned with a
    /// fresh durable id.
    #[test]
    fn invalidated_learning_stays_removed_across_reopen() {
        let (_manager, handle, _dir) = durable_session();
        let scope = project(1, "alpha");
        let mut store =
            SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        let mut entry = learning(1, "alpha", "one", "task end");
        entry.invalidation = InvalidationRule::TaskEnd;
        let digest = entry.pattern_digest();
        store.upsert(entry).unwrap();
        assert_eq!(
            store
                .remove_invalidated(&scope, &InvalidationContext::task_ended())
                .unwrap(),
            1
        );
        assert_eq!(store.len(), 0);

        let reopened = SessionLearningStore::open(handle.clone(), DEFAULT_MEMORY_CAPACITY).unwrap();
        assert_eq!(reopened.len(), 0, "tombstone survives reopen");
        assert!(reopened.by_pattern(&scope, digest).is_none());

        let mut store = reopened;
        let again = learning(1, "alpha", "one", "task end");
        assert_eq!(again.pattern_digest(), digest);
        let new_id = store.upsert(again).unwrap();
        assert_eq!(store.len(), 1, "re-learning after removal is allowed");
        let _ = (scope, new_id);
    }
}
