//! The persistence seam (audit items 92/106): [`LearningStore`] is the only
//! way learnings are persisted, with [`MemoryLearningStore`] as the bounded
//! in-process implementation.
//!
//! **No schema is owned here and no migration is added.** A durable adapter
//! maps the trait onto the session ledger's learning rows once that table
//! exists:
//!
//! ```text
//! upsert             -> UPDATE/INSERT one ledger row keyed by
//!                       (workspace_id, project_key, pattern_digest)
//! get/by_pattern     -> SELECT one row
//! page_for_scope     -> SELECT ... WHERE workspace_id/project_key = scope
//!                       ORDER BY rowid DESC LIMIT :limit OFFSET :offset
//!                       (newest-first is a trait contract)
//! remove_invalidated -> DELETE rows whose stored invalidation rule the
//!                       context invalidates (evaluated in the adapter from
//!                       the row's rule columns)
//! len/len_for_scope  -> COUNT(*)
//! ```
//!
//! The adapter hook is exactly this trait: `LearningService` is generic
//! over it, so the daemon wires a ledger-backed implementation without any
//! other crate knowing about learning persistence. Mutation methods return
//! [`LearningError`] so adapter I/O failures surface instead of being
//! swallowed.

use std::collections::{BTreeMap, HashMap};

use faktor_core::hash::FileHash;

use crate::episode::ProjectScope;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episode::test_support::{action, failure, project};
    use crate::episode::{EpisodeId, TaskClass};
    use crate::miner::{InvalidationRule, LearningPattern, ProjectLearning, StructuredAdvice};

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
}
