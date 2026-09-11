//! Snapshot/pack cache with hard bounds and delta-scoped invalidation
//! (audit 48-54/58/59).
//!
//! Cache keys are exactly `{workspace, source revision, provider id,
//! provider version, schema version}` (plus a request digest for packs, so
//! two different queries never collide on one base revision). An unchanged
//! re-request is served from the cache without touching the provider;
//! applying a delta invalidates only the refs whose entity ids changed.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use faktor_core::{FileHash, WorkspaceId};

use crate::types::{
    BoxFuture, SemanticContextPack, SemanticContextRequest, SemanticDelta, SemanticEntityId,
    SemanticEnvelope, SemanticError, SemanticProvider, SemanticProviderId, SemanticSnapshot,
    SemanticSnapshotId, SemanticSnapshotRequest, SEMANTIC_SCHEMA_VERSION,
};

/// Immutable identity of a cached snapshot/pack backing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticCacheKey {
    pub workspace: WorkspaceId,
    pub source_revision: String,
    pub provider_id: SemanticProviderId,
    pub provider_version: u32,
    pub schema_version: u32,
}

impl SemanticCacheKey {
    pub fn new(
        workspace: WorkspaceId,
        source_revision: impl Into<String>,
        provider_id: SemanticProviderId,
        provider_version: u32,
        schema_version: u32,
    ) -> Self {
        Self {
            workspace,
            source_revision: source_revision.into(),
            provider_id,
            provider_version,
            schema_version,
        }
    }
}

/// What one cached entry is backed by: the key, the provider snapshot it
/// describes, and the entity ids it covers. Entity ids drive delta-scoped
/// invalidation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticSnapshotRef {
    pub key: SemanticCacheKey,
    pub snapshot_id: SemanticSnapshotId,
    pub entity_ids: BTreeSet<SemanticEntityId>,
}

impl SemanticSnapshotRef {
    pub fn new(
        key: SemanticCacheKey,
        snapshot_id: SemanticSnapshotId,
        entity_ids: impl IntoIterator<Item = SemanticEntityId>,
    ) -> Self {
        Self {
            key,
            snapshot_id,
            entity_ids: entity_ids.into_iter().collect(),
        }
    }
}

/// The cache slot identity. Snapshots key on the base key; packs add a
/// digest of the request so different queries cannot alias.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SemanticCacheEntryKey {
    Snapshot(SemanticCacheKey),
    Pack {
        base: SemanticCacheKey,
        request: FileHash,
    },
}

impl SemanticCacheEntryKey {
    pub fn base(&self) -> &SemanticCacheKey {
        match self {
            Self::Snapshot(key) => key,
            Self::Pack { base, .. } => base,
        }
    }
}

/// A cached provider object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticCached {
    Snapshot(SemanticEnvelope<SemanticSnapshot>),
    Context(SemanticEnvelope<SemanticContextPack>),
}

/// One bounded cache entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheEntry {
    pub key: SemanticCacheEntryKey,
    pub reference: SemanticSnapshotRef,
    pub value: SemanticCached,
    last_used: u64,
}

/// The report returned by delta-driven invalidation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SemanticInvalidation {
    pub retained: Vec<SemanticEntityId>,
    pub invalidated: Vec<SemanticEntityId>,
}

impl SemanticInvalidation {
    pub fn retained_count(&self) -> usize {
        self.retained.len()
    }

    pub fn invalidated_count(&self) -> usize {
        self.invalidated.len()
    }
}

/// Bounded LRU cache. `cap == 0` keeps nothing (the cache is a pass-through).
#[derive(Debug, Clone)]
pub struct SemanticCache {
    cap: usize,
    entries: Vec<CacheEntry>,
    tick: u64,
}

impl SemanticCache {
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            entries: Vec::new(),
            tick: 0,
        }
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains_key(&self, key: &SemanticCacheEntryKey) -> bool {
        self.entries.iter().any(|entry| &entry.key == key)
    }

    /// Read an entry and mark it most recently used.
    pub fn get(&mut self, key: &SemanticCacheEntryKey) -> Option<SemanticCached> {
        let position = self.entries.iter().position(|entry| &entry.key == key)?;
        self.tick = self.tick.wrapping_add(1);
        self.entries[position].last_used = self.tick;
        Some(self.entries[position].value.clone())
    }

    /// Insert; evicts the least recently used entry when over the cap.
    /// Returns the evicted entry (also when `cap == 0`, where the inserted
    /// entry itself is returned and nothing is stored).
    pub fn insert(
        &mut self,
        key: SemanticCacheEntryKey,
        reference: SemanticSnapshotRef,
        value: SemanticCached,
    ) -> Option<CacheEntry> {
        self.tick = self.tick.wrapping_add(1);
        if let Some(position) = self.entries.iter().position(|entry| entry.key == key) {
            self.entries.remove(position);
        }
        let entry = CacheEntry {
            key,
            reference,
            value,
            last_used: self.tick,
        };
        if self.cap == 0 {
            return Some(entry);
        }
        self.entries.push(entry);
        if self.entries.len() > self.cap {
            let position = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(position, _)| position)?;
            return Some(self.entries.remove(position));
        }
        None
    }

    /// Remove one key.
    pub fn invalidate(&mut self, key: &SemanticCacheEntryKey) -> Option<CacheEntry> {
        let position = self.entries.iter().position(|entry| &entry.key == key)?;
        Some(self.entries.remove(position))
    }

    /// Invalidate every entry for one workspace revision (key change).
    pub fn invalidate_source_revision(
        &mut self,
        workspace: WorkspaceId,
        source_revision: &str,
    ) -> usize {
        let before = self.entries.len();
        self.entries.retain(|entry| {
            let base = entry.key.base();
            !(base.workspace == workspace && base.source_revision == source_revision)
        });
        before - self.entries.len()
    }

    /// Apply a delta: an entry is dropped only when one of the entity ids it
    /// covers changed. Entries for other workspaces and entries covering
    /// untouched ids survive.
    pub fn apply_delta(&mut self, delta: &SemanticDelta) -> SemanticInvalidation {
        let changed = delta.changed_entity_ids();
        let mut report = SemanticInvalidation::default();
        let mut remove = Vec::new();
        for entry in &self.entries {
            if entry.reference.key.workspace != delta.workspace {
                continue;
            }
            let mut poisoned = false;
            for id in &entry.reference.entity_ids {
                if changed.contains(id) {
                    report.invalidated.push(id.clone());
                    poisoned = true;
                } else {
                    report.retained.push(id.clone());
                }
            }
            if poisoned {
                remove.push(entry.key.clone());
            }
        }
        for key in remove {
            self.invalidate(&key);
        }
        report.retained.sort();
        report.invalidated.sort();
        report
    }
}

/// A provider wrapper that serves unchanged snapshot/context re-requests
/// from a bounded cache without calling the inner provider.
pub struct SemanticCachedProvider<P> {
    inner: P,
    cache: Mutex<SemanticCache>,
    calls: AtomicUsize,
}

impl<P: SemanticProvider> SemanticCachedProvider<P> {
    pub fn new(inner: P, cap: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(SemanticCache::new(cap)),
            calls: AtomicUsize::new(0),
        }
    }

    /// Number of actual inner provider invocations (cache hits excluded).
    pub fn provider_calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub fn inner(&self) -> &P {
        &self.inner
    }

    fn base_key(&self, workspace: WorkspaceId, source_revision: &str) -> SemanticCacheKey {
        SemanticCacheKey::new(
            workspace,
            source_revision,
            self.inner.id(),
            self.inner.version(),
            SEMANTIC_SCHEMA_VERSION,
        )
    }

    fn snapshot_key(&self, request: &SemanticSnapshotRequest) -> SemanticCacheEntryKey {
        SemanticCacheEntryKey::Snapshot(self.base_key(request.workspace, &request.source_revision))
    }

    fn context_key(&self, request: &SemanticContextRequest) -> SemanticCacheEntryKey {
        SemanticCacheEntryKey::Pack {
            base: self.base_key(request.workspace, &request.source_revision),
            request: context_request_digest(request),
        }
    }
}

fn context_request_digest(request: &SemanticContextRequest) -> FileHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&request.workspace.raw().to_le_bytes());
    hasher.update(&request.snapshot_id.as_file_hash().bytes());
    hasher.update(&(request.query.len() as u64).to_le_bytes());
    hasher.update(request.query.as_bytes());
    hasher.update(&(request.max_items as u64).to_le_bytes());
    hasher.update(&(request.max_bytes as u64).to_le_bytes());
    FileHash::from(*hasher.finalize().as_bytes())
}

impl<P: SemanticProvider> SemanticProvider for SemanticCachedProvider<P> {
    fn id(&self) -> SemanticProviderId {
        self.inner.id()
    }

    fn version(&self) -> u32 {
        self.inner.version()
    }

    fn capabilities(&self) -> crate::types::SemanticCapabilities {
        self.inner.capabilities()
    }

    fn descriptor(&self) -> crate::types::SemanticProviderDescriptor {
        self.inner.descriptor()
    }

    fn validated_descriptor(&self) -> Option<crate::types::SemanticProviderDescriptor> {
        self.inner.validated_descriptor()
    }

    fn transport_identity(&self) -> String {
        self.inner.transport_identity()
    }

    fn handshake(
        &self,
        cancel: faktor_core::CancellationToken,
    ) -> BoxFuture<'_, Result<crate::types::SemanticProviderDescriptor, SemanticError>> {
        self.inner.handshake(cancel)
    }

    fn snapshot(
        &self,
        request: SemanticSnapshotRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticSnapshot>, SemanticError>> {
        let key = self.snapshot_key(&request);
        if let Some(SemanticCached::Snapshot(envelope)) = self.cache.lock().unwrap().get(&key) {
            return Box::pin(std::future::ready(Ok(envelope)));
        }
        Box::pin(async move {
            let result = self.inner.snapshot(request).await;
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Ok(envelope) = &result {
                let reference = SemanticSnapshotRef::new(
                    self.base_key(envelope.workspace, &envelope.payload.source_revision),
                    envelope.snapshot_id,
                    std::iter::empty(),
                );
                self.cache.lock().unwrap().insert(
                    key,
                    reference,
                    SemanticCached::Snapshot(envelope.clone()),
                );
            }
            result
        })
    }

    fn context(
        &self,
        request: SemanticContextRequest,
    ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticContextPack>, SemanticError>> {
        let key = self.context_key(&request);
        if let Some(SemanticCached::Context(envelope)) = self.cache.lock().unwrap().get(&key) {
            return Box::pin(std::future::ready(Ok(envelope)));
        }
        let base = key.base().clone();
        Box::pin(async move {
            let result = self.inner.context(request).await;
            self.calls.fetch_add(1, Ordering::SeqCst);
            if let Ok(envelope) = &result {
                let entity_ids = envelope
                    .payload
                    .items
                    .iter()
                    .map(|item| item.entity.entity_id.clone());
                let reference = SemanticSnapshotRef::new(base, envelope.snapshot_id, entity_ids);
                self.cache.lock().unwrap().insert(
                    key,
                    reference,
                    SemanticCached::Context(envelope.clone()),
                );
            }
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{block_on, call, entity, provider_id, snapshot};
    use crate::types::{
        SemanticCapabilities, SemanticContextItem, SemanticDeltaChange, SemanticDeltaKind,
        SemanticEntityRef,
    };
    use faktor_core::WorkspaceId;

    struct SpyProvider {
        id: SemanticProviderId,
        calls: AtomicUsize,
    }

    impl SpyProvider {
        fn new() -> Self {
            Self {
                id: provider_id("spy"),
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl SemanticProvider for SpyProvider {
        fn id(&self) -> SemanticProviderId {
            self.id.clone()
        }

        fn version(&self) -> u32 {
            1
        }

        fn capabilities(&self) -> SemanticCapabilities {
            SemanticCapabilities::CONTEXT.union(SemanticCapabilities::SNAPSHOT)
        }

        fn snapshot(
            &self,
            request: SemanticSnapshotRequest,
        ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticSnapshot>, SemanticError>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let tree_hash = FileHash::from([9u8; 32]);
            let envelope = SemanticEnvelope::new(
                self.id.clone(),
                1,
                request.workspace,
                snapshot(request.workspace, &request.source_revision),
                5,
                SemanticSnapshot {
                    source_revision: request.source_revision,
                    tree_hash,
                    entity_count: 3,
                    created_ms: 5,
                },
            );
            Box::pin(async move { Ok(envelope) })
        }

        fn context(
            &self,
            request: SemanticContextRequest,
        ) -> BoxFuture<'_, Result<SemanticEnvelope<SemanticContextPack>, SemanticError>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let envelope = SemanticEnvelope::new(
                self.id.clone(),
                1,
                request.workspace,
                request.snapshot_id,
                5,
                SemanticContextPack {
                    items: vec![SemanticContextItem {
                        entity: entity("src/lib.rs", &request.query),
                        relevance_bps: 5000,
                        excerpt: request.query.clone(),
                    }],
                    truncated: false,
                    total_bytes: request.query.len(),
                    degraded: false,
                },
            );
            Box::pin(async move { Ok(envelope) })
        }
    }

    fn context_request(query: &str) -> SemanticContextRequest {
        SemanticContextRequest {
            call: call(),
            workspace: WorkspaceId::new(1),
            source_revision: "rev-1".to_string(),
            snapshot_id: snapshot(WorkspaceId::new(1), "rev-1"),
            query: query.to_string(),
            max_items: 8,
            max_bytes: 4096,
        }
    }

    fn snapshot_request() -> SemanticSnapshotRequest {
        SemanticSnapshotRequest {
            call: call(),
            workspace: WorkspaceId::new(1),
            source_revision: "rev-1".to_string(),
        }
    }

    fn cached_pack(entities: &[&str]) -> SemanticCached {
        let items = entities
            .iter()
            .map(|id| SemanticContextItem {
                entity: entity("src/lib.rs", id),
                relevance_bps: 100,
                excerpt: "x".to_string(),
            })
            .collect();
        SemanticCached::Context(SemanticEnvelope::new(
            provider_id("spy"),
            1,
            WorkspaceId::new(1),
            snapshot(WorkspaceId::new(1), "rev-1"),
            0,
            SemanticContextPack {
                items,
                truncated: false,
                total_bytes: 0,
                degraded: false,
            },
        ))
    }

    fn pack_key(revision: &str, query: &str) -> SemanticCacheEntryKey {
        let request = SemanticContextRequest {
            call: call(),
            workspace: WorkspaceId::new(1),
            source_revision: revision.to_string(),
            snapshot_id: snapshot(WorkspaceId::new(1), revision),
            query: query.to_string(),
            max_items: 8,
            max_bytes: 4096,
        };
        SemanticCacheEntryKey::Pack {
            base: SemanticCacheKey::new(
                WorkspaceId::new(1),
                revision,
                provider_id("spy"),
                1,
                SEMANTIC_SCHEMA_VERSION,
            ),
            request: context_request_digest(&request),
        }
    }

    fn reference(revision: &str, entities: &[&str]) -> SemanticSnapshotRef {
        SemanticSnapshotRef::new(
            SemanticCacheKey::new(
                WorkspaceId::new(1),
                revision,
                provider_id("spy"),
                1,
                SEMANTIC_SCHEMA_VERSION,
            ),
            snapshot(WorkspaceId::new(1), revision),
            entities
                .iter()
                .map(|id| SemanticEntityId::parse(id).unwrap()),
        )
    }

    fn delta(workspace: WorkspaceId, changed: &[&str]) -> SemanticDelta {
        SemanticDelta {
            workspace,
            from_snapshot: snapshot(WorkspaceId::new(1), "rev-1"),
            to_snapshot: snapshot(WorkspaceId::new(1), "rev-2"),
            changes: changed
                .iter()
                .map(|id| SemanticDeltaChange {
                    entity: SemanticEntityRef::new(
                        workspace,
                        crate::types::WorkspacePath::parse("src/lib.rs").unwrap(),
                        SemanticEntityId::parse(id).unwrap(),
                    ),
                    kind: SemanticDeltaKind::Modified,
                    old_hash: None,
                    new_hash: None,
                })
                .collect(),
            degraded: false,
        }
    }

    #[test]
    fn unchanged_re_request_hits_cache_without_calling_the_provider() {
        let provider = SemanticCachedProvider::new(SpyProvider::new(), 16);
        assert_eq!(provider.provider_calls(), 0);

        let first = block_on(provider.context(context_request("alpha"))).unwrap();
        assert_eq!(provider.provider_calls(), 1);
        let second = block_on(provider.context(context_request("alpha"))).unwrap();
        assert_eq!(
            provider.provider_calls(),
            1,
            "an unchanged re-request must be served from cache"
        );
        assert_eq!(first, second);

        // A different query is a different key and does call the provider.
        block_on(provider.context(context_request("beta"))).unwrap();
        assert_eq!(provider.provider_calls(), 2);

        // Same for snapshots.
        block_on(provider.snapshot(snapshot_request())).unwrap();
        assert_eq!(provider.provider_calls(), 3);
        block_on(provider.snapshot(snapshot_request())).unwrap();
        assert_eq!(provider.provider_calls(), 3);
    }

    #[test]
    fn cache_is_bounded_and_evicts_least_recently_used() {
        let mut cache = SemanticCache::new(2);
        let k1 = pack_key("rev-1", "one");
        let k2 = pack_key("rev-1", "two");
        let k3 = pack_key("rev-1", "three");
        assert!(cache
            .insert(k1.clone(), reference("rev-1", &["a"]), cached_pack(&["a"]))
            .is_none());
        assert!(cache
            .insert(k2.clone(), reference("rev-1", &["b"]), cached_pack(&["b"]))
            .is_none());
        // Touch k1 so k2 becomes the least recently used.
        assert!(cache.get(&k1).is_some());
        let evicted = cache
            .insert(k3.clone(), reference("rev-1", &["c"]), cached_pack(&["c"]))
            .expect("cap 2 must evict");
        assert_eq!(evicted.key, k2);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&k1));
        assert!(!cache.contains_key(&k2));
        assert!(cache.contains_key(&k3));

        // Zero cap keeps nothing.
        let mut disabled = SemanticCache::new(0);
        assert!(disabled.is_empty());
        let evicted = disabled.insert(k1.clone(), reference("rev-1", &["a"]), cached_pack(&["a"]));
        assert!(evicted.is_some());
        assert!(disabled.is_empty());
    }

    #[test]
    fn key_change_invalidates_matching_entries_only() {
        let mut cache = SemanticCache::new(16);
        let rev1 = pack_key("rev-1", "query");
        let rev2 = pack_key("rev-2", "query");
        cache.insert(
            rev1.clone(),
            reference("rev-1", &["a"]),
            cached_pack(&["a"]),
        );
        cache.insert(
            rev2.clone(),
            reference("rev-2", &["b"]),
            cached_pack(&["b"]),
        );
        assert_eq!(
            cache.invalidate_source_revision(WorkspaceId::new(1), "rev-1"),
            1
        );
        assert!(!cache.contains_key(&rev1));
        assert!(cache.contains_key(&rev2));
        // Another workspace's identical revision string is untouched.
        assert_eq!(
            cache.invalidate_source_revision(WorkspaceId::new(9), "rev-2"),
            0
        );
        assert!(cache.contains_key(&rev2));
    }

    #[test]
    fn delta_invalidation_retains_99_of_100_and_drops_the_changed_one() {
        let mut cache = SemanticCache::new(1024);
        let mut keys = Vec::new();
        for index in 0..100u32 {
            let id = format!("entity-{index}");
            let key = pack_key("rev-1", &id);
            cache.insert(key.clone(), reference("rev-1", &[&id]), cached_pack(&[&id]));
            keys.push(key);
        }
        assert_eq!(cache.len(), 100);

        let changed_id = "entity-50";
        let report = cache.apply_delta(&delta(WorkspaceId::new(1), &[changed_id]));
        assert_eq!(report.invalidated_count(), 1);
        assert_eq!(report.retained_count(), 99);
        assert_eq!(report.invalidated[0].as_str(), changed_id);
        assert_eq!(cache.len(), 99);
        assert!(!cache.contains_key(&keys[50]));
        assert!(cache.contains_key(&keys[49]));
        assert!(cache.contains_key(&keys[51]));

        // A delta for another workspace invalidates nothing here.
        let report = cache.apply_delta(&delta(WorkspaceId::new(7), &["entity-51"]));
        assert_eq!(report.invalidated_count(), 0);
        assert_eq!(report.retained_count(), 0);
        assert_eq!(cache.len(), 99);
    }

    #[test]
    fn entry_with_multiple_entities_is_invalidated_atomically() {
        let mut cache = SemanticCache::new(8);
        let key = pack_key("rev-1", "multi");
        cache.insert(
            key.clone(),
            reference("rev-1", &["a", "b"]),
            cached_pack(&["a", "b"]),
        );
        let report = cache.apply_delta(&delta(WorkspaceId::new(1), &["a"]));
        assert_eq!(report.invalidated_count(), 1);
        assert_eq!(report.retained_count(), 1);
        assert!(!cache.contains_key(&key));
        assert!(cache.is_empty());
    }

    #[test]
    fn provider_version_and_schema_are_part_of_the_key() {
        let base = SemanticCacheKey::new(
            WorkspaceId::new(1),
            "rev-1",
            provider_id("spy"),
            1,
            SEMANTIC_SCHEMA_VERSION,
        );
        let bumped = SemanticCacheKey::new(
            WorkspaceId::new(1),
            "rev-1",
            provider_id("spy"),
            2,
            SEMANTIC_SCHEMA_VERSION,
        );
        assert_ne!(base, bumped);
        assert_ne!(
            SemanticCachedProvider::<SpyProvider>::new(SpyProvider::new(), 1)
                .context_key(&context_request("q")),
            SemanticCacheEntryKey::Snapshot(base)
        );
    }
}
