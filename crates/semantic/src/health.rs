//! Bounded per-provider health for capability-driven failover
//! (audit 48-54/58/59).
//!
//! A semantic provider is optional acceleration, so a broken or slow
//! provider must never tax every call: after a recoverable failure its
//! health key enters a bounded cooldown window, during which dispatch skips
//! it entirely (no process spawn, no HTTP attempt, no timeout paid) and the
//! next compatible provider serves the operation.
//!
//! Health is keyed by `(provider id, transport identity, operation class)`:
//! two endpoints behind one provider id, or one endpoint serving different
//! operation classes, never share failures or cooldowns. State is bounded:
//! the tracker keeps at most [`MAX_HEALTH_ENTRIES`] keys and at most
//! [`MAX_LATENCY_SAMPLES`] latency samples per key.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use faktor_core::Clock;

use crate::types::{SemanticOp, SemanticProviderId};

/// Maximum health keys retained; the least recently updated key is evicted
/// first, so an attacker-controlled provider set can never grow state
/// without bound.
pub const MAX_HEALTH_ENTRIES: usize = 64;
/// Maximum latency samples retained per key (most recent kept).
pub const MAX_LATENCY_SAMPLES: usize = 16;
/// Cooldown after the first failure (doubles per consecutive failure).
pub const BASE_COOLDOWN_MS: i64 = 500;
/// Upper bound on one cooldown window.
pub const MAX_COOLDOWN_MS: i64 = 30_000;
/// Maximum length of the transport identity component of a health key.
pub const MAX_TRANSPORT_IDENTITY_BYTES: usize = 256;

/// One provider's health key: identity + transport + operation class.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProviderHealthKey {
    pub provider: SemanticProviderId,
    pub transport: String,
    pub op: SemanticOp,
}

impl ProviderHealthKey {
    pub fn new(provider: SemanticProviderId, transport: impl Into<String>, op: SemanticOp) -> Self {
        Self {
            provider,
            transport: bounded_transport_identity(&transport.into()),
            op,
        }
    }
}

/// Bound a transport identity string: short identities are kept verbatim,
/// long ones are replaced by a stable digest prefix (endpoints are already
/// size-bounded by config validation; this is defense in depth).
pub fn bounded_transport_identity(identity: &str) -> String {
    if identity.len() <= MAX_TRANSPORT_IDENTITY_BYTES {
        return identity.to_string();
    }
    let digest = blake3::hash(identity.as_bytes());
    format!("blake3:{}", digest.to_hex())
}

/// Observable health of one provider/transport/operation key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderHealth {
    /// Consecutive recoverable failures since the last success.
    pub consecutive_failures: u32,
    /// Wall-clock (registry clock) until which this key is skipped; `0`
    /// means not cooling.
    pub cooldown_until_ms: i64,
    /// Last successful call, when one happened.
    pub last_success_ms: Option<i64>,
    /// Last recoverable failure, when one happened.
    pub last_failure_ms: Option<i64>,
    /// Recorded latency samples (most recent last), bounded.
    pub latency_samples_ms: Vec<u64>,
}

impl ProviderHealth {
    fn new() -> Self {
        Self {
            consecutive_failures: 0,
            cooldown_until_ms: 0,
            last_success_ms: None,
            last_failure_ms: None,
            latency_samples_ms: Vec::new(),
        }
    }

    pub fn is_cooling_at(&self, now_ms: i64) -> bool {
        self.cooldown_until_ms > now_ms
    }
}

struct Tracked {
    health: ProviderHealth,
    touched_ms: i64,
}

/// Bounded health registry shared by dispatch. All methods are synchronous
/// and poison-tolerant: a panicking provider can never poison dispatch.
pub struct SemanticHealthTracker {
    clock: Arc<dyn Clock>,
    entries: Mutex<BTreeMap<ProviderHealthKey, Tracked>>,
    max_entries: usize,
    max_latency_samples: usize,
    base_cooldown_ms: i64,
    max_cooldown_ms: i64,
}

impl std::fmt::Debug for SemanticHealthTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticHealthTracker")
            .field("keys", &self.len())
            .field("max_entries", &self.max_entries)
            .field("base_cooldown_ms", &self.base_cooldown_ms)
            .field("max_cooldown_ms", &self.max_cooldown_ms)
            .finish()
    }
}

impl SemanticHealthTracker {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            entries: Mutex::new(BTreeMap::new()),
            max_entries: MAX_HEALTH_ENTRIES,
            max_latency_samples: MAX_LATENCY_SAMPLES,
            base_cooldown_ms: BASE_COOLDOWN_MS,
            max_cooldown_ms: MAX_COOLDOWN_MS,
        }
    }

    /// Explicit bounds for adversarial tests: tiny limits prove the state
    /// stays bounded without relying on the production constants.
    pub fn with_bounds(
        clock: Arc<dyn Clock>,
        max_entries: usize,
        max_latency_samples: usize,
        base_cooldown_ms: i64,
        max_cooldown_ms: i64,
    ) -> Self {
        Self {
            clock,
            entries: Mutex::new(BTreeMap::new()),
            max_entries: max_entries.max(1),
            max_latency_samples: max_latency_samples.max(1),
            base_cooldown_ms: base_cooldown_ms.max(1),
            max_cooldown_ms: max_cooldown_ms.max(base_cooldown_ms.max(1)),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<ProviderHealthKey, Tracked>> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn len(&self) -> usize {
        self.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// `true` when the key is inside its cooldown window and must be skipped
    /// until the window elapses.
    pub fn is_cooling(&self, key: &ProviderHealthKey) -> bool {
        let now = self.clock.now_ms();
        self.is_cooling_at(key, now)
    }

    pub fn is_cooling_at(&self, key: &ProviderHealthKey, now_ms: i64) -> bool {
        self.lock()
            .get(key)
            .is_some_and(|tracked| tracked.health.is_cooling_at(now_ms))
    }

    /// Record a successful call: reset the failure streak and cooldown and
    /// append one bounded latency sample.
    pub fn record_success(&self, key: ProviderHealthKey, latency_ms: u64) {
        let now = self.clock.now_ms();
        let mut entries = self.lock();
        if !entries.contains_key(&key) && entries.len() >= self.max_entries {
            evict_least_recently_touched(&mut entries);
        }
        let tracked = entries.entry(key).or_insert_with(|| Tracked {
            health: ProviderHealth::new(),
            touched_ms: now,
        });
        tracked.health.consecutive_failures = 0;
        tracked.health.cooldown_until_ms = 0;
        tracked.health.last_success_ms = Some(now);
        tracked.touched_ms = now;
        tracked.health.latency_samples_ms.push(latency_ms);
        if tracked.health.latency_samples_ms.len() > self.max_latency_samples {
            let excess = tracked.health.latency_samples_ms.len() - self.max_latency_samples;
            tracked.health.latency_samples_ms.drain(..excess);
        }
    }

    /// Record a recoverable failure: extend the consecutive-failure streak
    /// and enter (or extend) a bounded exponential cooldown window.
    pub fn record_failure(&self, key: ProviderHealthKey) {
        let now = self.clock.now_ms();
        let mut entries = self.lock();
        if !entries.contains_key(&key) && entries.len() >= self.max_entries {
            evict_least_recently_touched(&mut entries);
        }
        let tracked = entries.entry(key).or_insert_with(|| Tracked {
            health: ProviderHealth::new(),
            touched_ms: now,
        });
        tracked.health.consecutive_failures = tracked.health.consecutive_failures.saturating_add(1);
        let backoff = cooldown_ms(
            tracked.health.consecutive_failures,
            self.base_cooldown_ms,
            self.max_cooldown_ms,
        );
        tracked.health.cooldown_until_ms = now.saturating_add(backoff);
        tracked.health.last_failure_ms = Some(now);
        tracked.touched_ms = now;
    }

    /// One cloned health row, for introspection and tests.
    pub fn health_for(&self, key: &ProviderHealthKey) -> Option<ProviderHealth> {
        self.lock().get(key).map(|tracked| tracked.health.clone())
    }

    /// Every retained health row, deterministic by key order.
    pub fn snapshot(&self) -> Vec<(ProviderHealthKey, ProviderHealth)> {
        self.lock()
            .iter()
            .map(|(key, tracked)| (key.clone(), tracked.health.clone()))
            .collect()
    }
}

fn evict_least_recently_touched(entries: &mut BTreeMap<ProviderHealthKey, Tracked>) {
    let Some(oldest) = entries
        .iter()
        .min_by_key(|(_, tracked)| tracked.touched_ms)
        .map(|(key, _)| key.clone())
    else {
        return;
    };
    entries.remove(&oldest);
}

/// Bounded exponential backoff: `base * 2^(failures-1)`, capped. Overflow is
/// impossible (shift saturates before the multiplication).
fn cooldown_ms(consecutive_failures: u32, base_ms: i64, max_ms: i64) -> i64 {
    let shift = consecutive_failures.saturating_sub(1).min(31);
    let scaled = base_ms.saturating_mul(1i64 << shift);
    scaled.min(max_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::provider_id;
    use faktor_core::TestClock;

    fn key(provider: &str, op: SemanticOp) -> ProviderHealthKey {
        ProviderHealthKey::new(provider_id(provider), "endpoint-a", op)
    }

    #[test]
    fn failure_streak_enters_bounded_exponential_cooldown() {
        let clock = Arc::new(TestClock::new(1_000));
        let tracker = SemanticHealthTracker::with_bounds(clock.clone(), 16, 4, 100, 1_000);
        let k = key("a", SemanticOp::Verify);
        assert!(!tracker.is_cooling(&k));

        tracker.record_failure(k.clone());
        let health = tracker.health_for(&k).unwrap();
        assert_eq!(health.consecutive_failures, 1);
        assert_eq!(health.cooldown_until_ms, 1_100);
        assert!(tracker.is_cooling(&k));

        // Attempts during the window are skipped; attempts after it are not.
        clock.advance(99);
        assert!(tracker.is_cooling(&k));
        clock.advance(1);
        assert!(!tracker.is_cooling(&k), "exact boundary is elapsed");

        // Second consecutive failure doubles the window.
        tracker.record_failure(k.clone());
        assert_eq!(tracker.health_for(&k).unwrap().cooldown_until_ms, 1_300);
        // The cap holds under a hostile failure flood.
        for _ in 0..64 {
            tracker.record_failure(k.clone());
        }
        let capped = tracker.health_for(&k).unwrap();
        assert_eq!(capped.cooldown_until_ms, clock.now_ms() + 1_000);
        assert!(capped.consecutive_failures > 64);

        // Success resets the streak, clears the cooldown and records latency.
        tracker.record_success(k.clone(), 42);
        let healed = tracker.health_for(&k).unwrap();
        assert_eq!(healed.consecutive_failures, 0);
        assert_eq!(healed.cooldown_until_ms, 0);
        assert_eq!(healed.last_success_ms, Some(clock.now_ms()));
        assert_eq!(healed.latency_samples_ms, vec![42]);
        assert!(!tracker.is_cooling(&k));
    }

    #[test]
    fn health_keys_are_transport_and_operation_scoped() {
        let clock = Arc::new(TestClock::new(0));
        let tracker = SemanticHealthTracker::new(clock);
        let base = key("a", SemanticOp::Verify);
        tracker.record_failure(base.clone());
        let other_op = ProviderHealthKey::new(provider_id("a"), "endpoint-a", SemanticOp::Context);
        let other_transport =
            ProviderHealthKey::new(provider_id("a"), "endpoint-b", SemanticOp::Verify);
        assert!(tracker.is_cooling(&base));
        assert!(
            !tracker.is_cooling(&other_op),
            "a failing Verify provider must not cool its Context class"
        );
        assert!(
            !tracker.is_cooling(&other_transport),
            "a failing endpoint must not cool a sibling endpoint"
        );
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn latency_samples_and_health_entries_stay_bounded() {
        let clock = Arc::new(TestClock::new(0));
        let tracker = SemanticHealthTracker::with_bounds(clock.clone(), 4, 3, 100, 1_000);
        let k = key("a", SemanticOp::Verify);
        for sample in 0..10u64 {
            tracker.record_success(k.clone(), sample);
        }
        let health = tracker.health_for(&k).unwrap();
        assert_eq!(
            health.latency_samples_ms,
            vec![7, 8, 9],
            "only the most recent latency samples are retained"
        );

        for index in 0..10u32 {
            let mut key = key("a", SemanticOp::Verify);
            key.transport = format!("endpoint-{index}");
            clock.advance(1);
            tracker.record_success(key, 1);
        }
        assert_eq!(tracker.len(), 4, "entries are bounded");
        // The most recently touched endpoint survives; the oldest is evicted.
        let surviving: Vec<String> = tracker
            .snapshot()
            .into_iter()
            .map(|(key, _)| key.transport)
            .collect();
        assert_eq!(
            surviving,
            vec!["endpoint-6", "endpoint-7", "endpoint-8", "endpoint-9"]
        );
    }

    #[test]
    fn hostile_transport_identity_is_bounded() {
        let oversized = "x".repeat(MAX_TRANSPORT_IDENTITY_BYTES + 1);
        let bounded = bounded_transport_identity(&oversized);
        assert!(bounded.starts_with("blake3:"));
        assert!(bounded.len() <= MAX_TRANSPORT_IDENTITY_BYTES);
        assert_eq!(bounded, bounded_transport_identity(&oversized));
        let small = "process:/bin/true";
        assert_eq!(bounded_transport_identity(small), small);
    }
}
