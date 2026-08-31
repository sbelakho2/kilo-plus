//! Injectable clocks and deadlines. All runtime code takes a `Clock` so that
//! deadline/crash tests can control time without sleeping.

use std::sync::atomic::Ordering;

pub trait Clock: Send + Sync {
    /// Milliseconds since Unix epoch.
    fn now_ms(&self) -> i64;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }
}

/// Test clock; `advance` may go backwards to simulate skew (the journal's
/// monotonicity rules must absorb that).
#[derive(Debug, Clone, Default)]
pub struct TestClock(std::sync::Arc<std::sync::atomic::AtomicI64>);

impl TestClock {
    pub fn new(now_ms: i64) -> Self {
        let c = TestClock::default();
        c.0.store(now_ms, Ordering::SeqCst);
        c
    }

    pub fn advance(&self, delta_ms: i64) {
        self.0.fetch_add(delta_ms, Ordering::SeqCst);
    }

    pub fn set(&self, now_ms: i64) {
        self.0.store(now_ms, Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn now_ms(&self) -> i64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// A point in time; expired relative to a clock reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Deadline {
    at_ms: i64,
}

impl Deadline {
    pub const fn at(at_ms: i64) -> Self {
        Self { at_ms }
    }

    pub fn now_plus<C: Clock>(clock: &C, duration_ms: u64) -> Self {
        Self {
            at_ms: clock
                .now_ms()
                .saturating_add(duration_ms.min(i64::MAX as u64) as i64),
        }
    }

    pub fn is_expired(self, now_ms: i64) -> bool {
        now_ms >= self.at_ms
    }

    pub fn at_ms(self) -> i64 {
        self.at_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadline_expiry_is_inclusive_and_skew_safe() {
        let clock = TestClock::new(1000);
        let d = Deadline::now_plus(&clock, 100);
        assert!(!d.is_expired(clock.now_ms()));
        clock.advance(99);
        assert!(!d.is_expired(clock.now_ms()));
        clock.advance(1);
        assert!(d.is_expired(clock.now_ms()), "exact boundary counts as expired");
        // clock skew backwards: deadline relative to a future stamp still holds
        assert!(Deadline::at(500).is_expired(1000));
        assert!(!Deadline::at(1500).is_expired(1000));
    }

    #[test]
    fn deadline_at_overflow_saturates() {
        let clock = TestClock::new(i64::MAX);
        let d = Deadline::now_plus(&clock, u64::MAX);
        assert_eq!(d.at_ms(), i64::MAX);
    }

    #[test]
    fn test_clock_skew_is_controllable() {
        let clock = TestClock::new(100);
        clock.advance(-200); // simulate skew
        assert_eq!(clock.now_ms(), -100);
        clock.set(42);
        assert_eq!(clock.now_ms(), 42);
    }
}
