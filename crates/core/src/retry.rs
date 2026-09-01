//! Retry policies with jitter. Retries are *state-aware*: the caller decides
//! whether replaying is safe; this module only computes bounded backoff.

use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryClass {
    /// Transport failures only (connection refused, reset, timeout).
    Network,
    /// Also retry 429/rate-limit responses.
    RateLimited,
    /// Also retry 5xx.
    ServerError,
    /// Retry every retryable error kind.
    Always,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct RetryPolicy {
    /// Total attempts (1 = no retry).
    pub max_attempts: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    /// Jitter fraction in [0.0, 1.0]; delay is scaled by (1 ± jitter).
    pub jitter: f64,
    pub class: RetryClass,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            base_delay_ms: 250,
            max_delay_ms: 5_000,
            jitter: 0.2,
            class: RetryClass::Network,
        }
    }
}

impl RetryPolicy {
    /// Whether a retry should be attempted after `attempt` failures so far
    /// (0 = first failure). `max_attempts` is the total number of tries,
    /// so with `max_attempts = 1` no retry ever happens.
    pub fn should_retry(&self, attempt: u32, retryable: bool, rate_limited: bool) -> bool {
        if attempt.saturating_add(1) >= self.max_attempts || !retryable {
            return false;
        }
        match self.class {
            RetryClass::Network => !rate_limited, // network-class retries only pure transport errors
            RetryClass::RateLimited => true,
            RetryClass::ServerError => true,
            RetryClass::Always => true,
        }
    }

    /// Exponential backoff with jitter for `attempt` (0-based, before first retry).
    pub fn next_delay(&self, attempt: u32) -> Duration {
        let exp = 2u32.saturating_pow(attempt);
        let base = self.base_delay_ms.saturating_mul(exp as u64);
        let base = base.min(self.max_delay_ms.max(1));
        let jitter = (base as f64 * self.jitter.clamp(0.0, 1.0)) as u64;
        let lo = base.saturating_sub(jitter).max(1);
        let hi = base.saturating_add(jitter);
        // Deterministic midpoint for tests; callers may pass an RNG.
        Duration::from_millis((lo + hi) / 2)
    }

    pub fn next_delay_rng<R: rand::Rng>(&self, attempt: u32, rng: &mut R) -> Duration {
        let exp = 2u32.saturating_pow(attempt);
        let base = self.base_delay_ms.saturating_mul(exp as u64);
        let base = base.min(self.max_delay_ms.max(1));
        let jitter = (base as f64 * self.jitter.clamp(0.0, 1.0)) as u64;
        let lo = base.saturating_sub(jitter).max(1);
        let hi = base.saturating_add(jitter);
        Duration::from_millis(rng.random_range(lo..=hi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ErrorKind;

    #[test]
    fn no_retry_when_max_attempts_is_one() {
        let p = RetryPolicy::default();
        assert!(!p.should_retry(0, true, false));
        assert!(!p.should_retry(0, true, true));
    }

    #[test]
    fn network_class_never_retries_rate_limit() {
        let p = RetryPolicy {
            max_attempts: 5,
            ..Default::default()
        };
        assert!(p.should_retry(0, true, false));
        assert!(
            !p.should_retry(0, true, true),
            "rate-limit is not a network error"
        );
        assert!(
            !p.should_retry(0, false, false),
            "non-retryable never retried"
        );
    }

    #[test]
    fn attempts_are_bounded() {
        let p = RetryPolicy {
            max_attempts: 3,
            ..Default::default()
        };
        assert!(p.should_retry(0, true, false));
        assert!(p.should_retry(1, true, false));
        assert!(
            !p.should_retry(2, true, false),
            "attempt index 2 means 3rd try — stop"
        );
        assert!(!p.should_retry(99, true, false));
    }

    #[test]
    fn backoff_is_bounded_and_grows() {
        let p = RetryPolicy {
            max_attempts: 10,
            base_delay_ms: 100,
            max_delay_ms: 1000,
            jitter: 0.0,
            class: RetryClass::Always,
        };
        let d0 = p.next_delay(0).as_millis();
        let d1 = p.next_delay(1).as_millis();
        let d2 = p.next_delay(2).as_millis();
        assert_eq!(d0, 100);
        assert_eq!(d1, 200);
        assert_eq!(d2, 400);
        // cap applies
        assert_eq!(p.next_delay(10).as_millis(), 1000);
        assert_eq!(p.next_delay(100).as_millis(), 1000);
        // no overflow explosion
        assert!(p.next_delay(u32::MAX).as_millis() <= 1000);
    }

    #[test]
    fn jitter_never_goes_below_1ms() {
        let p = RetryPolicy {
            max_attempts: 5,
            base_delay_ms: 1,
            max_delay_ms: 10,
            jitter: 1.0,
            class: RetryClass::Always,
        };
        for a in 0..5 {
            let d = p.next_delay(a).as_millis();
            assert!((1..=10).contains(&d), "attempt {a} delay {d}");
        }
    }

    #[test]
    fn adversarial_jitter_bounds_with_rng() {
        let p = RetryPolicy {
            max_attempts: 20,
            base_delay_ms: 100,
            max_delay_ms: 1000,
            jitter: 0.5,
            class: RetryClass::Always,
        };
        let mut rng = rand::rng();
        for a in 0..20 {
            let d = p.next_delay_rng(a, &mut rng).as_millis();
            let max_allowed =
                ((100u64.saturating_mul(2u64.saturating_pow(a))).min(1000) as f64 * 1.5) as u128;
            assert!(d <= max_allowed.max(1000), "delay {d} exceeds bound");
            assert!(d >= 1);
        }
    }

    #[test]
    fn rate_limited_class_retries_rate_limits() {
        let p = RetryPolicy {
            max_attempts: 4,
            base_delay_ms: 10,
            max_delay_ms: 100,
            jitter: 0.0,
            class: RetryClass::RateLimited,
        };
        assert!(p.should_retry(0, true, true));
        assert!(
            !p.should_retry(0, false, true),
            "non-retryable error must not retry even with rate-limited class"
        );
    }

    #[test]
    fn retry_class_consistent_with_error_kinds() {
        assert!(ErrorKind::Network.is_retryable());
        assert!(ErrorKind::RateLimited.is_retryable());
        let p = RetryPolicy {
            max_attempts: 3,
            ..Default::default()
        };
        assert!(p.should_retry(0, ErrorKind::Network.is_retryable(), false));
    }
}
