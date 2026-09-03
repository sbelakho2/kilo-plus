//! Resource classes and concurrency budgets. Each class gets a concurrency
//! budget so that e.g. embedding/indexing work can never starve an
//! interactive coding session.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceClass {
    Model,
    DiskRead,
    DiskWrite,
    Cpu,
    Git,
    Network,
    Terminal,
    Mcp,
    Indexing,
}

impl ResourceClass {
    pub const ALL: [ResourceClass; 9] = [
        ResourceClass::Model,
        ResourceClass::DiskRead,
        ResourceClass::DiskWrite,
        ResourceClass::Cpu,
        ResourceClass::Git,
        ResourceClass::Network,
        ResourceClass::Terminal,
        ResourceClass::Mcp,
        ResourceClass::Indexing,
    ];
}

/// Default per-class concurrency limits (in-flight operations).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceLimits {
    pub limits: HashMap<ResourceClass, usize>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        let mut limits = HashMap::new();
        limits.insert(ResourceClass::Model, 1);
        limits.insert(ResourceClass::DiskRead, 16);
        limits.insert(ResourceClass::DiskWrite, 4);
        limits.insert(ResourceClass::Cpu, 2);
        limits.insert(ResourceClass::Git, 1); // serialized per repo in faktor-git
        limits.insert(ResourceClass::Network, 8);
        limits.insert(ResourceClass::Terminal, 2);
        limits.insert(ResourceClass::Mcp, 4);
        limits.insert(ResourceClass::Indexing, 1); // deliberately low: indexing yields
        Self { limits }
    }
}

impl ResourceLimits {
    pub fn get(&self, class: ResourceClass) -> usize {
        self.limits.get(&class).copied().unwrap_or(1)
    }
}

/// Tracks in-flight operations per class; `acquire` fails fast instead of
/// blocking so callers can queue by priority (interactive coding beats
/// background indexing).
#[derive(Debug, Default)]
pub struct ResourceGauge {
    in_flight: HashMap<ResourceClass, usize>,
}

impl ResourceGauge {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns Ok(()) if the class has capacity; otherwise Err with the
    /// current usage so the caller can retry later or yield.
    pub fn try_acquire(
        &mut self,
        class: ResourceClass,
        limits: &ResourceLimits,
    ) -> Result<(), ResourceBusy> {
        let used = self.in_flight.get(&class).copied().unwrap_or(0);
        let max = limits.get(class);
        if used >= max {
            return Err(ResourceBusy { class, used, max });
        }
        *self.in_flight.entry(class).or_insert(0) += 1;
        Ok(())
    }

    /// Release a slot; panics on underflow (release without acquire is a bug
    /// that must be loud, not silent).
    pub fn release(&mut self, class: ResourceClass) {
        let e = self.in_flight.entry(class).or_insert(0);
        *e = e
            .checked_sub(1)
            .expect("resource release without matching acquire");
    }

    pub fn usage(&self, class: ResourceClass) -> usize {
        self.in_flight.get(&class).copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceBusy {
    pub class: ResourceClass,
    pub used: usize,
    pub max: usize,
}

impl std::fmt::Display for ResourceBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "resource class {:?} busy: {} of {} in flight",
            self.class, self.used, self.max
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_are_per_class_and_bounded() {
        let limits = ResourceLimits::default();
        let mut g = ResourceGauge::new();
        // Model budget = 1: second model call must fail fast.
        assert!(g.try_acquire(ResourceClass::Model, &limits).is_ok());
        let err = g.try_acquire(ResourceClass::Model, &limits).unwrap_err();
        assert_eq!(err.used, 1);
        assert_eq!(err.max, 1);
        // Indexing budget 1: background indexing cannot starve anything.
        assert!(g.try_acquire(ResourceClass::Indexing, &limits).is_ok());
        assert!(g.try_acquire(ResourceClass::Indexing, &limits).is_err());
        // Disk reads are wide.
        for _ in 0..limits.get(ResourceClass::DiskRead) {
            g.try_acquire(ResourceClass::DiskRead, &limits).unwrap();
        }
        assert!(g.try_acquire(ResourceClass::DiskRead, &limits).is_err());
    }

    #[test]
    #[should_panic]
    fn double_release_is_loud() {
        let limits = ResourceLimits::default();
        let mut g = ResourceGauge::new();
        g.try_acquire(ResourceClass::Git, &limits).unwrap();
        g.release(ResourceClass::Git);
        g.release(ResourceClass::Git); // bug: must panic
    }

    #[test]
    fn release_restores_capacity() {
        let limits = ResourceLimits::default();
        let mut g = ResourceGauge::new();
        g.try_acquire(ResourceClass::Terminal, &limits).unwrap();
        g.try_acquire(ResourceClass::Terminal, &limits).unwrap();
        assert!(g.try_acquire(ResourceClass::Terminal, &limits).is_err());
        g.release(ResourceClass::Terminal);
        assert!(g.try_acquire(ResourceClass::Terminal, &limits).is_ok());
        assert_eq!(g.usage(ResourceClass::Terminal), 2);
    }

    #[test]
    fn class_absent_from_limits_defaults_to_budget_one() {
        let mut limits = ResourceLimits::default();
        // Remove Cpu from the map: it must fall back to budget 1, and the
        // fallback must be per-class (never 0, never unbounded).
        limits.limits.remove(&ResourceClass::Cpu);
        assert_eq!(limits.get(ResourceClass::Cpu), 1);
        let mut g = ResourceGauge::new();
        assert!(g.try_acquire(ResourceClass::Cpu, &limits).is_ok());
        assert!(g.try_acquire(ResourceClass::Cpu, &limits).is_err());
        // DiskWrite still has its configured budget of 4.
        assert_eq!(limits.get(ResourceClass::DiskWrite), 4);
    }

    #[test]
    fn adversarial_mixed_stress() {
        let limits = ResourceLimits::default();
        let mut g = ResourceGauge::new();
        let mut taken = vec![];
        for class in ResourceClass::ALL {
            for _ in 0..limits.get(class) {
                g.try_acquire(class, &limits).unwrap();
                taken.push(class);
            }
        }
        // All full: every acquire fails fast (no blocking).
        for class in ResourceClass::ALL {
            assert!(g.try_acquire(class, &limits).is_err());
        }
        // Release everything in reverse; usage must hit zero.
        for class in taken.iter().rev() {
            g.release(*class);
        }
        for class in ResourceClass::ALL {
            assert_eq!(g.usage(class), 0);
        }
    }
}
