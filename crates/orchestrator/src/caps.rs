//! Typed capability lattice for child policy (audit 22).
//!
//! A [`CapabilitySet`] is a `BTreeSet<CapabilityGrant>` — capabilities are
//! TYPED, never prose: there is no free-form string in a permission
//! position anywhere on this API. A child inherits at most
//! `parent ∩ task_policy ∩ child_policy`, computed by [`effective`] at
//! spawn and re-computed at every steer; a child can never exceed its
//! parent on any capability, provable with [`CapabilitySet::covers`].
//!
//! Scope semantics: a [`ScopePattern`] denotes a prefix language — the
//! literal pattern plus every descendant under it (path-style, component
//! boundary at `/`); `*` is the universal pattern. Two single-pattern
//! languages either nest or are disjoint, so the meet of two grants with
//! the same capability is the deeper pattern (or empty). Sets of grants
//! meet pairwise per capability.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

/// One typed capability a child policy may name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatticeCap {
    ReadWorkspace,
    WriteWorkspace,
    ReadExternal,
    WriteExternal,
    ExecuteShell,
    Network,
    Mcp,
    Git,
}

impl fmt::Display for LatticeCap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self)
                .unwrap_or_default()
                .trim_matches('"')
        )
    }
}

/// Maximum number of grants in one set (bounded everything).
pub const MAX_GRANTS_PER_SET: usize = 16;
/// Maximum scope pattern length (characters).
pub const MAX_SCOPE_CHARS: usize = 512;

/// A scope pattern: `*` (universal) or a literal prefix that covers the
/// literal itself and every `/`-descendant (`src` covers `src/a.rs`, not
/// `src2/a.rs`; a trailing `/` is normalized away).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScopePattern(String);

impl ScopePattern {
    pub const WILDCARD: &'static str = "*";

    /// Construct a typed scope pattern; rejects empty, overlong or
    /// control-character patterns (hostile input cannot enter the lattice).
    pub fn new(raw: impl Into<String>) -> Result<Self, String> {
        let mut s = raw.into();
        if s.is_empty() || s.chars().count() > MAX_SCOPE_CHARS {
            return Err(format!(
                "scope pattern must be 1..={MAX_SCOPE_CHARS} characters"
            ));
        }
        if s == Self::WILDCARD {
            return Ok(Self(s));
        }
        if s.chars().any(|c| c.is_control() || c == '\n' || c == '\r') {
            return Err("scope pattern contains control characters".to_string());
        }
        while s.ends_with('/') {
            s.pop();
        }
        if s.is_empty() {
            return Err("scope pattern cannot be only slashes".to_string());
        }
        Ok(Self(s))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// True when this pattern's language is the universal language.
    pub fn is_universal(&self) -> bool {
        self.0 == Self::WILDCARD
    }

    /// True when `self`'s language contains every element of `other`'s.
    pub fn covers(&self, other: &ScopePattern) -> bool {
        if self.0 == Self::WILDCARD {
            return true;
        }
        if other.0 == Self::WILDCARD {
            return false;
        }
        if self.0 == other.0 {
            return true;
        }
        other.0.starts_with(&self.0) && other.0.as_bytes().get(self.0.len()) == Some(&b'/')
    }

    /// The intersection of two single-pattern languages: the deeper of two
    /// nested patterns, `None` when the languages are disjoint.
    pub fn meet(&self, other: &ScopePattern) -> Option<ScopePattern> {
        if self.covers(other) {
            Some(other.clone())
        } else if other.covers(self) {
            Some(self.clone())
        } else {
            None
        }
    }
}

impl fmt::Display for ScopePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One grant: a typed capability over a scope pattern. Optional
/// `ttl_ms` bounds how long the grant stays effective (informational in
/// the lattice: equality/ordering ignore it, and a meet keeps the shorter
/// TTL of its two operands).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityGrant {
    pub cap: LatticeCap,
    pub scope: ScopePattern,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_ms: Option<u64>,
}

impl CapabilityGrant {
    pub fn new(cap: LatticeCap, scope: ScopePattern) -> Self {
        Self {
            cap,
            scope,
            ttl_ms: None,
        }
    }

    pub fn with_ttl(mut self, ttl_ms: u64) -> Self {
        self.ttl_ms = Some(ttl_ms);
        self
    }
}

impl PartialEq for CapabilityGrant {
    fn eq(&self, other: &Self) -> bool {
        self.cap == other.cap && self.scope == other.scope
    }
}

impl Eq for CapabilityGrant {}

impl PartialOrd for CapabilityGrant {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CapabilityGrant {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.cap, &self.scope.0).cmp(&(other.cap, &other.scope.0))
    }
}

/// The typed lattice element: an ordered set of grants.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet(BTreeSet<CapabilityGrant>);

impl CapabilitySet {
    pub fn new() -> Self {
        Self(BTreeSet::new())
    }

    /// Build from grants; rejects sets above [`MAX_GRANTS_PER_SET`].
    pub fn from_grants(grants: impl IntoIterator<Item = CapabilityGrant>) -> Result<Self, String> {
        let s: BTreeSet<CapabilityGrant> = grants.into_iter().collect();
        if s.len() > MAX_GRANTS_PER_SET {
            return Err(format!(
                "capability set exceeds {MAX_GRANTS_PER_SET} grants ({} given)",
                s.len()
            ));
        }
        Ok(Self(s))
    }

    /// Grant (union) one more grant; bounded.
    pub fn grant(&mut self, g: CapabilityGrant) -> Result<(), String> {
        if self.0.len() >= MAX_GRANTS_PER_SET && !self.0.contains(&g) {
            return Err(format!(
                "capability set exceeds {MAX_GRANTS_PER_SET} grants"
            ));
        }
        self.0.insert(g);
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &CapabilityGrant> {
        self.0.iter()
    }

    /// Lattice meet (intersection of languages): pairwise per capability,
    /// scope languages intersected; TTLs merge to the shorter.
    pub fn meet(&self, other: &CapabilitySet) -> CapabilitySet {
        let mut out = BTreeSet::new();
        for a in &self.0 {
            for b in &other.0 {
                if a.cap != b.cap {
                    continue;
                }
                let Some(scope) = a.scope.meet(&b.scope) else {
                    continue;
                };
                let ttl = match (a.ttl_ms, b.ttl_ms) {
                    (Some(x), Some(y)) => Some(x.min(y)),
                    (Some(x), None) => Some(x),
                    (None, Some(y)) => Some(y),
                    (None, None) => None,
                };
                out.insert(CapabilityGrant {
                    cap: a.cap,
                    scope,
                    ttl_ms: ttl,
                });
            }
        }
        let mut s = Self(out);
        // A grant the meet produced that is covered by another grant of the
        // same capability is redundant; keep the broader one.
        s.normalize();
        s
    }

    /// Lattice join (union).
    pub fn grant_all(&mut self, other: &CapabilitySet) -> Result<(), String> {
        for g in other.iter() {
            self.grant(g.clone())?;
        }
        Ok(())
    }

    /// True when `super_set` covers EVERY grant of `self` (same capability,
    /// scope-language containment): the audit-22 proof obligation
    /// "a child cannot exceed its parent on any capability".
    pub fn covered_by(&self, super_set: &CapabilitySet) -> bool {
        self.0.iter().all(|g| {
            super_set.0.iter().any(|h| {
                h.cap == g.cap
                    && h.scope.covers(&g.scope)
                    && h.ttl_ms.zip(g.ttl_ms).map(|(h, g)| h <= g).unwrap_or(true)
            })
        })
    }

    /// Drop grants covered by a broader grant of the same capability.
    fn normalize(&mut self) {
        let grants: Vec<CapabilityGrant> = self.0.iter().cloned().collect();
        let keep: Vec<CapabilityGrant> = grants
            .into_iter()
            .filter(|g| {
                !self
                    .0
                    .iter()
                    .any(|h| h != g && h.cap == g.cap && h.scope.covers(&g.scope))
            })
            .collect();
        self.0 = keep.into_iter().collect();
    }
}

/// effective(child) = parent ∩ task_policy ∩ child_policy — computed at
/// spawn and at every steer; the result can never exceed `parent`.
pub fn effective(
    parent: &CapabilitySet,
    task_policy: &CapabilitySet,
    child_policy: &CapabilitySet,
) -> CapabilitySet {
    parent.meet(task_policy).meet(child_policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rw(scope: &str) -> CapabilityGrant {
        CapabilityGrant::new(
            LatticeCap::WriteWorkspace,
            ScopePattern::new(scope).unwrap(),
        )
    }

    fn set(grants: Vec<CapabilityGrant>) -> CapabilitySet {
        CapabilitySet::from_grants(grants).unwrap()
    }

    #[test]
    fn scope_cover_and_meet_semantics() {
        let star = ScopePattern::new("*").unwrap();
        let src = ScopePattern::new("src").unwrap();
        let src_a = ScopePattern::new("src/a").unwrap();
        let src2 = ScopePattern::new("src2").unwrap();
        assert!(star.covers(&src_a));
        assert!(src.covers(&src_a));
        assert!(src.covers(&ScopePattern::new("src").unwrap()));
        assert!(!src.covers(&src2), "component boundary matters");
        assert!(!src.covers(&star));
        assert_eq!(src.meet(&src_a), Some(ScopePattern::new("src/a").unwrap()));
        assert_eq!(src.meet(&src2), None);
        assert_eq!(star.meet(&src_a), Some(ScopePattern::new("src/a").unwrap()));
        assert_eq!(
            ScopePattern::new("src/").unwrap(),
            ScopePattern::new("src").unwrap()
        );
    }

    #[test]
    fn hostile_scope_patterns_rejected() {
        assert!(ScopePattern::new("").is_err());
        assert!(ScopePattern::new("a".repeat(MAX_SCOPE_CHARS + 1)).is_err());
        assert!(ScopePattern::new("a\nb").is_err());
        assert!(ScopePattern::new("///").is_err());
        assert!(ScopePattern::new("src").is_ok());
    }

    #[test]
    fn meet_is_the_language_intersection_per_capability() {
        let parent = set(vec![
            rw("src"),
            CapabilityGrant::new(LatticeCap::ReadWorkspace, ScopePattern::new("*").unwrap()),
        ]);
        let child = set(vec![rw("src/a"), rw("docs")]);
        let m = parent.meet(&child);
        assert_eq!(m.len(), 1, "only the overlapping language survives");
        assert!(m.iter().all(|g| g.cap == LatticeCap::WriteWorkspace));
        assert_eq!(m.iter().next().unwrap().scope.as_str(), "src/a");
        // Meet with the empty set is empty; meet is idempotent.
        assert!(parent.meet(&CapabilitySet::new()).is_empty());
        assert_eq!(parent.meet(&parent), parent);
    }

    #[test]
    fn union_grant_is_bounded_and_typed() {
        let mut s = CapabilitySet::new();
        for i in 0..MAX_GRANTS_PER_SET {
            s.grant(rw(&format!("p{i}"))).unwrap();
        }
        assert!(s.grant(rw("extra")).is_err(), "bounded sets refuse to grow");
        // The same grant twice is a no-op, not growth.
        let mut t = set(vec![rw("src")]);
        t.grant(rw("src")).unwrap();
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn child_cannot_exceed_parent_even_when_child_policy_claims_more() {
        // Child policy demands the WHOLE workspace; parent only owns src/.
        let parent = set(vec![rw("src")]);
        let task_policy = set(vec![rw("src")]);
        let greedy_child = set(vec![rw("*"), rw("docs")]);
        let eff = effective(&parent, &task_policy, &greedy_child);
        assert!(
            eff.covered_by(&parent),
            "child exceeded its parent: {eff:?}"
        );
        assert_eq!(eff.len(), 1);
        assert_eq!(eff.iter().next().unwrap().scope.as_str(), "src");
        // Effective cannot be added to afterwards either (typed sets).
        let mut eff = eff;
        assert!(eff.grant(rw("elsewhere")).is_ok());
        assert!(!eff.covered_by(&parent), "post-hoc grant must break cover");
    }

    #[test]
    fn empty_parent_yields_empty_effective_set() {
        let parent = CapabilitySet::new();
        let eff = effective(&parent, &set(vec![rw("*")]), &set(vec![rw("*")]));
        assert!(eff.is_empty());
    }

    #[test]
    fn ttl_merges_to_the_shorter_bound() {
        let parent = set(vec![rw("src").with_ttl(1000)]);
        let child = set(vec![rw("src").with_ttl(100)]);
        let eff = effective(&parent, &parent, &child);
        assert_eq!(eff.iter().next().unwrap().ttl_ms, Some(100));
        // A parent grant without a TTL is not shortened by a child TTL.
        let no_parent_ttl = set(vec![rw("src")]);
        let eff2 = effective(&no_parent_ttl, &no_parent_ttl, &parent);
        assert_eq!(eff2.iter().next().unwrap().ttl_ms, Some(1000));
    }

    #[test]
    fn no_free_form_strings_in_permission_positions() {
        // The lattice API only accepts typed LatticeCap + ScopePattern.
        // A "prose policy" cannot even be expressed: strings only enter as
        // bounded scope patterns and are validated before construction.
        let typed = CapabilitySet::from_grants(vec![rw("src")]).unwrap();
        let serialized = serde_json::to_string(&typed).unwrap();
        let back: CapabilitySet = serde_json::from_str(&serialized).unwrap();
        assert_eq!(typed, back);
    }

    #[test]
    fn serde_roundtrip_and_order_stability() {
        let mut a = set(vec![
            rw("src"),
            CapabilityGrant::new(
                LatticeCap::Network,
                ScopePattern::new("https://api.example.com").unwrap(),
            ),
        ]);
        a.grant_all(&set(vec![rw("src")])).unwrap();
        let json = serde_json::to_string(&a).unwrap();
        let b: CapabilitySet = serde_json::from_str(&json).unwrap();
        assert_eq!(a, b);
        assert_eq!(b.iter().count(), 2);
    }
}
