//! kilop-memory — long-term structured session memory.
//!
//! The transcript is *not* memory. Durable task state and structured facts
//! are. `kilop-memory` wraps the `memory_fact` table and provides a compact
//! context render for the "semi-stable" memory class.

use std::sync::Arc;

use kilop_core::id::SessionId;
use kilop_store::Store;

/// A single durable fact. `kind` is a small taxonomy (decision, constraint,
/// known_failure, preference, discovered_symbol, ...).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryFact {
    pub kind: String,
    pub key: String,
    pub value: String,
    pub updated_ms: i64,
}

/// Structured long-term memory for one session, backed by the store.
#[derive(Debug, Clone)]
pub struct SessionMemory {
    store: Arc<Store>,
    session: SessionId,
}

impl SessionMemory {
    pub fn new(store: Arc<Store>, session: SessionId) -> Self {
        Self { store, session }
    }

    /// Upsert a fact; the same (kind, key) is overwritten, never duplicated.
    pub fn remember(
        &self,
        kind: &str,
        key: &str,
        value: &str,
    ) -> Result<(), kilop_store::StoreError> {
        self.store
            .upsert_memory_fact(self.session, kind, key, value)
    }

    pub fn facts(&self) -> Result<Vec<MemoryFact>, kilop_store::StoreError> {
        Ok(self
            .store
            .memory_facts(self.session)?
            .into_iter()
            .map(|(kind, key, value)| MemoryFact {
                kind,
                key,
                value,
                updated_ms: 0,
            })
            .collect())
    }

    pub fn by_kind(&self, kind: &str) -> Result<Vec<MemoryFact>, kilop_store::StoreError> {
        Ok(self
            .facts()?
            .into_iter()
            .filter(|f| f.kind == kind)
            .collect())
    }

    pub fn latest(&self, kind: &str, key: &str) -> Result<Option<String>, kilop_store::StoreError> {
        Ok(self
            .facts()?
            .into_iter()
            .find(|f| f.kind == kind && f.key == key)
            .map(|f| f.value))
    }

    /// Compact render for the context engine. Bounded: if the full fact list
    /// exceeds `max_chars`, only the most recent (by insertion order) facts
    /// survive, oldest dropped first.
    pub fn render_for_context(&self, max_chars: usize) -> Result<String, kilop_store::StoreError> {
        let facts = self.facts()?;
        let mut out = String::new();
        for f in facts.iter().rev() {
            let line = format!("- {}: {}\n", f.key, f.value);
            if out.len() + line.len() > max_chars {
                break;
            }
            out.push_str(&line);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn fixture() -> (tempfile::TempDir, Arc<Store>, SessionId) {
        let dir = tempdir().unwrap();
        let store = Arc::new(Store::open(dir.path(), true).unwrap());
        let ws = store.create_workspace("/w").unwrap();
        let session = store.create_session(ws, "t", "p", "m").unwrap();
        (dir, store, session.id)
    }

    #[test]
    fn upsert_never_duplicates() {
        let (_d, store, session) = fixture();
        let mem = SessionMemory::new(store.clone(), session);
        mem.remember("decision", "framework", "rust").unwrap();
        mem.remember("decision", "framework", "rust+tokio").unwrap();
        mem.remember("decision", "framework", "rust+tokio+axum")
            .unwrap();
        assert_eq!(mem.by_kind("decision").unwrap().len(), 1);
        assert_eq!(
            mem.latest("decision", "framework").unwrap().as_deref(),
            Some("rust+tokio+axum")
        );
    }

    #[test]
    fn facts_survive_reopen() {
        let dir = tempdir().unwrap();
        let session = {
            let store = Arc::new(Store::open(dir.path(), true).unwrap());
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let mem = SessionMemory::new(store.clone(), s.id);
            mem.remember("preference", "language", "rust").unwrap();
            mem.remember("known_failure", "test_e2e", "flaky on CI")
                .unwrap();
            s.id
        };
        let store = Arc::new(Store::open(dir.path(), true).unwrap());
        let mem = SessionMemory::new(store.clone(), session);
        assert_eq!(mem.facts().unwrap().len(), 2);
        assert_eq!(mem.by_kind("known_failure").unwrap().len(), 1);
    }

    #[test]
    fn render_is_bounded() {
        let (_d, store, session) = fixture();
        let mem = SessionMemory::new(store.clone(), session);
        for i in 0..200 {
            mem.remember("decision", &format!("k{i}"), &"v".repeat(50))
                .unwrap();
        }
        let render = mem.render_for_context(300).unwrap();
        assert!(render.len() <= 300, "render {} exceeds bound", render.len());
        assert!(!render.is_empty());
        // Zero budget → empty, never panic.
        assert_eq!(mem.render_for_context(0).unwrap(), "");
    }

    #[test]
    fn malicious_fact_values_are_stored_verbatim() {
        let (_d, store, session) = fixture();
        let mem = SessionMemory::new(store.clone(), session);
        let evil = "line1\nline2; DROP TABLE memory_fact;--\n\"quotes\"";
        mem.remember("decision", "note", evil).unwrap();
        // SQL injection attempt must not destroy the table.
        let facts = mem.facts().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].value, evil);
        // And the store still works.
        mem.remember("decision", "after", "ok").unwrap();
        assert_eq!(mem.facts().unwrap().len(), 2);
    }

    #[test]
    fn kind_taxonomy_is_free_but_queries_are_exact() {
        let (_d, store, session) = fixture();
        let mem = SessionMemory::new(store.clone(), session);
        mem.remember("decision", "a", "1").unwrap();
        mem.remember("decision", "b", "2").unwrap();
        mem.remember("constraint", "a", "3").unwrap();
        assert_eq!(mem.by_kind("decision").unwrap().len(), 2);
        assert_eq!(mem.by_kind("constraint").unwrap().len(), 1);
        assert_eq!(mem.by_kind("nonexistent").unwrap().len(), 0);
    }

    #[test]
    fn memory_is_per_session_isolated() {
        let (_d, store, _s1) = fixture();
        let ws = store.create_workspace("/w").unwrap();
        let s2 = store.create_session(ws, "other", "p", "m").unwrap();
        let m1 = SessionMemory::new(store.clone(), SessionId::new(1));
        let m2 = SessionMemory::new(store.clone(), s2.id);
        m1.remember("decision", "secret", "s1").unwrap();
        m2.remember("decision", "secret", "s2").unwrap();
        assert_eq!(
            m1.latest("decision", "secret").unwrap().as_deref(),
            Some("s1")
        );
        assert_eq!(
            m2.latest("decision", "secret").unwrap().as_deref(),
            Some("s2")
        );
    }
}
