//! faktor-memory — long-term structured session memory.
//!
//! The transcript is *not* memory. Durable task state and structured facts
//! are. `faktor-memory` wraps the `memory_fact` table and provides a compact
//! context render for the "semi-stable" memory class.

use std::sync::Arc;

use faktor_core::id::SessionId;
use faktor_store::Store;

/// A single durable fact. `kind` is a small taxonomy (decision, constraint,
/// known_failure, preference, discovered_symbol, ...).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryFact {
    pub kind: String,
    pub key: String,
    pub value: String,
    pub updated_ms: i64,
}

/// Bounded page size for memory-fact paging (paging is fundamental).
pub const MAX_FACT_PAGE_SIZE: i64 = 200;

/// One deterministic page of memory facts with explicit paging metadata:
/// `{size, cursor, has_more, total_estimate}`. Facts are ordered
/// newest-first by `(updated_ms DESC, kind DESC, key DESC)`; `cursor` is the
/// `(updated_ms, kind, key)` position AFTER this page — pass it back
/// verbatim. Replaying a cursor returns the same window; an upsert moves a
/// row to the newest end, so a backward walk never sees a fact twice.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FactsPage {
    pub facts: Vec<MemoryFact>,
    /// The applied page size (requested limit after clamping).
    pub size: i64,
    /// Cursor for the next older page; `None` on the final page.
    pub cursor: Option<(i64, String, String)>,
    /// True when at least one older page exists.
    pub has_more: bool,
    /// Exact fact count for the session (cheap: facts are few per session).
    pub total_estimate: i64,
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
    ) -> Result<(), faktor_store::StoreError> {
        self.store
            .upsert_memory_fact(self.session, kind, key, value)
    }

    pub fn facts(&self) -> Result<Vec<MemoryFact>, faktor_store::StoreError> {
        // The full read follows the SAME deterministic total order as the
        // paged read (newest-first by updated_ms, tie-broken by kind/key):
        // the paged read is a bounded window of this order.
        let (rows, _has_more) = self.store.memory_facts_page(self.session, None, u64::MAX)?;
        Ok(rows
            .into_iter()
            .map(|r| MemoryFact {
                kind: r.kind,
                key: r.key,
                value: r.value,
                updated_ms: r.updated_ms,
            })
            .collect())
    }

    /// One deterministic page of facts with explicit paging metadata (see
    /// [`FactsPage`]). Bounded: one page + one probe row, never more.
    pub fn facts_page(
        &self,
        after: Option<&(i64, String, String)>,
        limit: i64,
    ) -> Result<FactsPage, faktor_store::StoreError> {
        let limit = limit.clamp(1, MAX_FACT_PAGE_SIZE);
        let (rows, has_more) = self
            .store
            .memory_facts_page(self.session, after, limit as u64)?;
        let cursor = if has_more {
            rows.last()
                .map(|r| (r.updated_ms, r.kind.clone(), r.key.clone()))
        } else {
            None
        };
        let total_estimate = self.store.memory_fact_count(self.session)?;
        Ok(FactsPage {
            facts: rows
                .into_iter()
                .map(|r| MemoryFact {
                    kind: r.kind,
                    key: r.key,
                    value: r.value,
                    updated_ms: r.updated_ms,
                })
                .collect(),
            size: limit,
            cursor,
            has_more,
            total_estimate,
        })
    }

    pub fn by_kind(&self, kind: &str) -> Result<Vec<MemoryFact>, faktor_store::StoreError> {
        Ok(self
            .facts()?
            .into_iter()
            .filter(|f| f.kind == kind)
            .collect())
    }

    pub fn latest(
        &self,
        kind: &str,
        key: &str,
    ) -> Result<Option<String>, faktor_store::StoreError> {
        Ok(self
            .facts()?
            .into_iter()
            .find(|f| f.kind == kind && f.key == key)
            .map(|f| f.value))
    }

    /// Compact render for the context engine. Bounded: if the full fact list
    /// exceeds `max_chars`, only the most recent (by insertion order) facts
    /// survive, oldest dropped first.
    pub fn render_for_context(&self, max_chars: usize) -> Result<String, faktor_store::StoreError> {
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
    #[test]
    fn tmp_probe_remember_read() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(faktor_store::Store::open(dir.path(), true).unwrap());
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        eprintln!("PROBE sid={}", s.id);
        store.upsert_memory_fact(s.id, "k", "key", "v").unwrap();
        let n = store.memory_fact_count(s.id).unwrap();
        eprintln!("PROBE count={n}");
        let (rows, more) = store.memory_facts_page(s.id, None, 10).unwrap();
        eprintln!("PROBE direct rows={} more={more}", rows.len());
        let mem = SessionMemory::new(store.clone(), s.id);
        mem.remember("k2", "key2", "v2").unwrap();
        let n2 = store.memory_fact_count(s.id).unwrap();
        eprintln!("PROBE count after remember={n2}");
        let (rows2, more2) = store.memory_facts_page(s.id, None, 10).unwrap();
        eprintln!(
            "PROBE direct rows after remember={} more={more2}",
            rows2.len()
        );
        let fs = mem.facts().unwrap();
        eprintln!("PROBE mem.facts len={}", fs.len());
        let f2 = mem.by_kind("k2").unwrap();
        eprintln!("PROBE by_kind len={}", f2.len());
        assert_eq!(fs.len(), 2);
    }

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
    fn empty_store_facts_page_carries_full_metadata() {
        let (_d, store, session) = fixture();
        let mem = SessionMemory::new(store.clone(), session);
        let p = mem.facts_page(None, 9).unwrap();
        assert!(p.facts.is_empty());
        assert!(!p.has_more);
        assert_eq!(p.cursor, None);
        assert_eq!(p.size, 9, "empty page still reports the applied size");
        assert_eq!(p.total_estimate, 0);
        // Hostile cursors: an empty final page, never an error.
        for hostile in [
            Some((i64::MIN, String::new(), String::new())),
            Some((-1, "z".into(), "z".into())),
        ] {
            let p = mem.facts_page(hostile.as_ref(), 9).unwrap();
            assert!(p.facts.is_empty());
            assert!(!p.has_more);
        }
    }

    #[test]
    fn facts_page_walk_covers_every_fact_exactly_once() {
        let (_d, store, session) = fixture();
        let mem = SessionMemory::new(store.clone(), session);
        for i in 0..25 {
            mem.remember("decision", &format!("k{i:03}"), &format!("v{i}"))
                .unwrap();
        }
        let full = mem.facts().unwrap();
        assert_eq!(full.len(), 25);
        // Deterministic ordering: newest first, ties broken by kind/key.
        let mut seen: Vec<(String, String)> = Vec::new();
        let mut cursor: Option<(i64, String, String)> = None;
        loop {
            let p = mem.facts_page(cursor.as_ref(), 8).unwrap();
            assert!(p.facts.len() <= 8, "never more than one page");
            assert_eq!(p.size, 8);
            if p.facts.is_empty() {
                assert!(!p.has_more, "empty page closes the walk");
                break;
            }
            for f in &p.facts {
                assert!(
                    !seen.contains(&(f.kind.clone(), f.key.clone())),
                    "duplicate fact {f:?}"
                );
                seen.push((f.kind.clone(), f.key.clone()));
            }
            if !p.has_more {
                assert_eq!(p.cursor, None);
                break;
            }
            cursor = p.cursor.clone();
        }
        let mut sorted = seen.clone();
        sorted.sort();
        let mut full_ids: Vec<(String, String)> = full
            .iter()
            .map(|f| (f.kind.clone(), f.key.clone()))
            .collect();
        full_ids.sort();
        assert_eq!(sorted, full_ids, "walk must cover every fact exactly once");
        assert!(full[0].updated_ms > 0, "updated_ms is real, not 0");
        // Cursor replay is deterministic.
        let a = mem.facts_page(None, 8).unwrap();
        let b = mem.facts_page(None, 8).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.total_estimate, 25);
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
