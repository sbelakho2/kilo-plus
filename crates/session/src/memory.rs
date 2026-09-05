//! Durable task state and long-term memory facts.

use crate::handle::SessionHandle;
use crate::{json_bytes, SessionError, MAX_LEDGER_BYTES};

const MAX_FACT_KEY_BYTES: usize = 512;
const MAX_FACT_VALUE_BYTES: usize = 4096;
/// Bounded page size for memory-fact paging (paging is fundamental).
pub const MAX_FACT_PAGE_SIZE: i64 = 200;

/// One deterministic page of memory facts with explicit paging metadata:
/// `{size, cursor, has_more, total_estimate}`. Facts are ordered
/// newest-first by `(updated_ms DESC, kind DESC, key DESC)`; the cursor is
/// the `(updated_ms, kind, key)` position after this page — pass it back
/// verbatim. Replaying a cursor returns the same window.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryFactsPage {
    /// (kind, key, value), newest-first.
    pub facts: Vec<(String, String, String)>,
    /// The applied page size (requested limit after clamping).
    pub size: i64,
    /// Cursor for the next older page; `None` on the final page.
    pub cursor: Option<(i64, String, String)>,
    /// True when at least one older page exists.
    pub has_more: bool,
    /// Exact fact count for the session (cheap; the facts table is small).
    pub total_estimate: i64,
}

impl SessionHandle {
    /// Upsert one memory fact (kind/key uniquely identify it). Bounds are
    /// enforced before the write; empty keys are malformed.
    pub fn upsert_memory_fact(
        &self,
        kind: &str,
        key: &str,
        value: &str,
    ) -> faktor_core::Result<()> {
        if kind.is_empty() || key.is_empty() {
            return Err(
                SessionError::Malformed("fact kind and key must be non-empty".into()).into(),
            );
        }
        if kind.len() > 64 || key.len() > MAX_FACT_KEY_BYTES {
            return Err(SessionError::Oversized("fact kind/key too long".into()).into());
        }
        if value.len() > MAX_FACT_VALUE_BYTES {
            return Err(SessionError::Oversized(format!(
                "fact value of {} bytes exceeds MAX_FACT_VALUE_BYTES",
                value.len()
            ))
            .into());
        }
        self.manager
            .store()
            .upsert_memory_fact(self.id, kind, key, value)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn memory_facts(&self) -> faktor_core::Result<Vec<(String, String, String)>> {
        self.manager
            .store()
            .memory_facts(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// One deterministic page of memory facts (see [`MemoryFactsPage`]).
    /// Bounded: never loads more than one page + one probe row, and the
    /// probe runs inside the same query.
    pub fn memory_facts_page(
        &self,
        after: Option<&(i64, String, String)>,
        limit: i64,
    ) -> faktor_core::Result<MemoryFactsPage> {
        let limit = limit.clamp(1, MAX_FACT_PAGE_SIZE);
        let (rows, has_more) = self
            .manager
            .store()
            .memory_facts_page(self.id, after, limit as u64)
            .map_err(crate::map_store_err)?;
        let cursor = if has_more {
            rows.last()
                .map(|r| (r.updated_ms, r.kind.clone(), r.key.clone()))
        } else {
            None
        };
        let total_estimate = self
            .manager
            .store()
            .memory_fact_count(self.id)
            .map_err(crate::map_store_err)?;
        Ok(MemoryFactsPage {
            facts: rows.into_iter().map(|r| (r.kind, r.key, r.value)).collect(),
            size: limit,
            cursor,
            has_more,
            total_estimate,
        })
    }

    /// The durable task ledger (goal, completed/open steps, decisions, ...).
    /// The journal is the source of truth for *what happened*; the ledger is
    /// the compact structured projection that survives compaction.
    pub fn get_task_ledger(&self) -> faktor_core::Result<Option<serde_json::Value>> {
        self.manager
            .store()
            .get_task_ledger(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn put_task_ledger(&self, ledger: serde_json::Value) -> faktor_core::Result<()> {
        if json_bytes(&ledger) > MAX_LEDGER_BYTES {
            return Err(SessionError::Oversized(format!(
                "ledger of {} bytes exceeds MAX_LEDGER_BYTES",
                json_bytes(&ledger)
            ))
            .into());
        }
        self.manager
            .store()
            .put_task_ledger(self.id, ledger)
            .map_err(|e| crate::map_store_err(e).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};

    #[test]
    fn memory_facts_upsert_and_bounds() {
        let (_d, m) = test_manager();
        let s = session(&m);
        s.upsert_memory_fact("preference", "language", "rust")
            .unwrap();
        s.upsert_memory_fact("preference", "language", "go")
            .unwrap();
        let facts = s.memory_facts().unwrap();
        assert_eq!(facts.len(), 1, "upsert replaces by (kind, key)");
        assert_eq!(
            facts[0],
            (
                "preference".to_string(),
                "language".to_string(),
                "go".to_string()
            )
        );
        // Empty kind/key are malformed; oversized values are rejected.
        assert!(s.upsert_memory_fact("", "k", "v").is_err());
        assert!(s.upsert_memory_fact("k", "", "v").is_err());
        assert!(s
            .upsert_memory_fact("k", "k", &"v".repeat(MAX_FACT_VALUE_BYTES + 1))
            .is_err());
        assert_eq!(
            s.memory_facts().unwrap().len(),
            1,
            "rejected facts leave no trace"
        );
    }

    #[test]
    fn task_ledger_roundtrip_and_bound() {
        let (_d, m) = test_manager();
        let s = session(&m);
        assert_eq!(s.get_task_ledger().unwrap(), None);
        let ledger = serde_json::json!({
            "goal": "implement faktor-session",
            "completed_steps": ["journal"],
            "open_steps": ["recovery"]
        });
        s.put_task_ledger(ledger.clone()).unwrap();
        assert_eq!(s.get_task_ledger().unwrap(), Some(ledger));
        // Replace, don't accumulate.
        s.put_task_ledger(serde_json::json!({"goal": "v2"}))
            .unwrap();
        assert_eq!(
            s.get_task_ledger().unwrap(),
            Some(serde_json::json!({"goal": "v2"}))
        );
        // Oversized ledgers are rejected before the write.
        let huge = serde_json::json!({ "blob": "x".repeat(MAX_LEDGER_BYTES + 1) });
        assert!(s.put_task_ledger(huge).is_err());
        assert_eq!(
            s.get_task_ledger().unwrap(),
            Some(serde_json::json!({"goal": "v2"}))
        );
    }

    #[test]
    fn memory_and_ledger_are_per_session() {
        let (_d, m) = test_manager();
        let s1 = session(&m);
        let s2 = {
            let ws = m.create_workspace("/w2").unwrap();
            m.create_session(ws, "t2", "p", "m").unwrap()
        };
        s1.upsert_memory_fact("k", "key", "s1").unwrap();
        s2.upsert_memory_fact("k", "key", "s2").unwrap();
        assert_eq!(s1.memory_facts().unwrap()[0].2, "s1");
        assert_eq!(s2.memory_facts().unwrap()[0].2, "s2");
        s1.put_task_ledger(serde_json::json!({"owner": "s1"}))
            .unwrap();
        assert!(s2.get_task_ledger().unwrap().is_none());
    }

    #[test]
    fn empty_facts_store_page_carries_full_metadata() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let page = s.memory_facts_page(None, 17).unwrap();
        assert!(page.facts.is_empty());
        assert!(!page.has_more);
        assert_eq!(page.cursor, None);
        assert_eq!(page.size, 17, "empty page still reports the page size");
        assert_eq!(page.total_estimate, 0);
        // Hostile cursors are empty final pages, never an error.
        for hostile in [
            Some((i64::MIN, String::new(), String::new())),
            Some((i64::MIN, "z".into(), "z".into())),
            Some((-5, "a".into(), "b".into())),
        ] {
            let p = s.memory_facts_page(hostile.as_ref(), 7).unwrap();
            assert!(p.facts.is_empty());
            assert!(!p.has_more);
            assert_eq!(p.cursor, None);
        }
    }

    #[test]
    fn facts_page_walk_reaches_every_fact_exactly_once() {
        let (_d, m) = test_manager();
        let s = session(&m);
        for i in 0..23 {
            // A shared kind exercises (updated_ms, kind, key) tie-breaks:
            // same-millisecond rows must still cut deterministically.
            s.upsert_memory_fact("decision", &format!("k{i:03}"), &format!("v{i}"))
                .unwrap();
        }
        // The legacy full read agrees with the walk's total.
        let full: Vec<(String, String, String)> = s.memory_facts().unwrap();
        assert_eq!(full.len(), 23);
        let mut seen: Vec<(String, String, String)> = Vec::new();
        let mut cursor: Option<(i64, String, String)> = None;
        loop {
            let p = s.memory_facts_page(cursor.as_ref(), 7).unwrap();
            assert!(p.facts.len() <= 7, "never more than one page");
            assert_eq!(p.size, 7, "every page echoes the applied size");
            if p.facts.is_empty() {
                assert!(!p.has_more, "empty page must close the walk");
                break;
            }
            for f in &p.facts {
                assert!(!seen.contains(f), "duplicate fact in walk: {f:?}");
                seen.push(f.clone());
            }
            if !p.has_more {
                // Final page: cursor must be None so clients stop.
                assert_eq!(p.cursor, None);
                break;
            }
            cursor = p.cursor.clone();
        }
        assert_eq!(seen.len(), full.len(), "walk must cover the full set");
        let mut sorted = seen.clone();
        sorted.sort();
        let mut full_sorted = full.clone();
        full_sorted.sort();
        assert_eq!(sorted, full_sorted, "gap: walk missed a fact");
        // Replaying the first cursor yields the identical window again
        // (deterministic idempotence, never a duplicate page).
        let p1 = s.memory_facts_page(None, 7).unwrap();
        assert!(p1.has_more);
        let p1b = s.memory_facts_page(None, 7).unwrap();
        assert_eq!(p1, p1b, "same cursor, same page");
        assert_eq!(p1.total_estimate, 23);
    }

    #[test]
    fn facts_walk_under_concurrent_upserts_never_duplicates() {
        // Adversarial: a writer upserts facts WHILE a reader walks pages.
        // An upsert moves a row to the NEWEST end of the order, so the walk
        // must never see the same (kind, key) twice; rows never rewritten
        // during the walk are still covered exactly once (no gap over the
        // stable rows), and rows rewritten before their turn simply leave
        // the walk window instead of being emitted twice.
        let (_d, m) = test_manager();
        let s = session(&m);
        for i in 0..30 {
            s.upsert_memory_fact("seed", &format!("s{i:03}"), "v")
                .unwrap();
        }
        let mut seen: Vec<(String, String, String)> = Vec::new();
        let mut cursor: Option<(i64, String, String)> = None;
        let mut page_no = 0;
        loop {
            let p = s.memory_facts_page(cursor.as_ref(), 6).unwrap();
            // Between pages, the writer interleaves: rewrite one old seed
            // row (moves it to the newest end) and add one new fact.
            s.upsert_memory_fact("seed", &format!("s{:03}", page_no * 2), "rewritten")
                .unwrap();
            s.upsert_memory_fact("late", &format!("n{page_no}"), "new")
                .unwrap();
            if p.facts.is_empty() {
                assert!(!p.has_more);
                break;
            }
            for f in &p.facts {
                assert!(
                    !seen.contains(f),
                    "a fact must never appear twice in one walk: {f:?}"
                );
                seen.push(f.clone());
            }
            if !p.has_more {
                break;
            }
            cursor = p.cursor.clone();
            page_no += 1;
        }
        // Every fact the walk saw is a real seed row (newly appended rows
        // land at the newest end and never leak into an ongoing walk).
        assert!(
            seen.iter().all(|(k, _key, v)| k == "seed" && v == "v"),
            "{seen:?}"
        );
        // No duplicate (kind, key): the walk's emitted identities are all
        // distinct regardless of the writer's interleaving.
        let mut ids: Vec<(&String, &String)> = seen.iter().map(|(k, key, _)| (k, key)).collect();
        ids.sort();
        let distinct: usize =
            ids.windows(2).filter(|w| w[0] != w[1]).count() + usize::from(!ids.is_empty());
        assert_eq!(distinct, ids.len(), "duplicate fact identity in walk");
        // No gap over the stable window: the odd seed rows are NEVER
        // rewritten by the writer, so each must appear exactly once.
        for i in (1..30).step_by(2) {
            let key = format!("s{i:03}");
            assert_eq!(
                seen.iter()
                    .filter(|(k, k2, _)| k == "seed" && *k2 == key)
                    .count(),
                1,
                "stable fact {key} must appear exactly once (no gap)"
            );
        }
        // The walk always terminates, and replaying the cursor of the page
        // that closed it re-yields that same page (deterministic replay,
        // never a duplicate or a rewind).
        if let Some(c) = &cursor {
            let probe: Option<(i64, String, String)> = Some((c.0, c.1.clone(), c.2.clone()));
            let replay = s.memory_facts_page(probe.as_ref(), 6).unwrap();
            let again = s.memory_facts_page(probe.as_ref(), 6).unwrap();
            assert_eq!(replay, again, "cursor replay is deterministic");
        }
    }
}
