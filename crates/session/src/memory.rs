//! Durable task state and long-term memory facts.

use crate::handle::SessionHandle;
use crate::{json_bytes, SessionError, MAX_LEDGER_BYTES};

const MAX_FACT_KEY_BYTES: usize = 512;
const MAX_FACT_VALUE_BYTES: usize = 4096;

impl SessionHandle {
    /// Upsert one memory fact (kind/key uniquely identify it). Bounds are
    /// enforced before the write; empty keys are malformed.
    pub fn upsert_memory_fact(&self, kind: &str, key: &str, value: &str) -> kilop_core::Result<()> {
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

    pub fn memory_facts(&self) -> kilop_core::Result<Vec<(String, String, String)>> {
        self.manager
            .store()
            .memory_facts(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    /// The durable task ledger (goal, completed/open steps, decisions, ...).
    /// The journal is the source of truth for *what happened*; the ledger is
    /// the compact structured projection that survives compaction.
    pub fn get_task_ledger(&self) -> kilop_core::Result<Option<serde_json::Value>> {
        self.manager
            .store()
            .get_task_ledger(self.id)
            .map_err(|e| crate::map_store_err(e).into())
    }

    pub fn put_task_ledger(&self, ledger: serde_json::Value) -> kilop_core::Result<()> {
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
            "goal": "implement kilop-session",
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
}
