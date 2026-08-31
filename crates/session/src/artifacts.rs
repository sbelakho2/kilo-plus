//! Artifacts: bounded blobs in the CAS with a summary row in the store.
//! Large command output never lands in SQLite or the transcript — it lands
//! here, and the model sees a summary plus an `artifact://` reference.

use std::collections::HashMap;
use std::sync::Mutex;

use kilop_core::hash::FileHash;

use crate::handle::SessionHandle;
use crate::{
    MAX_ARTIFACT_BYTES, MAX_ARTIFACT_SUMMARY_BYTES, SessionError,
};

/// In-memory map of artifact sizes (the store row keeps the size but the
/// store API does not expose it). Used to reject oversized reads *before*
/// loading the blob.
#[derive(Debug, Default)]
pub(crate) struct ArtifactSizes {
    inner: Mutex<HashMap<FileHash, usize>>,
}

impl ArtifactSizes {
    pub fn record(&self, hash: FileHash, size: usize) {
        self.inner.lock().expect("artifact sizes poisoned").insert(hash, size);
    }

    pub fn size_of(&self, hash: FileHash) -> Option<usize> {
        self.inner.lock().expect("artifact sizes poisoned").get(&hash).copied()
    }
}

impl SessionHandle {
    /// Store a bounded artifact blob in the CAS and record its summary row.
    /// Returns the content address. Payloads over `MAX_ARTIFACT_BYTES` are
    /// rejected before hashing or writing anything.
    pub fn put_artifact(
        &self,
        kind: &str,
        bytes: &[u8],
        summary: &str,
    ) -> kilop_core::Result<FileHash> {
        if kind.is_empty() || kind.len() > 64 {
            return Err(SessionError::Malformed(format!("invalid artifact kind {kind:?}")).into());
        }
        if summary.len() > MAX_ARTIFACT_SUMMARY_BYTES {
            return Err(SessionError::Oversized(format!(
                "artifact summary of {} bytes exceeds MAX_ARTIFACT_SUMMARY_BYTES",
                summary.len()
            ))
            .into());
        }
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(SessionError::Oversized(format!(
                "artifact of {} bytes exceeds MAX_ARTIFACT_BYTES",
                bytes.len()
            ))
            .into());
        }
        let hash = self
            .manager
            .cas()
            .put(bytes)
            .map_err(SessionError::from)?;
        self.manager
            .store()
            .put_artifact(
                self.id,
                kind,
                &hash.to_hex(),
                summary,
                bytes.len() as i64,
            )
            .map_err(crate::map_store_err)?;
        self.manager.artifact_sizes.record(hash, bytes.len());
        Ok(hash)
    }

    /// Read an artifact back, verifying its hash. `max_bytes` bounds the read:
    /// tracked artifacts are rejected before any I/O; untracked artifacts
    /// (post-restart) are still hard-capped by `MAX_ARTIFACT_BYTES` from the
    /// put path.
    pub fn artifact_blob(&self, hash: FileHash, max_bytes: usize) -> kilop_core::Result<Vec<u8>> {
        if let Some(size) = self.manager.artifact_sizes.size_of(hash) {
            if size > max_bytes {
                return Err(SessionError::Oversized(format!(
                    "artifact {hash} is {size} bytes, limit {max_bytes}"
                ))
                .into());
            }
        }
        let bytes = self
            .manager
            .cas()
            .get(hash)
            .map_err(SessionError::from)?;
        if bytes.len() > max_bytes {
            return Err(SessionError::Oversized(format!(
                "artifact {hash} is {} bytes, limit {max_bytes}",
                bytes.len()
            ))
            .into());
        }
        Ok(bytes)
    }

    /// The durable (summary, kind) row for an artifact.
    pub fn artifact_summary(
        &self,
        hash: FileHash,
    ) -> kilop_core::Result<Option<(String, String)>> {
        self.manager
            .store()
            .artifact(&hash.to_hex())
            .map_err(|e| crate::map_store_err(e).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};

    #[test]
    fn artifact_roundtrip_dedup_and_bounded_reads() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let blob = b"hello artifact world".to_vec();
        let h1 = s.put_artifact("tool_output", &blob, "greeting").unwrap();
        let h2 = s.put_artifact("tool_output", &blob, "greeting again").unwrap();
        assert_eq!(h1, h2, "identical content addresses identically");
        assert_eq!(s.artifact_blob(h1, 1 << 20).unwrap(), blob);
        // A smaller bound on a tracked artifact fails before reading.
        let err = s.artifact_blob(h1, 5).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Oversized);
        // Summary row is durable.
        let (summary, kind) = s.artifact_summary(h1).unwrap().unwrap();
        assert_eq!(summary, "greeting");
        assert_eq!(kind, "tool_output");
        // Missing artifact is NotFound.
        let missing = FileHash::from([9; 32]);
        let err = s.artifact_blob(missing, 1 << 20).unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::NotFound);
        assert!(s.artifact_summary(missing).unwrap().is_none());
    }

    #[test]
    fn oversized_artifact_rejected_before_write() {
        let (_d, m) = test_manager();
        let s = session(&m);
        let big = vec![0u8; MAX_ARTIFACT_BYTES + 1];
        let err = s.put_artifact("tool_output", &big, "too big").unwrap_err();
        assert_eq!(err.kind, kilop_core::ErrorKind::Oversized);
        assert!(s.artifact_summary(FileHash::from([0; 32])).unwrap().is_none());
        // Bad kinds and oversized summaries are malformed/oversized, no I/O.
        assert!(s.put_artifact("", b"x", "s").is_err());
        assert!(s.put_artifact(&"k".repeat(65), b"x", "s").is_err());
        assert!(s.put_artifact("k", b"x", &"s".repeat(MAX_ARTIFACT_SUMMARY_BYTES + 1)).is_err());
    }

    #[test]
    fn corrupted_cas_blob_is_detected_not_served() {
        let (dir, m) = test_manager();
        let s = session(&m);
        let h = s.put_artifact("log", b"integrity matters", "line").unwrap();
        // Tamper with the blob on disk; CAS must refuse to serve it.
        let path = dir.path().join("cas").join(h.cas_path());
        std::fs::write(&path, b"corrupted").unwrap();
        assert!(s.artifact_blob(h, 1 << 20).is_err());
    }
}
