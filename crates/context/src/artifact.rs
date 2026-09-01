//! Artifact storage: historical command output, old source reads, enormous
//! test logs live in the CAS. The prompt carries `artifact://<hash>` refs
//! plus short summaries — never the unbounded content itself.

use std::sync::Arc;

use kilop_cas::Cas;
use kilop_core::hash::FileHash;
use kilop_core::id::SessionId;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactRef {
    /// Present when content fit inside the inline cap.
    pub inline: Option<String>,
    /// `artifact://<hash>` when stored in the CAS.
    pub artifact: Option<String>,
    pub summary: String,
    pub size: usize,
}

impl ArtifactRef {
    pub fn render(&self) -> String {
        if let Some(a) = &self.artifact {
            format!("[artifact: {a}] {}", self.summary)
        } else {
            self.inline.clone().unwrap_or_default()
        }
    }
}

pub struct ArtifactWriter {
    cas: Arc<Cas>,
    session: SessionId,
}

impl ArtifactWriter {
    pub fn new(cas: Arc<Cas>, session: SessionId) -> Self {
        Self { cas, session }
    }

    pub fn session(&self) -> SessionId {
        self.session
    }

    /// Store bytes: inline if within `max_inline`, otherwise CAS.
    pub fn store(
        &self,
        _kind: &str,
        bytes: &[u8],
        max_inline: usize,
    ) -> kilop_core::Result<ArtifactRef> {
        let size = bytes.len();
        if size <= max_inline {
            let text = String::from_utf8_lossy(bytes).to_string();
            let summary = make_summary(&text);
            return Ok(ArtifactRef {
                inline: Some(text),
                artifact: None,
                summary,
                size,
            });
        }
        let hash = self.cas.put(bytes).map_err(cas_err)?;
        let text = String::from_utf8_lossy(bytes);
        Ok(ArtifactRef {
            inline: None,
            artifact: Some(format!("artifact://{}", hash.to_hex())),
            summary: make_summary(&text),
            size,
        })
    }

    /// Read an artifact back by its `artifact://<hash>` reference.
    pub fn read_ref(&self, artifact_ref: &str) -> kilop_core::Result<Vec<u8>> {
        let hash_hex = artifact_ref
            .strip_prefix("artifact://")
            .ok_or_else(|| kilop_core::error::Error::malformed("bad artifact reference"))?;
        let hash = FileHash::from_hex(hash_hex)
            .ok_or_else(|| kilop_core::error::Error::malformed("bad artifact hash"))?;
        self.cas.get(hash).map_err(cas_err)
    }

    /// Verify an artifact reference resolves and hashes correctly.
    pub fn verify(&self, artifact_ref: &str) -> kilop_core::Result<bool> {
        let bytes = self.read_ref(artifact_ref)?;
        let hash = FileHash::from(blake3::hash(&bytes).into());
        let hash_hex = artifact_ref
            .strip_prefix("artifact://")
            .ok_or_else(|| kilop_core::error::Error::malformed("bad artifact reference"))?;
        Ok(hash.to_hex() == hash_hex)
    }
}

fn make_summary(text: &str) -> String {
    let first = text.lines().next().unwrap_or("").trim();
    let summarized = truncate(first, 200);
    if summarized.is_empty() {
        "[empty output]".to_string()
    } else {
        summarized.to_string()
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

fn cas_err(e: kilop_cas::CasError) -> kilop_core::error::Error {
    kilop_core::error::Error::new(kilop_core::error::ErrorKind::Store, format!("cas: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn fixture() -> (tempfile::TempDir, Arc<Cas>, SessionId) {
        let dir = tempdir().unwrap();
        let cas = Arc::new(Cas::open(dir.path().join("cas")).unwrap());
        (dir, cas, SessionId::new(1))
    }

    #[test]
    fn small_content_stays_inline() {
        let (_d, cas, session) = fixture();
        let w = ArtifactWriter::new(cas.clone(), session);
        let r = w.store("command_output", b"ok", 4096).unwrap();
        assert!(r.inline.is_some());
        assert!(r.artifact.is_none());
        assert_eq!(r.render(), "ok");
        assert_eq!(r.size, 2);
    }

    #[test]
    fn overflow_goes_to_cas_and_roundtrips_exactly() {
        let (_d, cas, session) = fixture();
        let w = ArtifactWriter::new(cas.clone(), session);
        let blob: Vec<u8> = (0..20_000).map(|i| (i % 251) as u8).collect();
        let r = w.store("log", &blob, 4096).unwrap();
        assert!(r.inline.is_none());
        let artifact = r.artifact.as_ref().expect("artifact ref");
        assert!(artifact.starts_with("artifact://"));
        assert_eq!(artifact.len(), 11 + 64);
        let back = w.read_ref(artifact).unwrap();
        assert_eq!(back, blob);
        assert!(w.verify(artifact).unwrap());
    }

    #[test]
    fn tampered_cas_blob_detected_by_verify() {
        let (_d, cas, session) = fixture();
        let w = ArtifactWriter::new(cas.clone(), session);
        let r = w.store("log", &vec![1u8; 9000], 4096).unwrap();
        let artifact = r.artifact.unwrap();
        // Corrupt the stored blob behind the writer's back: read its bytes
        // from the CAS, then write garbage at the same address by storing
        // the blob path via the shard layout (first two hex chars as dir).
        let hash_hex = artifact.strip_prefix("artifact://").unwrap();
        let path = cas.root().join(&hash_hex[..2]).join(&hash_hex[2..]);
        std::fs::write(path, b"corrupted").unwrap();
        assert!(w.read_ref(&artifact).is_err(), "corruption must error");
        assert!(w.verify(&artifact).is_err() || !w.verify(&artifact).unwrap());
    }

    #[test]
    fn identical_artifacts_dedupe() {
        let (_d, cas, session) = fixture();
        let w = ArtifactWriter::new(cas.clone(), session);
        let blob = vec![7u8; 8000];
        let a = w.store("log", &blob, 100).unwrap();
        let b = w.store("log", &blob, 100).unwrap();
        assert_eq!(a.artifact, b.artifact);
        assert_eq!(
            cas.blob_count(),
            1,
            "ten checkpoints of one file = one copy"
        );
    }

    #[test]
    fn malformed_refs_are_rejected() {
        let (_d, cas, session) = fixture();
        let w = ArtifactWriter::new(cas.clone(), session);
        assert!(w.read_ref("artifact://zz").is_err());
        assert!(w.read_ref("https://evil.com/x").is_err());
        assert!(w.read_ref("").is_err());
        assert!(w.read_ref("artifact://").is_err());
    }

    #[test]
    fn empty_and_binary_artifacts() {
        let (_d, cas, session) = fixture();
        let w = ArtifactWriter::new(cas.clone(), session);
        let r = w.store("cmd", b"", 10).unwrap();
        assert_eq!(r.render(), "");
        // Binary garbage > inline cap: summary never panics on invalid UTF-8.
        let bytes: Vec<u8> = (0..=255).cycle().take(10_000).collect();
        let r = w.store("cmd", &bytes, 100).unwrap();
        assert!(r.artifact.is_some());
        assert!(!r.summary.is_empty());
        assert!(w.verify(r.artifact.as_ref().unwrap()).unwrap());
    }

    #[test]
    fn summary_is_bounded() {
        let (_d, cas, session) = fixture();
        let w = ArtifactWriter::new(cas.clone(), session);
        let long = format!("line1\n{}", "y".repeat(10_000));
        let r = w.store("cmd", long.as_bytes(), 4096).unwrap();
        assert!(
            r.summary.len() <= 204,
            "summary bounded: {}",
            r.summary.len()
        );
        assert!(r.summary.starts_with("line1"));
    }
}
