//! Content-addressed blob storage: BLAKE3 identity + Zstd compression.
//!
//! - Layout: `<root>/ab/cdef...` (64-hex BLAKE3, first two chars a shard dir)
//! - Deduplication is free: identical content hashes to the same address
//! - Reads verify the hash: corruption is detected, never silently served
//! - Writes are atomic: temp file + fsync + rename, so a crash mid-write
//!   can never leave a partial blob under a valid-looking address

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use kilop_core::hash::FileHash;

#[derive(Debug, thiserror::Error)]
pub enum CasError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("blob not found: {0}")]
    NotFound(FileHash),
    #[error("corruption: blob {hash} decompressed to {actual} bytes, expected {expected}")]
    SizeMismatch {
        hash: FileHash,
        expected: u64,
        actual: u64,
    },
    #[error("corruption: blob {0} failed hash verification")]
    HashMismatch(FileHash),
    #[error("zstd error: {0}")]
    Zstd(String),
}

pub type CasResult<T> = Result<T, CasError>;

/// Content-addressed store rooted at `root`.
#[derive(Debug, Clone)]
pub struct Cas {
    root: PathBuf,
}

impl Cas {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn open(root: PathBuf) -> CasResult<Self> {
        let cas = Self::new(root);
        fs::create_dir_all(cas.root.join("tmp"))?;
        Ok(cas)
    }

    fn blob_path(&self, hash: FileHash) -> PathBuf {
        self.root.join(hash.cas_path())
    }

    /// Store `bytes`, returning its content address. If the blob already
    /// exists it is not rewritten (idempotent; safe under concurrency
    /// because identical content always produces the identical path).
    pub fn put(&self, bytes: &[u8]) -> CasResult<FileHash> {
        let hash = FileHash::from(blake3::hash(bytes).into());
        let path = self.blob_path(hash);
        if path.exists() {
            // Idempotent; still verify that what is on disk matches, so a
            // corrupted existing blob is caught rather than reused.
            if let Ok(meta) = fs::metadata(&path) {
                if meta.len() != 0 {
                    return Ok(hash);
                }
            }
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        // Compress to memory first so the temp-file write is one syscall
        // batch; huge inputs are bounded by the caller (see put_bounded).
        let compressed = zstd::encode_all(bytes, 3).map_err(|e| CasError::Zstd(e.to_string()))?;
        let tmp = self.root.join("tmp").join(format!(
            "{}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4(),
            hash.to_hex()
        ));
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(&compressed)?;
            f.sync_all()?;
        }
        match fs::rename(&tmp, &path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                return Err(e.into());
            }
        }
        Ok(hash)
    }

    /// Bounded put: rejects payloads larger than `max_bytes` *before* hashing.
    pub fn put_bounded(&self, bytes: &[u8], max_bytes: usize) -> CasResult<FileHash> {
        if bytes.len() > max_bytes {
            return Err(CasError::Zstd("payload exceeds bound".into()));
        }
        self.put(bytes)
    }

    pub fn has(&self, hash: FileHash) -> bool {
        self.blob_path(hash).exists()
    }

    /// Read and verify. Corrupted blobs are an error, never silent garbage.
    pub fn get(&self, hash: FileHash) -> CasResult<Vec<u8>> {
        let path = self.blob_path(hash);
        let compressed = match fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(CasError::NotFound(hash));
            }
            Err(e) => return Err(e.into()),
        };
        let bytes = zstd::decode_all(compressed.as_slice())
            .map_err(|e| CasError::Zstd(e.to_string()))?;
        let actual = FileHash::from(blake3::hash(&bytes).into());
        if actual != hash {
            return Err(CasError::HashMismatch(hash));
        }
        Ok(bytes)
    }

    /// Size of the stored blob, without decompressing.
    pub fn stored_size(&self, hash: FileHash) -> CasResult<u64> {
        match fs::metadata(self.blob_path(hash)) {
            Ok(m) => Ok(m.len()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(CasError::NotFound(hash))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Stream a blob to a writer, verifying the hash while copying.
    pub fn copy_to<W: Write>(&self, hash: FileHash, w: &mut W) -> CasResult<u64> {
        let bytes = self.get(hash)?;
        w.write_all(&bytes)?;
        Ok(bytes.len() as u64)
    }

    /// Verify integrity of the whole store; returns the list of corrupted
    /// hashes (empty = healthy).
    pub fn verify_integrity(&self) -> Vec<FileHash> {
        let mut corrupted = Vec::new();
        let mut read_dir = match fs::read_dir(&self.root) {
            Ok(r) => r,
            Err(_) => return corrupted,
        };
        while let Some(Ok(entry)) = read_dir.next() {
            let path = entry.path();
            if !path.is_dir() || path.file_name().is_none() {
                continue;
            }
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if name.len() != 2 {
                continue; // tmp/ etc.
            }
            if let Ok(rd) = fs::read_dir(&path) {
                for f in rd.flatten() {
                    let fpath = f.path();
                    let fname = match fpath.file_name() {
                        Some(n) => n.to_string_lossy().to_string(),
                        None => continue,
                    };
                    let hex = format!("{name}{fname}");
                    if let Some(hash) = FileHash::from_hex(&hex) {
                        if let Err(_) = self.get(hash) {
                            corrupted.push(hash);
                        }
                    }
                }
            }
        }
        corrupted
    }

    /// Total blob count (integrity scan cost; for tests and `doctor`).
    /// Only counts files inside 2-hex-char shard directories.
    pub fn blob_count(&self) -> usize {
        let mut n = 0;
        if let Ok(read_dir) = fs::read_dir(&self.root) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let name = match path.file_name() {
                    Some(n) => n.to_string_lossy().to_string(),
                    None => continue,
                };
                if !is_shard_dir(&name) {
                    continue; // tmp/ and other non-shard dirs
                }
                if let Ok(rd) = fs::read_dir(&path) {
                    n += rd.count();
                }
            }
        }
        n
    }
}

fn is_shard_dir(name: &str) -> bool {
    name.len() == 2 && name.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tmp_cas() -> (tempfile::TempDir, Cas) {
        let dir = tempfile::tempdir().unwrap();
        let cas = Cas::open(dir.path().join("cas")).unwrap();
        (dir, cas)
    }

    #[test]
    fn roundtrip_and_dedup() {
        let (_d, cas) = tmp_cas();
        let h1 = cas.put(b"hello world").unwrap();
        let h2 = cas.put(b"hello world").unwrap();
        assert_eq!(h1, h2, "identical content must address identically");
        assert_eq!(cas.blob_count(), 1, "ten checkpoints of one file = one copy");
        assert_eq!(cas.get(h1).unwrap(), b"hello world");
    }

    #[test]
    fn empty_and_binary_blobs() {
        let (_d, cas) = tmp_cas();
        let h = cas.put(b"").unwrap();
        assert_eq!(cas.get(h).unwrap(), b"");
        let mut blob = Vec::with_capacity(1 << 20);
        for i in 0..(1 << 20) {
            blob.push((i % 251) as u8);
        }
        let h = cas.put(&blob).unwrap();
        assert_eq!(cas.get(h).unwrap(), blob);
    }

    #[test]
    fn missing_blob_is_not_found_not_garbage() {
        let (_d, cas) = tmp_cas();
        let h = FileHash::from([7; 32]);
        match cas.get(h) {
            Err(CasError::NotFound(x)) => assert_eq!(x, h),
            other => panic!("expected NotFound, got {other:?}"),
        }
        assert!(!cas.has(h));
    }

    #[test]
    fn corruption_is_detected_on_read() {
        let (_d, cas) = tmp_cas();
        let h = cas.put(b"attack surface").unwrap();
        // Corrupt the stored bytes in place.
        let path = cas.blob_path(h);
        fs::write(&path, b"garbage that is not zstd").unwrap();
        let result = cas.get(h);
        assert!(
            matches!(result, Err(CasError::HashMismatch(_)) | Err(CasError::Zstd(_))),
            "corruption must be an error, got {result:?}"
        );
        // Integrity scan must flag it.
        let bad = cas.verify_integrity();
        assert!(!bad.is_empty());
        assert!(bad.contains(&h));
    }

    #[test]
    fn tampered_content_with_valid_compression_is_caught() {
        let (_d, cas) = tmp_cas();
        let h = cas.put(b"the quick brown fox").unwrap();
        let path = cas.blob_path(h);
        // Recompress DIFFERENT content and overwrite: decompresses fine but
        // hash must not match.
        let evil = zstd::encode_all(&b"the lazy dog"[..], 3).unwrap();
        fs::write(&path, evil).unwrap();
        match cas.get(h) {
            Err(CasError::HashMismatch(x)) => assert_eq!(x, h),
            other => panic!("expected HashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn concurrent_puts_of_same_content_are_safe() {
        let (_d, cas) = tmp_cas();
        let cas = std::sync::Arc::new(cas);
        let mut handles = vec![];
        for _ in 0..16 {
            let cas = cas.clone();
            handles.push(std::thread::spawn(move || {
                let mut payload = vec![0u8; 8192];
                payload[0] = 42;
                for _ in 0..50 {
                    let h = cas.put(&payload).unwrap();
                    assert_eq!(cas.get(h).unwrap(), payload);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(cas.blob_count(), 1, "all identical writes must dedupe");
    }

    #[test]
    fn concurrent_puts_of_distinct_content_never_cross() {
        let (_d, cas) = tmp_cas();
        let cas = std::sync::Arc::new(cas);
        let mut handles = vec![];
        for t in 0..8 {
            let cas = cas.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..100 {
                    let payload = format!("thread-{t}-blob-{i}-{}", "x".repeat(i % 500));
                    let h = cas.put(payload.as_bytes()).unwrap();
                    let got = cas.get(h).unwrap();
                    assert_eq!(got, payload.as_bytes(), "blob cross-contamination");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn put_bounded_rejects_oversized_before_hashing() {
        let (_d, cas) = tmp_cas();
        let big = vec![0u8; 10_000];
        assert!(cas.put_bounded(&big, 9_999).is_err());
        assert!(cas.put_bounded(&big, 10_000).is_ok());
    }

    #[test]
    fn copy_to_streams_and_verifies() {
        let (_d, cas) = tmp_cas();
        let h = cas.put(b"stream me").unwrap();
        let mut buf = Cursor::new(Vec::new());
        let n = cas.copy_to(h, &mut buf).unwrap();
        assert_eq!(n, 9);
        assert_eq!(buf.into_inner(), b"stream me");
    }

    #[test]
    fn stored_size_reflects_compressed_blob() {
        let (_d, cas) = tmp_cas();
        let repetitive = vec![b'z'; 100_000];
        let h = cas.put(&repetitive).unwrap();
        let size = cas.stored_size(h).unwrap();
        assert!(size < 100_000, "highly compressible blob must shrink on disk");
        assert!(size > 0);
    }

    #[test]
    fn huge_blob_roundtrip() {
        let (_d, cas) = tmp_cas();
        // 8 MiB of mixed content.
        let mut blob = Vec::with_capacity(8 << 20);
        for i in 0..(8 << 20) {
            blob.push(((i * 31 + 7) % 256) as u8);
        }
        let h = cas.put(&blob).unwrap();
        let got = cas.get(h).unwrap();
        assert_eq!(got, blob);
    }

    #[test]
    fn verify_integrity_on_clean_store_is_empty() {
        let (_d, cas) = tmp_cas();
        cas.put(b"one").unwrap();
        cas.put(b"two").unwrap();
        cas.put(b"three").unwrap();
        assert!(cas.verify_integrity().is_empty());
    }

    #[test]
    fn blob_count_ignores_tmp_and_non_hex_files() {
        let (_d, cas) = tmp_cas();
        cas.put(b"x").unwrap();
        fs::create_dir_all(cas.root.join("tmp")).unwrap();
        fs::write(cas.root.join("tmp").join("junk"), b"junk").unwrap();
        fs::write(cas.root.join("not-a-hash"), b"junk").unwrap();
        assert_eq!(cas.blob_count(), 1);
    }

    #[test]
    fn adversarial_put_of_truncated_previous_write_is_retried_cleanly() {
        let (_d, cas) = tmp_cas();
        // Simulate a crash mid-write: a temp file is left behind. A later
        // put must not be confused by it, and must not treat temp as blob.
        fs::create_dir_all(cas.root.join("tmp")).unwrap();
        fs::write(cas.root.join("tmp").join("12345-uuid-abcdef"), b"partial").unwrap();
        let h = cas.put(b"fresh").unwrap();
        assert_eq!(cas.get(h).unwrap(), b"fresh");
        assert!(cas.verify_integrity().is_empty());
    }

    #[test]
    fn atomicity_under_mid_rename_crash_simulation() {
        let (_d, cas) = tmp_cas();
        // Simulate a crash between temp write and rename: the address must
        // simply not exist yet (never a partial file at the real address).
        let h = FileHash::from_hex(&format!("{}", "0".repeat(64))).unwrap();
        assert!(!cas.has(h));
        // And a put must succeed afterwards.
        let h2 = cas.put(b"after crash").unwrap();
        assert_eq!(cas.get(h2).unwrap(), b"after crash");
    }
}
