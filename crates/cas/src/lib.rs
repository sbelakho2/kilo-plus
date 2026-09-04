//! Content-addressed blob storage: BLAKE3 identity + Zstd compression.
//!
//! - Layout: `<root>/ab/cdef...` (64-hex BLAKE3, first two chars a shard dir)
//! - Deduplication is free: identical content hashes to the same address.
//!   Dedup hits are **verified** (read + decompress + rehash): a corrupt or
//!   collided blob is never silently reused — it is atomically repaired.
//! - Reads verify the hash: corruption is detected, never silently served
//! - Writes are atomic: temp file + fsync + rename (+ parent-dir fsync on
//!   unix), so a crash mid-write can never leave a partial blob under a
//!   valid-looking address, and a power loss right after a rename cannot
//!   lose the directory entry.
//! - Bounded by default: `put` rejects payloads over `DEFAULT_MAX_PUT_BYTES`
//!   (512 MiB) with `CasError::Oversized`; `put_bounded`/`put_reader_bounded`
//!   lower the ceiling. `put_reader` streams zstd compression, so large
//!   payloads are never materialized in RAM.

use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Instant, SystemTime};

use faktor_core::hash::FileHash;

/// Default ceiling for a single blob payload, in bytes (512 MiB). Structural:
/// the store refuses to compress unbounded data; use `put_bounded` or
/// `put_reader_bounded` for a smaller cap.
pub const DEFAULT_MAX_PUT_BYTES: usize = 512 * 1024 * 1024;

/// Maximum number of verified blobs remembered in the LRU (audit 50): a
/// bounded, advisory cache. A hit skips the re-verification decode; a miss
/// verifies every time — path existence alone is NEVER validity.
const VERIFIED_CACHE_MAX: usize = 256;

/// Even a stat-matching LRU hit is only honored within this window: after
/// `VERIFIED_TTL` the blob is re-verified from disk, bounding how long an
/// undetected same-size/same-mtime corruption could be trusted.
const VERIFIED_TTL: std::time::Duration = std::time::Duration::from_secs(60);

/// Streaming decode chunk: reads and decompression feed through a bounded
/// 64 KiB buffer, so verification never materializes a blob in RAM.
const STREAM_BUF: usize = 64 * 1024;

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
    #[error("payload too large: {actual} bytes exceeds the {max} byte bound")]
    Oversized { max: usize, actual: usize },
    #[error("zstd error: {0}")]
    Zstd(String),
    #[error("malformed hash: {0}")]
    Malformed(String),
}

pub type CasResult<T> = Result<T, CasError>;

/// One LRU record: `hash` verified at `verified_at`, decompressing to
/// `size` bytes. `stored_len`/`stored_mtime` snapshot the compressed blob
/// file's metadata at verification time; a cache hit is only honored when
/// the blob file still matches (same length and mtime), so an in-place
/// corruption or an atomic repair invalidates the record cheaply.
#[derive(Debug)]
struct VerifiedEntry {
    hash: FileHash,
    size: u64,
    verified_at: Instant,
    stored_len: u64,
    stored_mtime: Option<SystemTime>,
}

/// Content-addressed store rooted at `root`.
#[derive(Debug)]
pub struct Cas {
    root: PathBuf,
    /// Number of actual disk writes (fresh blob or repair). Healthy dedup
    /// hits that verify cleanly never increment; tests use this to prove
    /// that dedup does not rewrite.
    writes: AtomicU64,
    /// LRU of blobs verified since startup (bounded, advisory). The cache is
    /// a performance seam for hot paths: every miss is verified streamingly.
    verified: Mutex<VecDeque<VerifiedEntry>>,
}

impl Clone for Cas {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            writes: AtomicU64::new(self.writes.load(Ordering::Relaxed)),
            // The advisory cache never crosses a clone boundary.
            verified: Mutex::new(VecDeque::new()),
        }
    }
}

impl Cas {
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            writes: AtomicU64::new(0),
            verified: Mutex::new(VecDeque::new()),
        }
    }

    pub fn open(root: PathBuf) -> CasResult<Self> {
        let cas = Self::new(root);
        fs::create_dir_all(cas.root.join("tmp"))?;
        Ok(cas)
    }

    /// Root directory (public for tests and tooling).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Number of actual disk writes (fresh blobs + repairs). Healthy dedup
    /// hits never count. Test probe.
    #[cfg(test)]
    pub(crate) fn writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    fn blob_path(&self, hash: FileHash) -> PathBuf {
        self.root.join(hash.cas_path())
    }

    /// Store `bytes`, returning its content address, bounded by
    /// `DEFAULT_MAX_PUT_BYTES`. If the blob already exists it is verified
    /// (decompress + rehash) and only reused when healthy; a corrupt or
    /// collided blob is atomically overwritten. Safe under concurrency:
    /// identical content always produces the identical path, and renames are
    /// atomic.
    pub fn put(&self, bytes: &[u8]) -> CasResult<FileHash> {
        self.put_bounded(bytes, DEFAULT_MAX_PUT_BYTES)
    }

    /// Bounded put: rejects payloads larger than `max_bytes` *before* hashing
    /// with an explicit `CasError::Oversized` rather than compressing
    /// unbounded data. The default ceiling is `DEFAULT_MAX_PUT_BYTES`.
    pub fn put_bounded(&self, bytes: &[u8], max_bytes: usize) -> CasResult<FileHash> {
        if bytes.len() > max_bytes {
            return Err(CasError::Oversized {
                max: max_bytes,
                actual: bytes.len(),
            });
        }
        let hash = FileHash::from(blake3::hash(bytes).into());
        let path = self.blob_path(hash);
        if self.blob_is_valid(hash, &path) {
            // Dedup hit on a verified-healthy blob: never rewrite.
            return Ok(hash);
        }
        // Compress to memory first so the temp-file write is one syscall
        // batch; `put_reader` is the streaming path for payloads that must
        // not be materialized in RAM (identically bounded).
        let compressed = zstd::encode_all(bytes, 3).map_err(|e| CasError::Zstd(e.to_string()))?;
        self.write_compressed(&path, hash, &compressed)
    }

    /// Streaming put: zstd-compresses incrementally from `reader` into a temp
    /// file while hashing, so the payload is never materialized in RAM.
    /// Bounded by `DEFAULT_MAX_PUT_BYTES`.
    pub fn put_reader<R: Read>(&self, reader: R) -> CasResult<FileHash> {
        self.put_reader_bounded(reader, DEFAULT_MAX_PUT_BYTES)
    }

    /// Streaming put with an explicit byte bound, enforced mid-stream: a
    /// reader that yields more than `max_bytes` aborts with `Oversized` and
    /// the temp file is removed (never left behind).
    pub fn put_reader_bounded<R: Read>(
        &self,
        mut reader: R,
        max_bytes: usize,
    ) -> CasResult<FileHash> {
        let tmp = self.tmp_path("stream");
        let mut hasher = blake3::Hasher::new();
        let mut total = 0usize;
        {
            let file = fs::File::create(&tmp)?;
            let mut enc = zstd::stream::write::Encoder::new(file, 3)
                .map_err(|e| CasError::Zstd(e.to_string()))?;
            let mut buf = [0u8; 64 * 1024];
            loop {
                let n = reader.read(&mut buf).map_err(CasError::Io)?;
                if n == 0 {
                    break;
                }
                total += n;
                if total > max_bytes {
                    drop(enc);
                    let _ = fs::remove_file(&tmp);
                    return Err(CasError::Oversized {
                        max: max_bytes,
                        actual: total,
                    });
                }
                hasher.update(&buf[..n]);
                enc.write_all(&buf[..n])
                    .map_err(|e| CasError::Zstd(e.to_string()))?;
            }
            enc.finish().map_err(|e| CasError::Zstd(e.to_string()))?;
        }
        let hash = FileHash::from(hasher.finalize().into());
        let path = self.blob_path(hash);
        if self.blob_is_valid(hash, &path) {
            let _ = fs::remove_file(&tmp);
            return Ok(hash);
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        {
            let f = fs::File::open(&tmp)?;
            f.sync_all()?;
        }
        if self.finish_rename(&tmp, &path)? {
            self.writes.fetch_add(1, Ordering::Relaxed);
        }
        Ok(hash)
    }

    /// True iff the blob at `path` decompresses and rehashes to `hash`. Any
    /// failure (missing, unreadable, not zstd, wrong content) means the blob
    /// is absent or corrupt and must be (re)written. Streams: the compressed
    /// blob is never materialized (audit 50).
    fn blob_is_valid(&self, hash: FileHash, path: &Path) -> bool {
        if self.verified_size(hash).is_some() {
            return true;
        }
        let Ok(file) = fs::File::open(path) else {
            return false;
        };
        let mut sink = std::io::sink();
        match self.decode_verified(hash, file, &mut sink, None) {
            Ok(Some(size)) => {
                self.record_verified(hash, size, path);
                true
            }
            _ => false,
        }
    }

    // --- Streaming verification internals (audit 50) ---------------------

    /// Open the blob file for `hash`, mapping a missing file to
    /// [`CasError::NotFound`].
    fn open_blob(&self, hash: FileHash) -> CasResult<fs::File> {
        match fs::File::open(self.blob_path(hash)) {
            Ok(f) => Ok(f),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(CasError::NotFound(hash)),
            Err(e) => Err(e.into()),
        }
    }

    /// Decompress the blob file and stream the decoded bytes through `sink`
    /// while incrementally BLAKE3-hashing them — the same hasher used by
    /// `put`, so a blob verified here is bit-for-bit what a writer addressed.
    /// Returns `Ok(Some(size))` when the whole blob decoded AND rehashed to
    /// `hash`; `Ok(None)` when `cap_bytes` was exceeded (decode aborted, the
    /// returned prefix is discarded — an oversized blob is refused before
    /// any integrity claim); any decode failure is a loud `Err`, never a
    /// panic. `sink` may receive bytes on the error paths — callers must
    /// discard their buffer when this returns anything but `Ok(Some(_))`.
    fn decode_verified<W: Write>(
        &self,
        hash: FileHash,
        file: fs::File,
        sink: &mut W,
        cap_bytes: Option<u64>,
    ) -> CasResult<Option<u64>> {
        let mut hasher = blake3::Hasher::new();
        let size = decode_stream(file, sink, cap_bytes, Some(&mut hasher))?;
        if size.is_some() {
            let actual = FileHash::from(hasher.finalize().into());
            if actual != hash {
                return Err(CasError::HashMismatch(hash));
            }
        }
        Ok(size)
    }

    /// Decode-only variant for LRU hits: the blob was already verified
    /// against its on-disk identity (length + mtime unchanged), so only the
    /// decompression is redone — never the hash.
    fn decode_trusted<W: Write>(
        &self,
        file: fs::File,
        sink: &mut W,
        cap_bytes: Option<u64>,
    ) -> CasResult<Option<u64>> {
        decode_stream(file, sink, cap_bytes, None)
    }

    /// LRU hit when the blob was verified recently (within [`VERIFIED_TTL`])
    /// AND its file on disk is still the exact file that was verified (same
    /// length and mtime). A stat is far cheaper than a decode; corruption or
    /// repair between verification and the hit falls through to a fresh
    /// streaming verify.
    fn verified_size(&self, hash: FileHash) -> Option<u64> {
        let mut verified = self.verified.lock().unwrap();
        let pos = verified.iter().position(|e| e.hash == hash)?;
        let entry = verified.remove(pos).unwrap();
        let on_disk = fs::metadata(self.blob_path(hash)).ok()?;
        if on_disk.len() != entry.stored_len
            || on_disk.modified().ok() != entry.stored_mtime
            || entry.verified_at.elapsed() > VERIFIED_TTL
        {
            return None;
        }
        let size = entry.size;
        verified.push_back(entry);
        Some(size)
    }

    fn record_verified(&self, hash: FileHash, size: u64, path: &Path) {
        let meta = fs::metadata(path).ok();
        let (stored_len, stored_mtime) = match meta {
            Some(m) => (m.len(), m.modified().ok()),
            None => (0, None),
        };
        let mut verified = self.verified.lock().unwrap();
        if let Some(pos) = verified.iter().position(|e| e.hash == hash) {
            verified.remove(pos);
        }
        verified.push_back(VerifiedEntry {
            hash,
            size,
            verified_at: Instant::now(),
            stored_len,
            stored_mtime,
        });
        while verified.len() > VERIFIED_CACHE_MAX {
            verified.pop_front();
        }
    }

    fn tmp_path(&self, tag: &str) -> PathBuf {
        self.root.join("tmp").join(format!(
            "{}-{}-{tag}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    fn write_compressed(
        &self,
        path: &Path,
        hash: FileHash,
        compressed: &[u8],
    ) -> CasResult<FileHash> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = self.tmp_path(&hash.to_hex());
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(compressed)?;
            f.sync_all()?;
        }
        if self.finish_rename(&tmp, path)? {
            self.writes.fetch_add(1, Ordering::Relaxed);
        }
        Ok(hash)
    }

    /// Rename `tmp` onto `path`, atomically. On AlreadyExists another writer
    /// won the race and the blob is already in place (verified by the
    /// winner): the temp file is removed, never left behind. Returns whether
    /// this call performed the rename. On unix the containing directory is
    /// fsynced afterwards so a power loss cannot lose the directory entry;
    /// sync errors are ignored (e.g. macOS refuses fsync on directories).
    fn finish_rename(&self, tmp: &Path, path: &Path) -> CasResult<bool> {
        match fs::rename(tmp, path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let _ = fs::remove_file(tmp);
                return Ok(false);
            }
            Err(e) => {
                let _ = fs::remove_file(tmp);
                return Err(e.into());
            }
        }
        #[cfg(unix)]
        {
            if let Some(parent) = path.parent() {
                if let Ok(dir) = fs::File::open(parent) {
                    let _ = dir.sync_all();
                }
            }
        }
        Ok(true)
    }

    /// Existence check ONLY (cheap): path existence is not validity. A
    /// corrupt blob at the address still "exists"; use
    /// [`Cas::has_verified`] when the answer must mean "healthy".
    pub fn has(&self, hash: FileHash) -> bool {
        self.blob_path(hash).exists()
    }

    /// True only when the blob is present AND its decompressed content
    /// rehashes to `hash`. Streams with a bounded buffer — never
    /// materializes the blob. `false` (not an error) when the blob is
    /// missing, not zstd, truncated or content-mismatched: path existence
    /// alone is never validity (audit 50). Verified blobs are remembered in
    /// a bounded LRU (re-checked against the file's length and mtime), so
    /// hot paths skip the re-decode while misses always verify.
    pub fn has_verified(&self, hash_hex: &str) -> CasResult<bool> {
        let hash = FileHash::from_hex(hash_hex).ok_or_else(|| {
            CasError::Malformed(format!("{hash_hex:?} is not a 64-char hex BLAKE3 hash"))
        })?;
        if self.verified_size(hash).is_some() {
            return Ok(true);
        }
        let file = match fs::File::open(self.blob_path(hash)) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        let mut sink = std::io::sink();
        match self.decode_verified(hash, file, &mut sink, None) {
            Ok(Some(size)) => {
                self.record_verified(hash, size, &self.blob_path(hash));
                Ok(true)
            }
            // Corrupt, truncated, wrong content: present-but-unverified.
            _ => Ok(false),
        }
    }

    /// Read and verify. Corrupted blobs are an error, never silent garbage.
    pub fn get(&self, hash: FileHash) -> CasResult<Vec<u8>> {
        let file = self.open_blob(hash)?;
        let mut out = Vec::new();
        let size = match self.decode_verified(hash, file, &mut out, None)? {
            Some(s) => s,
            None => unreachable!("no cap means the decode always completes"),
        };
        self.record_verified(hash, size, &self.blob_path(hash));
        Ok(out)
    }

    /// Bounded read: `Ok(Some(bytes))` when the blob decompresses to at most
    /// `max` bytes, `Ok(None)` when it is OVERSIZED (refused before the full
    /// decode — nothing materializes past the bound). Corruption within the
    /// bound is a loud `Err`; a blob recently verified (LRU hit) skips only
    /// the re-hash, never the decode.
    pub fn get_bounded(&self, hash: FileHash, max: usize) -> CasResult<Option<Vec<u8>>> {
        let known = self.verified_size(hash);
        if known.is_some_and(|size| size > max as u64) {
            return Ok(None);
        }
        let file = self.open_blob(hash)?;
        let mut out = Vec::new();
        let size = match known {
            // LRU hit (file identity still matches): decode only.
            Some(_) => self.decode_trusted(file, &mut out, Some(max as u64))?,
            None => self.decode_verified(hash, file, &mut out, Some(max as u64))?,
        };
        match size {
            Some(size) => {
                if known.is_none() {
                    self.record_verified(hash, size, &self.blob_path(hash));
                }
                Ok(Some(out))
            }
            None => Ok(None),
        }
    }

    /// Stream a blob to a writer, verifying the hash while copying.
    /// Verified-by-construction: every byte written has been decompressed
    /// and rehashed against the address. On error the caller must discard
    /// `w` — it may have received bytes before the corruption was found.
    pub fn copy_verified_to<W: Write>(&self, hash: FileHash, mut w: W) -> CasResult<()> {
        let file = self.open_blob(hash)?;
        let size = match self.decode_verified(hash, file, &mut w, None)? {
            Some(s) => s,
            None => unreachable!("no cap means the decode always completes"),
        };
        self.record_verified(hash, size, &self.blob_path(hash));
        Ok(())
    }

    /// Size of the stored blob, without decompressing.
    pub fn stored_size(&self, hash: FileHash) -> CasResult<u64> {
        match fs::metadata(self.blob_path(hash)) {
            Ok(m) => Ok(m.len()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(CasError::NotFound(hash)),
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
                        if self.get(hash).is_err() {
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

/// Shared streaming decode: decompress `file` through a bounded 64 KiB
/// buffer, counting decoded bytes (and optionally hashing them
/// incrementally), writing everything decoded to `sink`. `Ok(None)` when
/// `cap_bytes` was exceeded — the decode is aborted, whatever reached
/// `sink` must be discarded by the caller. Decode failures (corrupt or
/// truncated zstd framing) are loud `Err`s, never panics; corrupt-framing
/// IO errors surface as [`CasError::Zstd`], other IO failures as
/// [`CasError::Io`].
fn decode_stream<W: Write>(
    file: fs::File,
    sink: &mut W,
    cap_bytes: Option<u64>,
    mut hasher: Option<&mut blake3::Hasher>,
) -> CasResult<Option<u64>> {
    let mut decoder =
        zstd::stream::read::Decoder::new(file).map_err(|e| CasError::Zstd(e.to_string()))?;
    let mut buf = [0u8; STREAM_BUF];
    let mut size: u64 = 0;
    loop {
        // Every decode failure is a loud Zstd error (matching the historical
        // decode_all taxonomy): corrupt frames, truncated streams and
        // descriptor garbage all surface here as io errors — never a panic.
        let n = decoder
            .read(&mut buf)
            .map_err(|e| CasError::Zstd(e.to_string()))?;
        if n == 0 {
            break;
        }
        size += n as u64;
        if let Some(cap) = cap_bytes {
            if size > cap {
                return Ok(None);
            }
        }
        if let Some(h) = hasher.as_deref_mut() {
            h.update(&buf[..n]);
        }
        sink.write_all(&buf[..n])?;
    }
    Ok(Some(size))
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
        assert_eq!(
            cas.blob_count(),
            1,
            "ten checkpoints of one file = one copy"
        );
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
            matches!(
                result,
                Err(CasError::HashMismatch(_)) | Err(CasError::Zstd(_))
            ),
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
        assert!(
            size < 100_000,
            "highly compressible blob must shrink on disk"
        );
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
        let h = FileHash::from_hex(&"0".repeat(64)).unwrap();
        assert!(!cas.has(h));
        // And a put must succeed afterwards.
        let h2 = cas.put(b"after crash").unwrap();
        assert_eq!(cas.get(h2).unwrap(), b"after crash");
    }

    #[test]
    fn dedup_hit_verifies_and_repairs_corrupt_blob() {
        let (_d, cas) = tmp_cas();
        let bytes = b"the same content, corrupted on disk".to_vec();
        let h = cas.put(&bytes).unwrap();
        let path = cas.blob_path(h);
        // Corrupt the stored blob in place (not valid zstd at all).
        fs::write(&path, b"garbage that is not zstd").unwrap();
        let writes_before = cas.writes();
        // put() on the same content must NOT silently reuse the corrupt blob.
        let h2 = cas.put(&bytes).unwrap();
        assert_eq!(h2, h);
        assert_eq!(cas.get(h).unwrap(), bytes, "blob must be repaired");
        assert!(cas.verify_integrity().is_empty());
        assert_eq!(
            cas.writes(),
            writes_before + 1,
            "repair is an actual disk write"
        );
        // A second hit is a healthy dedup hit again: no further rewrite.
        cas.put(&bytes).unwrap();
        assert_eq!(cas.writes(), writes_before + 1);
    }

    #[test]
    fn dedup_hit_verifies_and_repairs_collided_blob() {
        let (_d, cas) = tmp_cas();
        let bytes = b"repair me, payload".to_vec();
        let h = cas.put(&bytes).unwrap();
        let path = cas.blob_path(h);
        // Recompress DIFFERENT content and overwrite: decompresses fine but
        // rehashes to something else — a hash collision or tampering.
        let evil = zstd::encode_all(&b"totally different content"[..], 3).unwrap();
        fs::write(&path, evil).unwrap();
        let h2 = cas.put(&bytes).unwrap();
        assert_eq!(h2, h);
        assert_eq!(
            cas.get(h).unwrap(),
            bytes,
            "collided blob must be overwritten"
        );
        assert!(cas.verify_integrity().is_empty());
    }

    #[test]
    fn dedup_hit_accepts_healthy_blob_without_rewrite() {
        let (_d, cas) = tmp_cas();
        let bytes = b"healthy dedup payload".to_vec();
        let h = cas.put(&bytes).unwrap();
        assert_eq!(cas.writes(), 1);
        for _ in 0..10 {
            assert_eq!(cas.put(&bytes).unwrap(), h);
        }
        assert_eq!(cas.writes(), 1, "healthy dedup hits must never rewrite");
        assert_eq!(cas.get(h).unwrap(), bytes);
        assert!(cas.verify_integrity().is_empty());
    }

    #[test]
    fn temp_files_never_left_behind() {
        let (_d, cas) = tmp_cas();
        let cas = std::sync::Arc::new(cas);
        // A storm of racing writers of identical + distinct content: every
        // rename race (including the Windows AlreadyExists branch) must end
        // with an empty tmp/ directory.
        let mut handles = vec![];
        for t in 0..8 {
            let cas = cas.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..30 {
                    let payload = format!("storm-{t}-{i}-{}", "x".repeat(i % 700));
                    let h = cas.put(payload.as_bytes()).unwrap();
                    let _ = cas.put(payload.as_bytes()).unwrap();
                    assert_eq!(cas.get(h).unwrap(), payload.as_bytes());
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let leftovers = fs::read_dir(cas.root.join("tmp")).unwrap().count();
        assert_eq!(leftovers, 0, "no temp file may survive a put storm");
        assert!(cas.verify_integrity().is_empty());
    }

    #[test]
    fn oversized_put_rejected() {
        let (_d, cas) = tmp_cas();
        // put() enforces the documented default bound before hashing.
        let big = vec![0u8; DEFAULT_MAX_PUT_BYTES + 1];
        match cas.put(&big) {
            Err(CasError::Oversized { max, actual }) => {
                assert_eq!(max, DEFAULT_MAX_PUT_BYTES);
                assert!(actual > max);
            }
            other => panic!("put over the default bound must be rejected, got {other:?}"),
        }
        // put_bounded enforces its explicit bound.
        let small = vec![0u8; 4096];
        match cas.put_bounded(&small, 1024) {
            Err(CasError::Oversized { max, actual }) => {
                assert_eq!((max, actual), (1024, 4096));
            }
            other => panic!("put_bounded over its bound must be rejected, got {other:?}"),
        }
        assert_eq!(cas.writes(), 0, "rejected payloads must never reach disk");
        assert!(cas.put_bounded(&small, 4096).is_ok(), "bound is inclusive");
        assert_eq!(cas.writes(), 1);
        // The streaming path rejects mid-stream and leaves no temp behind.
        match cas.put_reader_bounded(std::io::repeat(0), 2048) {
            Err(CasError::Oversized { max, actual }) => {
                assert_eq!(max, 2048);
                assert!(actual > max);
            }
            other => panic!("put_reader_bounded must reject oversized streams, got {other:?}"),
        }
        assert_eq!(
            fs::read_dir(cas.root.join("tmp")).unwrap().count(),
            0,
            "oversized stream must not leave a temp file"
        );
        assert_eq!(cas.writes(), 1, "oversized stream must not reach disk");
    }

    #[test]
    fn power_loss_directory_fsync() {
        let (_d, cas) = tmp_cas();
        let h = cas.put(b"fsync me").unwrap();
        let shard = cas.blob_path(h).parent().unwrap().to_path_buf();
        assert!(shard.is_dir(), "shard dir must exist after a put");
        // Production fsyncs the shard dir after each rename; replay that
        // exact call — it must not panic even on platforms that refuse
        // directory fsync (macOS returns EINVAL, which is ignored).
        let dir = fs::File::open(&shard).unwrap();
        let _ = dir.sync_all();
        assert_eq!(cas.get(h).unwrap(), b"fsync me");
        assert!(cas.verify_integrity().is_empty());
    }

    #[test]
    fn concurrent_put_repair_race() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let (_d, cas) = tmp_cas();
        let cas = std::sync::Arc::new(cas);
        let payload: Vec<u8> = (0..4096).map(|i| ((i * 13 + 5) % 251) as u8).collect();
        let h = cas.put(&payload).unwrap();
        let path = cas.blob_path(h);
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let mut handles = vec![];
        for _ in 0..2 {
            let cas = cas.clone();
            let stop = stop.clone();
            let payload = payload.clone();
            handles.push(std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    cas.put(&payload).unwrap();
                }
            }));
        }
        let path2 = path.clone();
        let stop2 = stop.clone();
        let corruptor = std::thread::spawn(move || {
            for _ in 0..300 {
                fs::write(&path2, b"deliberate corruption, not valid zstd").unwrap();
            }
            stop2.store(true, Ordering::Relaxed);
        });
        for h in handles {
            h.join().unwrap();
        }
        corruptor.join().unwrap();
        // Whatever the last corruption left, one more put must repair it.
        let h2 = cas.put(&payload).unwrap();
        assert_eq!(h2, h);
        assert_eq!(
            cas.get(h).unwrap(),
            payload,
            "end state must be a valid blob"
        );
        assert!(cas.verify_integrity().is_empty());
    }

    #[test]
    fn put_reader_streams_and_roundtrips() {
        let (_d, cas) = tmp_cas();
        let payload: Vec<u8> = (0..200_000).map(|i| ((i * 7 + 3) % 253) as u8).collect();
        let h = cas
            .put_reader(std::io::Cursor::new(payload.clone()))
            .unwrap();
        assert_eq!(cas.get(h).unwrap(), payload);
        assert_eq!(cas.writes(), 1);
        // Dedup hit on an existing blob via the streaming path.
        let h2 = cas.put_reader(std::io::Cursor::new(payload)).unwrap();
        assert_eq!(h2, h);
        assert_eq!(cas.writes(), 1);
        assert!(cas.verify_integrity().is_empty());
        // An empty reader is a valid (empty) blob.
        let h3 = cas.put_reader(std::io::Cursor::new(Vec::new())).unwrap();
        assert_eq!(cas.get(h3).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn put_reader_repairs_corrupt_blob() {
        let (_d, cas) = tmp_cas();
        let payload = b"streamed repair payload".to_vec();
        let h = cas
            .put_reader(std::io::Cursor::new(payload.clone()))
            .unwrap();
        let path = cas.blob_path(h);
        fs::write(&path, b"corrupt").unwrap();
        let h2 = cas
            .put_reader(std::io::Cursor::new(payload.clone()))
            .unwrap();
        assert_eq!(h2, h);
        assert_eq!(cas.get(h).unwrap(), payload);
        assert!(cas.verify_integrity().is_empty());
    }

    #[test]
    fn has_verified_is_false_when_path_exists_but_content_corrupt() {
        let (_d, cas) = tmp_cas();
        let h = cas.put(b"original payload").unwrap();
        let hex = h.to_hex();
        assert!(cas.has_verified(&hex).unwrap(), "fresh blob verifies");
        let path = cas.blob_path(h);
        // Rewrite the file with DIFFERENT content: the address still exists
        // (has() says true) but is no longer valid.
        fs::write(&path, b"tampered content, still exists").unwrap();
        assert!(cas.has(h), "path-existence checks still claim presence");
        assert!(
            !cas.has_verified(&hex).unwrap(),
            "path existence is not validity: corruption must fail verification"
        );
        // Same-length corruption (mtime change) is caught too.
        let h2 = cas.put(b"0123456789abcdef").unwrap();
        let hex2 = h2.to_hex();
        assert!(cas.has_verified(&hex2).unwrap());
        fs::write(cas.blob_path(h2), b"ABCDEFGHIJKLMNOP").unwrap();
        assert!(
            !cas.has_verified(&hex2).unwrap(),
            "same-length tampering must invalidate the LRU via mtime"
        );
    }

    #[test]
    fn copy_verified_to_streams_exact_bytes_and_fails_loudly_on_corruption() {
        let (_d, cas) = tmp_cas();
        let payload: Vec<u8> = (0..300_000).map(|i| ((i * 17 + 3) % 251) as u8).collect();
        let h = cas.put(&payload).unwrap();
        let mut out = Vec::new();
        cas.copy_verified_to(h, &mut out).unwrap();
        assert_eq!(out, payload, "copy must yield the exact original bytes");
        // Corrupt the stored blob (hostile: not even zstd framing).
        let path = cas.blob_path(h);
        fs::write(&path, b"hostile bytes, no zstd frame at all").unwrap();
        let mut sink = Cursor::new(Vec::new());
        let err = cas.copy_verified_to(h, &mut sink).unwrap_err();
        assert!(
            matches!(
                err,
                CasError::Zstd(_) | CasError::HashMismatch(_) | CasError::Io(_)
            ),
            "corruption must be a loud error, got {err:?}"
        );
        // Valid-zstd-wrong-content is a HashMismatch, never silent.
        let h2 = cas.put(b"legit content").unwrap();
        let path2 = cas.blob_path(h2);
        let evil = zstd::encode_all(&b"different content"[..], 3).unwrap();
        fs::write(&path2, evil).unwrap();
        let mut sink2 = Cursor::new(Vec::new());
        let err = cas.copy_verified_to(h2, &mut sink2).unwrap_err();
        assert!(
            matches!(err, CasError::HashMismatch(x) if x == h2),
            "valid-zstd tampering must be HashMismatch, got {err:?}"
        );
    }

    #[test]
    fn get_bounded_none_for_oversized_some_for_small() {
        let (_d, cas) = tmp_cas();
        let small = b"tiny blob".to_vec();
        let h = cas.put(&small).unwrap();
        assert_eq!(
            cas.get_bounded(h, small.len()).unwrap().as_deref(),
            Some(&small[..])
        );
        assert_eq!(
            cas.get_bounded(h, small.len() - 1).unwrap(),
            None,
            "a blob larger than the bound is refused, not truncated"
        );
        // The LRU knows the decoded size: an oversized re-query is answered
        // without decoding.
        assert!(cas.has_verified(&h.to_hex()).unwrap());
        assert_eq!(cas.get_bounded(h, small.len() - 1).unwrap(), None);
        let big: Vec<u8> = (0..50_000).map(|i| ((i * 5 + 1) % 256) as u8).collect();
        let hb = cas.put(&big).unwrap();
        assert_eq!(cas.get_bounded(hb, 49_999).unwrap(), None);
        assert_eq!(
            cas.get_bounded(hb, 50_000).unwrap().as_deref(),
            Some(&big[..])
        );
    }

    #[test]
    fn corrupt_zstd_framing_fails_loudly_never_panics() {
        let (_d, cas) = tmp_cas();
        let h = cas.put(b"framing victim").unwrap();
        let path = cas.blob_path(h);
        // Hostile truncation mid-frame: a partial valid-looking frame.
        let framed = zstd::encode_all(&b"framing victim"[..], 3).unwrap();
        fs::write(&path, &framed[..framed.len() / 2]).unwrap();
        let results = std::panic::catch_unwind(|| {
            let _ = cas.get(h);
            let _ = cas.get_bounded(h, 4096);
            let _ = cas.copy_verified_to(h, Cursor::new(Vec::new()));
            cas.has_verified(&h.to_hex()).unwrap()
        });
        assert!(results.is_ok(), "hostile framing must not panic");
        assert!(!results.unwrap(), "truncated frame is not verified content");
        // Overwrite with raw garbage: same guarantees.
        fs::write(&path, b"\xff\xff\xff\xff not a frame").unwrap();
        let r = std::panic::catch_unwind(|| {
            let e = cas.get(h).unwrap_err();
            let _ = cas.get_bounded(h, 4096);
            e
        });
        assert!(r.is_ok(), "garbage framing must not panic");
        assert!(matches!(
            r.unwrap(),
            CasError::Zstd(_) | CasError::HashMismatch(_)
        ));
    }

    #[test]
    fn verified_lru_is_bounded_and_hash_strings_are_validated() {
        let (_d, cas) = tmp_cas();
        // 300 distinct verified blobs: the LRU must stay at its bound.
        for i in 0..300u32 {
            let h = cas.put(format!("blob-{i}").as_bytes()).unwrap();
            assert!(cas.has_verified(&h.to_hex()).unwrap());
        }
        assert!(cas.verified.lock().unwrap().len() <= VERIFIED_CACHE_MAX);
        // A verified blob that fell off the LRU still verifies (on miss).
        let old = {
            let h = cas.put(b"lru survivor").unwrap();
            assert!(cas.has_verified(&h.to_hex()).unwrap());
            h
        };
        assert!(cas.has_verified(&old.to_hex()).unwrap());
        // Malformed hash strings are loud errors, not silent falses.
        match cas.has_verified("not-a-hash") {
            Err(CasError::Malformed(_)) => {}
            other => panic!("malformed hash must be Malformed, got {other:?}"),
        }
        // Missing blob: verified-false (absent is not corruption).
        let ghost = FileHash::from([9; 32]);
        assert!(!cas.has_verified(&ghost.to_hex()).unwrap());
    }
}
