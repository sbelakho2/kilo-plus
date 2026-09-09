//! Exact configured-secret registry (audit P0-38).
//!
//! [`SecretRegistry`] detects **exact** credentials (provider keys,
//! bearer tokens, arbitrary byte strings) in a payload at their precise
//! byte offsets — complementing the pattern scanner in [`crate::payload`].
//!
//! ## Fingerprints, not plaintext
//!
//! The registry stores only a **non-reversible digest** per secret plus
//! its length. The task brief names blake3; this crate is std-only (no
//! `Cargo.toml` dependency additions are permitted for this crate), so the
//! registry uses a from-scratch, known-answer-tested SHA-256
//! ([`sha256`]). SHA-256 and blake3 are interchangeable here: both are
//! deterministic, collision-resistant, non-reversible digests whose only
//! role is exact look-up, and neither is a MAC — **an offline attacker who
//! guesses a low-entropy secret can confirm the guess against the
//! fingerprint**, exactly as with blake3. Treat the registry like the
//! config it is derived from (protect it like a credential file); do not
//! rely on hashing alone to hide weak secrets.
//!
//! The plaintext secret is passed to [`SecretRegistry::register`] once,
//! digested, and never stored or printed: the registry's [`Debug`]
//! implementation is redacted (counts and lengths only), never the values,
//! and scan hits carry `kind: "configured_secret"` with an empty snippet.
//!
//! ## Matching
//!
//! [`SecretRegistry::scan_exact`] finds every registered secret in the
//! payload. A 64-bit rolling hash is used as a pre-filter (one pass per
//! distinct secret length, O(payload) time per length class) and every
//! pre-filter hit is verified with the full SHA-256 fingerprint, so there
//! are no false positives and no false negatives. The plaintext is only
//! ever needed at registration time.
//!
//! ## Active secrets
//!
//! [`SecretRegistry::with_active_source`] attaches an [`ExactSecretSource`]
//! (`Arc<dyn ExactSecretSource>`) that supplies the *live* credentials at
//! scan time (rotating tokens, env-provided keys). Their values are hashed
//! per scan and matched the same way; they are never stored.

use std::fmt;
use std::sync::Arc;

use crate::SecretHit;

/// Canonical kind label of exact registry hits.
pub const CONFIGURED_SECRET_KIND: &str = "configured_secret";

/// A live source of exact secrets (active credentials). Values are read at
/// scan time and never stored by the registry.
pub trait ExactSecretSource: Send + Sync {
    /// The active secret values right now. Returned plaintext is digested
    /// and dropped immediately; it must not be logged by implementations.
    fn active_secrets(&self) -> Vec<Arc<[u8]>>;
}

/// One registered fingerprint: SHA-256 of the secret, its rolling-hash
/// pre-filter value, and its length. Plaintext is never retained.
#[derive(Clone)]
struct Fingerprint {
    digest: [u8; 32],
    roll: u64,
    len: usize,
}

/// Exact configured-secret registry. See the module docs.
#[derive(Default)]
pub struct SecretRegistry {
    fingerprints: Vec<Fingerprint>,
    active: Option<Arc<dyn ExactSecretSource>>,
}

/// Deterministic, collision-resistant, non-reversible fingerprint. The
/// task brief names blake3; this crate cannot add dependencies, so a
/// compact, known-answer-tested SHA-256 is used instead (see module docs).
fn sha256(data: &[u8]) -> [u8; 32] {
    let mut ctx = Sha256::new();
    ctx.update(data);
    ctx.finalize()
}

impl SecretRegistry {
    /// An empty registry.
    pub fn new() -> SecretRegistry {
        SecretRegistry::default()
    }

    /// Register one exact secret. The plaintext is hashed immediately and
    /// never retained; registering the same value twice is a no-op.
    pub fn register(&mut self, secret: &[u8]) {
        if secret.is_empty() {
            return; // an empty secret matches everywhere; refuse it
        }
        if self.fingerprints.iter().any(|f| f.len == secret.len()) {
            // Fast pre-filter already exists for this length; check exact.
            let digest = sha256(secret);
            if self
                .fingerprints
                .iter()
                .any(|f| f.len == secret.len() && f.digest == digest)
            {
                return;
            }
        }
        let digest = sha256(secret);
        self.fingerprints.push(Fingerprint {
            digest,
            roll: roll_hash(secret),
            len: secret.len(),
        });
    }

    /// Register every value from an iterator of byte strings.
    pub fn register_all<'a, I>(&mut self, secrets: I)
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        for s in secrets {
            self.register(s);
        }
    }

    /// Attach a live source of active secrets (see module docs). The
    /// existing configured set is kept; active values are merged per scan.
    pub fn with_active_source(mut self, source: Arc<dyn ExactSecretSource>) -> SecretRegistry {
        self.active = Some(source);
        self
    }

    /// Number of registered configured secrets (excluding the active
    /// source's live values).
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// True when no secrets are registered.
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }

    /// Exact-match scan: every occurrence of every registered (and active)
    /// secret in `payload`, as hits with exact byte offsets and
    /// `kind: "configured_secret"`. Results are ordered by offset; when
    /// several registered secrets start at the same offset, the longest
    /// span is reported.
    pub fn scan_exact(&self, payload: &[u8]) -> Vec<SecretHit> {
        if payload.is_empty() {
            return Vec::new();
        }
        let mut fps = self.fingerprints.clone();
        if let Some(source) = &self.active {
            for value in source.active_secrets() {
                if !value.is_empty() {
                    fps.push(Fingerprint {
                        digest: sha256(&value),
                        roll: roll_hash(&value),
                        len: value.len(),
                    });
                }
            }
        }
        let mut all = scan_exact_impl(payload, &fps);
        // Longest first within one start offset, then collapse duplicates.
        all.sort_unstable_by_key(|h| (h.0, std::cmp::Reverse(h.1)));
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(all.len());
        for (offset, len) in all {
            match merged.last_mut() {
                Some((off, cur)) if *off == offset => *cur = (*cur).max(len),
                _ => merged.push((offset, len)),
            }
        }
        merged
            .into_iter()
            .map(|(offset, len)| SecretHit {
                pattern_index: usize::MAX,
                kind: CONFIGURED_SECRET_KIND.to_string(),
                snippet: String::new(), // never echo candidate bytes
                redacted: format!("<redacted:{CONFIGURED_SECRET_KIND}>"),
                offset,
                len,
            })
            .collect()
    }
}

/// Rolling-hash constant: an odd 64-bit multiplier (no particular
/// cryptographic role — the SHA-256 digest is the authority; this is only a
/// pre-filter, false positives are eliminated by verification).
const ROLL_BASE: u64 = 0x9E37_79B1_85EB_CA87;

fn roll_hash(secret: &[u8]) -> u64 {
    secret.iter().fold(0u64, |h, b| {
        h.wrapping_mul(ROLL_BASE).wrapping_add(*b as u64)
    })
}

/// The real scan: rolling pre-filter + SHA-256 verification per candidate.
fn scan_exact_impl(payload: &[u8], fps: &[Fingerprint]) -> Vec<(usize, usize)> {
    if fps.is_empty() {
        return Vec::new();
    }
    // Group fingerprints by length so one rolling pass covers all of them.
    let mut by_len: std::collections::BTreeMap<usize, Vec<&Fingerprint>> = Default::default();
    for f in fps {
        if f.len <= payload.len() {
            by_len.entry(f.len).or_default().push(f);
        }
    }
    let mut hits = Vec::new();
    for (len, group) in by_len {
        let mut hash = 0u64;
        let mut pow = 1u64;
        for _ in 0..len.saturating_sub(1) {
            pow = pow.wrapping_mul(ROLL_BASE);
        }
        for (i, b) in payload.iter().enumerate() {
            if i >= len {
                // Slide: remove the byte leaving the window FIRST, then
                // append (multiply-add after the drop keeps the head term's
                // coefficient at B^(len-1)).
                hash = hash.wrapping_sub(pow.wrapping_mul(payload[i - len] as u64));
                hash = hash.wrapping_mul(ROLL_BASE).wrapping_add(*b as u64);
                let cand_start = i + 1 - len;
                if group.iter().any(|f| f.roll == hash) {
                    for f in &group {
                        if f.roll == hash {
                            // Verify: never trust the 64-bit pre-filter.
                            let slice = &payload[cand_start..cand_start + len];
                            if sha256(slice) == f.digest {
                                hits.push((cand_start, len));
                            }
                        }
                    }
                }
            } else {
                hash = hash.wrapping_mul(ROLL_BASE).wrapping_add(*b as u64);
                if i + 1 == len {
                    let cand_start = 0;
                    if group.iter().any(|f| f.roll == hash) {
                        for f in &group {
                            if f.roll == hash && sha256(&payload[..len]) == f.digest {
                                hits.push((cand_start, len));
                            }
                        }
                    }
                }
            }
        }
    }
    hits
}

// ---------------------------------------------------------------------------
// std-only SHA-256 (compact, known-answer tested)
// ---------------------------------------------------------------------------

/// Incremental SHA-256 state.
struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total: u64,
}

impl Sha256 {
    fn new() -> Sha256 {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                compress(&mut self.h, &block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            compress(&mut self.h, &block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.total.wrapping_mul(8);
        // Padding: 0x80 then zeros then the 8-byte big-endian bit length.
        let pad_len = if self.buf_len < 56 {
            56 - self.buf_len
        } else {
            120 - self.buf_len
        };
        let mut pad = vec![0u8; pad_len];
        pad[0] = 0x80;
        self.update(&pad);
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&bit_len.to_be_bytes());
        self.update(&len_bytes);
        debug_assert_eq!(self.buf_len, 0);
        let mut out = [0u8; 32];
        for (i, word) in self.h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn compress(h: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([
            block[i * 4],
            block[i * 4 + 1],
            block[i * 4 + 2],
            block[i * 4 + 3],
        ]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

impl fmt::Debug for SecretRegistry {
    /// Redacted: prints counts and lengths only — never secret values,
    /// never digest material that could confirm an offline guess faster.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lens: Vec<usize> = self.fingerprints.iter().map(|fp| fp.len).collect();
        f.debug_struct("SecretRegistry")
            .field("registered_count", &lens.len())
            .field("registered_lengths", &lens)
            .field("active_source", &self.active.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha256_known_answers() {
        // NIST/FIPS vectors.
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Two-block message (55+ byte boundary crossing).
        let big = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
        assert_eq!(
            hex(&sha256(big)),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
    }

    #[test]
    fn sha256_incremental_chunks_equal_oneshot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i % 251) as u8).collect();
        let whole = sha256(&data);
        let mut ctx = Sha256::new();
        for chunk in data.chunks(7) {
            ctx.update(chunk);
        }
        let incr = ctx.finalize();
        assert_eq!(whole, incr);
        // Odd boundary: single byte at a time.
        let mut ctx = Sha256::new();
        for b in &data {
            ctx.update(&[*b]);
        }
        assert_eq!(whole, ctx.finalize());
    }

    #[test]
    fn bearer_at_offset_zero_and_at_end_detected() {
        let mut reg = SecretRegistry::new();
        let token: Vec<u8> = (0..32u8).collect(); // arbitrary 32-byte bearer
        reg.register(&token);
        let mut payload = Vec::new();
        payload.extend_from_slice(&token); // offset 0
        payload.extend_from_slice(b"prefix junk .... ");
        payload.extend_from_slice(&token); // again mid
        payload.extend_from_slice(b"tail=");
        payload.extend_from_slice(&token); // at the end
        let hits = reg.scan_exact(&payload);
        assert_eq!(hits.len(), 3, "all three occurrences, {hits:?}");
        assert_eq!(hits[0].offset, 0);
        assert_eq!(hits[0].len, 32);
        assert_eq!(hits[0].kind, CONFIGURED_SECRET_KIND);
        assert_eq!(hits[1].offset, 49, "prefix junk filler is 17 bytes");
        assert_eq!(hits[2].offset, payload.len() - 32);
        assert!(hits.iter().all(|h| h.snippet.is_empty()));
    }

    #[test]
    fn plaintext_never_appears_in_debug_or_hits() {
        let mut reg = SecretRegistry::new();
        let secret = b"super-secret-config-token-0123456789";
        reg.register(secret);
        let dbg = format!("{reg:?}");
        assert!(
            !dbg.contains("super-secret") && !dbg.contains("0123456789"),
            "debug leaks the value: {dbg}"
        );
        assert!(dbg.contains("registered_lengths"));
        // Hits carry no snippet either.
        let hits = reg.scan_exact(b"x super-secret-config-token-0123456789 y");
        assert_eq!(hits.len(), 1);
        assert!(hits[0].snippet.is_empty());
        assert!(!format!("{hits:?}").contains("super-secret"));
    }

    #[test]
    fn debug_never_leaks_digest_material_that_confirms_an_offline_guess() {
        // The registry's Debug must not expose even the fingerprint prefix:
        // an attacker who can read logs and guesses a weak secret could
        // confirm the guess against a leaked digest.
        let mut reg = SecretRegistry::new();
        let secret = b"guessable-config-value-1234";
        reg.register(secret);
        let dbg = format!("{reg:?}");
        let digest = hex(&sha256(secret));
        assert!(!dbg.contains(&digest), "debug leaks the full digest: {dbg}");
        assert!(
            !dbg.contains(&digest[..16]),
            "debug leaks enough digest to confirm guesses: {dbg}"
        );
        assert!(!dbg.contains("sha") && !dbg.contains("digest"), "{dbg}");
    }

    #[test]
    fn exact_secret_at_the_final_byte_is_detected_across_length_groups() {
        // Adversarial: several registered lengths (distinct rolling passes);
        // the secret ENDS exactly at the last byte of the payload while a
        // LONGER registered secret of a different length group sits beside
        // it — the shorter tail must not be swallowed by the longer group.
        let mut reg = SecretRegistry::new();
        let short: Vec<u8> = b"tail-secret-9f".to_vec();
        let long: Vec<u8> = b"0123456789abcdef0123456789abcdef0123456789abcdef".to_vec();
        reg.register(&short);
        reg.register(&long);
        let mut payload = vec![b'z'; 7000];
        payload.extend_from_slice(&short); // ends at payload.len()
        let hits = reg.scan_exact(&payload);
        assert_eq!(hits.len(), 1, "{hits:?}");
        assert_eq!(hits[0].len, short.len());
        assert_eq!(
            hits[0].offset + hits[0].len,
            payload.len(),
            "the hit must reach the very last byte"
        );
        // Single-byte-long tail scan: offset 0, ends at the last byte.
        let mut reg = SecretRegistry::new();
        reg.register(b"!");
        let payload = format!("{}!", "a".repeat(4096));
        let hits = reg.scan_exact(payload.as_bytes());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].offset + hits[0].len, payload.len());
        assert_eq!(hits[0].offset, 4096);
    }

    #[test]
    fn registry_integrates_with_the_pattern_payload_scan_path() {
        // The daemon scan path runs the whole-payload pattern scan and then
        // the exact registry over the SAME body (egress shape): lock the
        // integration — same payload, both engines, kind-deduplicated
        // merge, registry hits snippet-free, and nothing ever echoed.
        let exact = b"ghp_0123456789abcdefghijklmnopqrstuv";
        let mut reg = SecretRegistry::new();
        reg.register(exact);
        let compiled =
            crate::CompiledSecretPolicy::try_from(crate::SecretPolicy::default()).unwrap();
        let mut payload = b"prefix ghp_0123456789abcdefghijklmnopqrstuv suffix".to_vec();
        payload.extend_from_slice(exact); // also at the very end
        let outcome = crate::payload::scan_payload_compiled(
            &payload,
            &crate::payload::ScanPolicy::default(),
            &compiled,
        );
        let mut kinds: Vec<String> = match outcome {
            crate::payload::ScanOutcome::Found(hits) => {
                let mut seen = Vec::new();
                for kind in hits.into_iter().map(|h| h.kind) {
                    if !seen.contains(&kind) {
                        seen.push(kind);
                    }
                }
                seen
            }
            other => panic!("payload must be Found: {other:?}"),
        };
        let exact_hits = reg.scan_exact(&payload);
        assert_eq!(exact_hits.len(), 2, "exact hits at prefix and last byte");
        assert_eq!(exact_hits[1].offset + exact_hits[1].len, payload.len());
        for hit in &exact_hits {
            assert_eq!(hit.kind, CONFIGURED_SECRET_KIND);
            assert!(hit.snippet.is_empty());
            if !kinds.contains(&hit.kind) {
                kinds.push(hit.kind.clone());
            }
        }
        assert_eq!(kinds, vec!["github_token", CONFIGURED_SECRET_KIND]);
        let everything = format!("{kinds:?} {reg:?} {exact_hits:?}");
        assert!(
            !everything.contains("ghp_0123456789"),
            "the merged path never echoes a value: {everything}"
        );
    }

    #[test]
    fn near_miss_and_case_variants_do_not_match() {
        let mut reg = SecretRegistry::new();
        reg.register(b"abc");
        reg.register(b"xyz");
        // NOTE: "abcd" is NOT here — it legitimately contains "abc".
        for payload in ["ab", "abd", "abXc", "ABC", "xy", "xzyz"] {
            assert!(
                reg.scan_exact(payload.as_bytes()).is_empty(),
                "{payload:?} must be clean"
            );
        }
        let hits = reg.scan_exact(b"xxabcyyzzzz");
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].offset, hits[0].len), (2, 3));
    }

    #[test]
    fn substring_of_registered_secret_still_exact() {
        // Only the FULL registered value matches: registering "tok123"
        // must not fire on "tok12" or "123".
        let mut reg = SecretRegistry::new();
        reg.register(b"tok123");
        assert!(reg.scan_exact(b"tok12").is_empty());
        assert!(reg.scan_exact(b"x123").is_empty());
        assert_eq!(reg.scan_exact(b"tok123").len(), 1);
    }

    #[test]
    fn overlapping_registered_secrets_report_longest_per_offset() {
        let mut reg = SecretRegistry::new();
        reg.register(b"abc");
        reg.register(b"abcd");
        let hits = reg.scan_exact(b"xxabcd");
        // "abcd" contains "abc": both digest-match at offset 2; the
        // longest span wins for the shared start.
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].offset, hits[0].len), (2, 4));
        // "abc" alone elsewhere is still its own hit.
        let hits = reg.scan_exact(b"abc abcd");
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn empty_and_binary_payloads_are_safe() {
        let mut reg = SecretRegistry::new();
        reg.register(b"key");
        assert!(reg.scan_exact(b"").is_empty());
        let bin: Vec<u8> = (0u8..=255).collect();
        let hits = reg.scan_exact(&bin);
        assert!(hits.is_empty());
        reg.register(&bin);
        assert_eq!(reg.scan_exact(&bin).len(), 1);
        // Binary secret inside binary payload with hostile bytes.
        let mut payload = vec![0u8, 0xff, 0x00, 0x80];
        payload.extend_from_slice(&bin);
        payload.push(0xfe);
        assert_eq!(reg.scan_exact(&payload).len(), 1);
    }

    #[test]
    fn duplicate_registration_is_a_noop() {
        let mut reg = SecretRegistry::new();
        reg.register(b"dup");
        reg.register(b"dup");
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.scan_exact(b"dup dup").len(), 2);
    }

    struct RotatingSource {
        value: Arc<[u8]>,
    }

    impl ExactSecretSource for RotatingSource {
        fn active_secrets(&self) -> Vec<Arc<[u8]>> {
            vec![self.value.clone()]
        }
    }

    #[test]
    fn active_source_values_are_scanned_without_registration() {
        let value: Arc<[u8]> = Arc::from(&b"live-token-abcdef"[..]);
        let reg = SecretRegistry::new().with_active_source(Arc::new(RotatingSource {
            value: value.clone(),
        }));
        assert!(reg.is_empty(), "active values are not registered storage");
        let payload = format!("Bearer {} end", String::from_utf8_lossy(&value));
        let hits = reg.scan_exact(payload.as_bytes());
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].offset, 7);
        assert_eq!(hits[0].len, value.len());
        // Debug stays redacted with an active source attached.
        let dbg = format!("{reg:?}");
        assert!(!dbg.contains("live-token"), "{dbg}");
    }

    #[test]
    fn long_secret_crosses_length_groups_exactly() {
        // Distinct lengths exercise the per-length rolling passes.
        let mut reg = SecretRegistry::new();
        let a: Vec<u8> = b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA".to_vec();
        let b: Vec<u8> = b"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB".to_vec();
        reg.register(&a);
        reg.register(&b);
        let mut payload = a.clone();
        payload.extend_from_slice(&b);
        payload.extend_from_slice(&a);
        let hits = reg.scan_exact(&payload);
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].offset, 0);
        assert_eq!(hits[0].len, a.len());
        assert_eq!(hits[1].offset, a.len());
        assert_eq!(hits[1].len, b.len());
        assert_eq!(hits[2].offset, a.len() + b.len());
    }
}
