//! Content hashes (BLAKE3, 32 bytes). `faktor-core` stays dependency-free, so
//! the hash *value* type lives here and the hashing implementation lives in
//! `faktor-cas`.

use std::fmt;

/// A BLAKE3 content hash. Displayed as lowercase hex (64 chars).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct FileHash([u8; 32]);

impl FileHash {
    pub const fn from(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, chunk) in hex.as_bytes().as_chunks::<2>().0.iter().enumerate() {
            let hi = hexval(chunk[0])?;
            let lo = hexval(chunk[1])?;
            out[i] = (hi << 4) | lo;
        }
        Some(Self(out))
    }

    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.0 {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// CAS path fragment: `ab/cdef...` (first two hex chars as directory).
    pub fn cas_path(self) -> String {
        let hex = self.to_hex();
        format!("{}/{}", &hex[0..2], &hex[2..])
    }
}

fn hexval(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

impl fmt::Display for FileHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl serde::Serialize for FileHash {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> serde::Deserialize<'de> for FileHash {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        FileHash::from_hex(&s)
            .ok_or_else(|| serde::de::Error::custom(format!("invalid 32-byte hex hash: {s:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        let h = FileHash::from([0xAB; 32]);
        assert_eq!(h.to_hex().len(), 64);
        assert!(h.to_hex().starts_with("abababab"));
        assert_eq!(FileHash::from_hex(&h.to_hex()), Some(h));
    }

    #[test]
    fn malformed_hex_rejected() {
        assert!(FileHash::from_hex("").is_none());
        assert!(FileHash::from_hex("abc").is_none());
        assert!(FileHash::from_hex(&"z".repeat(64)).is_none());
        assert!(FileHash::from_hex(&"g".repeat(64)).is_none());
        assert!(FileHash::from_hex(&"0".repeat(63)).is_none());
        assert!(FileHash::from_hex(&"0".repeat(65)).is_none());
    }

    #[test]
    fn uppercase_hex_accepted() {
        let h = FileHash::from_hex(&"A".repeat(64)).unwrap();
        assert_eq!(h.to_hex(), "a".repeat(64));
    }

    #[test]
    fn json_roundtrip_and_garbage() {
        let h = FileHash::from([1; 32]);
        let s = serde_json::to_string(&h).unwrap();
        assert_eq!(serde_json::from_str::<FileHash>(&s).unwrap(), h);
        assert!(serde_json::from_str::<FileHash>("\"xyz\"").is_err());
        assert!(serde_json::from_str::<FileHash>("42").is_err());
        assert!(serde_json::from_str::<FileHash>("null").is_err());
    }

    #[test]
    fn cas_path_split() {
        let hex = format!("ab{}", "cd".repeat(31));
        assert_eq!(hex.len(), 64);
        let h = FileHash::from_hex(&hex).unwrap();
        let expected = format!("ab/{}", &hex[2..]);
        assert_eq!(h.cas_path(), expected);
        let path = h.cas_path();
        let (shard, rest) = path.split_once('/').unwrap();
        assert_eq!(shard.len(), 2);
        assert_eq!(rest.len(), 62);
    }
}
