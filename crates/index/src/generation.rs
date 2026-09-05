//! Generation-file codec and workspace fingerprints (audits 30/64).
//!
//! A published generation is one immutable, durable file:
//!
//! ```text
//! <data_root>/generations/<workspace>/gen-<g>.json
//! ```
//!
//! The file holds the generation envelope (format tag, workspace, built
//! generation, the workspace fingerprint the build saw, and the full
//! per-workspace inverted/symbol index). Builders write to
//! `<data_root>/scratch/<workspace>/gen-<g>-<nonce>.tmp`, fsync, and a
//! single rename publishes the file — a reader holding generation `g` never
//! observes a partial `g+1`, because the file only ever appears complete.
//!
//! The envelope is written with deterministic map order (BTreeMap), so
//! re-serializing a loaded generation produces byte-identical JSON — a
//! loaded generation can be re-persisted verbatim.

use std::collections::BTreeMap;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{FileEntry, Symbol, WorkspaceIndex};

/// Format tag of a generation file; bumped on incompatible envelope shapes.
pub const GENERATION_FILE_FORMAT: u32 = 1;

/// One entry of a workspace fingerprint: the identity of a regular file the
/// walker saw (relative path, size, mtime ms). Compared entry-wise after
/// sorting; a difference means the filesystem changed and the generation is
/// stale.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FingerprintEntry {
    pub path: String,
    pub size: u64,
    pub modified_ms: i64,
}

/// The durable generation file: everything needed to serve generation `g`
/// after a restart without rescanning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationFile {
    pub format: u32,
    pub workspace: u64,
    pub generation: u64,
    pub built_ms: i64,
    /// Filesystem fingerprint the build was scanned against.
    pub fingerprint: Vec<FingerprintEntry>,
    pub data: WorkspaceData,
}

/// Deterministically ordered per-workspace index payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceData {
    /// rel path (sorted) -> file entry.
    pub files: BTreeMap<String, StoredFile>,
    /// token -> (rel path -> freq), sorted.
    pub postings: BTreeMap<String, BTreeMap<String, u32>>,
    /// symbol name (lowercase) -> sorted hit list.
    pub symbols: BTreeMap<String, Vec<StoredSymbolHit>>,
    pub token_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredFile {
    pub tokens: Vec<String>,
    pub symbols: Vec<Symbol>,
    pub modified_ms: i64,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredSymbolHit {
    pub path: String,
    pub symbol: Symbol,
}

impl GenerationFile {
    /// Capture the per-workspace maps of `index` (which must hold exactly
    /// this workspace's data — the service's per-workspace indexes satisfy
    /// that) into a deterministic envelope.
    pub fn capture(
        workspace: u64,
        generation: u64,
        index: &WorkspaceIndex,
        fingerprint: Vec<FingerprintEntry>,
    ) -> Self {
        let mut files = BTreeMap::new();
        if let Some(map) = index.files.get(&crate::WorkspaceId::new(workspace)) {
            for (path, e) in map {
                files.insert(
                    path.clone(),
                    StoredFile {
                        tokens: e.tokens.clone(),
                        symbols: e.symbols.clone(),
                        modified_ms: e.modified_ms,
                        size: e.size,
                    },
                );
            }
        }
        let mut postings = BTreeMap::new();
        if let Some(map) = index.postings.get(&crate::WorkspaceId::new(workspace)) {
            for (tok, m) in map {
                postings.insert(
                    tok.clone(),
                    m.iter().map(|(k, v)| (k.clone(), *v)).collect(),
                );
            }
        }
        let mut symbols = BTreeMap::new();
        if let Some(map) = index.symbols.get(&crate::WorkspaceId::new(workspace)) {
            for (name, hits) in map {
                let mut list: Vec<StoredSymbolHit> = hits
                    .iter()
                    .map(|(path, symbol)| StoredSymbolHit {
                        path: path.clone(),
                        symbol: symbol.clone(),
                    })
                    .collect();
                list.sort_by(|a, b| a.path.cmp(&b.path).then(a.symbol.line.cmp(&b.symbol.line)));
                symbols.insert(name.clone(), list);
            }
        }
        Self {
            format: GENERATION_FILE_FORMAT,
            workspace,
            generation,
            built_ms: now_ms(),
            fingerprint,
            data: WorkspaceData {
                files,
                postings,
                symbols,
                token_count: index.token_count() as u64,
            },
        }
    }

    /// Materialize a fresh single-workspace in-memory index from the
    /// envelope. A corrupt or version-skewed file is a typed error — the
    /// caller fails loudly (durable Failed), never serves a silent empty
    /// index.
    pub fn materialize(&self) -> Result<WorkspaceIndex, String> {
        if self.format != GENERATION_FILE_FORMAT {
            return Err(format!(
                "generation file format {} unsupported (expected {GENERATION_FILE_FORMAT})",
                self.format
            ));
        }
        let ws = crate::WorkspaceId::new(self.workspace);
        let mut idx = WorkspaceIndex::new();
        let mut files = HashMap::new();
        for (path, f) in &self.data.files {
            files.insert(
                path.clone(),
                FileEntry {
                    tokens: f.tokens.clone(),
                    symbols: f.symbols.clone(),
                    modified_ms: f.modified_ms,
                    size: f.size,
                },
            );
        }
        idx.files.insert(ws, files);
        let mut postings = HashMap::new();
        for (tok, map) in &self.data.postings {
            postings.insert(
                tok.clone(),
                map.iter().map(|(k, v)| (k.clone(), *v)).collect(),
            );
        }
        idx.postings.insert(ws, postings);
        let mut symbols: HashMap<String, Vec<(String, Symbol)>> = HashMap::new();
        for (name, hits) in &self.data.symbols {
            symbols.insert(
                name.clone(),
                hits.iter()
                    .map(|h| (h.path.clone(), h.symbol.clone()))
                    .collect(),
            );
        }
        idx.symbols.insert(ws, symbols);
        idx.token_count = self.data.token_count as usize;
        Ok(idx)
    }

    /// Serialize deterministically (sorted maps; stable field order).
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("generation encode: {e}"))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let file: Self =
            serde_json::from_slice(bytes).map_err(|e| format!("generation decode: {e}"))?;
        if file.format != GENERATION_FILE_FORMAT {
            return Err(format!(
                "generation file format {} unsupported (expected {GENERATION_FILE_FORMAT})",
                file.format
            ));
        }
        Ok(file)
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SymbolKind, WorkspaceId};
    use std::path::Path;

    const SRC: &str = "pub fn alpha() {}\npub struct Beta {}\n";

    fn sample() -> (WorkspaceIndex, u64) {
        let mut idx = WorkspaceIndex::new();
        idx.index_file(
            WorkspaceId::new(7),
            Path::new("src/a.rs"),
            SRC.as_bytes(),
            123,
        )
        .unwrap();
        idx.index_file(
            WorkspaceId::new(7),
            Path::new("src/b.py"),
            b"def gamma(): pass",
            456,
        )
        .unwrap();
        (idx, 7)
    }

    #[test]
    fn envelope_roundtrip_materializes_identical_index() {
        let (idx, ws) = sample();
        let fp = vec![FingerprintEntry {
            path: "src/a.rs".into(),
            size: SRC.len() as u64,
            modified_ms: 123,
        }];
        let env = GenerationFile::capture(ws, 3, &idx, fp);
        let bytes = env.to_bytes().unwrap();
        let back = GenerationFile::from_bytes(&bytes).unwrap();
        assert_eq!(back, env);
        assert_eq!(back.generation, 3);
        assert_eq!(back.data.files.len(), 2);

        let mat = back.materialize().unwrap();
        // The materialized index serves the exact same lookups.
        assert!(!mat
            .files_for_token(WorkspaceId::new(7), "alpha", 10)
            .is_empty());
        assert!(!mat
            .files_for_token(WorkspaceId::new(7), "gamma", 10)
            .is_empty());
        let syms = mat.symbols_in(WorkspaceId::new(7), Path::new("src/a.rs"));
        let names: Vec<&str> = syms.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"Beta"));
        assert_eq!(
            syms.iter().find(|s| s.name == "Beta").unwrap().kind,
            SymbolKind::Struct
        );
        // Determinism: re-serializing a loaded generation is byte-identical
        // modulo the capture timestamp (built_ms is wall-clock).
        let mut again = GenerationFile::capture(ws, 3, &mat, back.fingerprint);
        again.built_ms = back.built_ms;
        assert_eq!(again.to_bytes().unwrap(), bytes);
    }

    #[test]
    fn hostile_envelope_fails_loudly() {
        for evil in [
            b"not json".as_slice(),
            b"{}".as_slice(),
            br#"{"format":999,"generation":1}"#,
            br#"{"format":1,"workspace":7,"generation":3,"data":[]}"#,
        ] {
            assert!(GenerationFile::from_bytes(evil).is_err(), "{evil:?}");
        }
    }

    #[test]
    fn capture_sorts_deterministically_regardless_of_insertion_order() {
        let (a, ws) = sample();
        // Same content inserted in the opposite order must capture the same
        // envelope bytes.
        let mut b = WorkspaceIndex::new();
        b.index_file(
            WorkspaceId::new(7),
            Path::new("src/b.py"),
            b"def gamma(): pass",
            456,
        )
        .unwrap();
        b.index_file(
            WorkspaceId::new(7),
            Path::new("src/a.rs"),
            SRC.as_bytes(),
            123,
        )
        .unwrap();
        let ea = GenerationFile::capture(ws, 1, &a, vec![]);
        let eb = GenerationFile::capture(ws, 1, &b, vec![]);
        assert_eq!(ea.to_bytes().unwrap(), eb.to_bytes().unwrap());
    }

    #[test]
    fn fingerprint_order_is_canonical() {
        let mut fp = vec![
            FingerprintEntry {
                path: "z.rs".into(),
                size: 1,
                modified_ms: 1,
            },
            FingerprintEntry {
                path: "a.rs".into(),
                size: 2,
                modified_ms: 2,
            },
        ];
        fp.sort();
        let (idx, ws) = sample();
        let env = GenerationFile::capture(ws, 1, &idx, fp.clone());
        assert_eq!(env.fingerprint[0].path, "a.rs");
    }
}
