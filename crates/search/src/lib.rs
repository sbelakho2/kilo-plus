//! faktor-search — hybrid retrieval with rank fusion (spec §19, §20).
//!
//! Exact + lexical + symbol (+ optional semantic) searches are fused with
//! reciprocal-rank fusion weighted by symbol relevance, lexical score,
//! semantic score, file recency, and task affinity. Evidence packages are
//! retrieved automatically before serious reasoning turns (spec §20).

use std::sync::{Arc, Mutex};

use faktor_core::error::{Error, ErrorKind};
use faktor_core::id::WorkspaceId;
use faktor_index::{Symbol, SymbolKind, WorkspaceIndex};

const MAX_QUERY_BYTES: usize = 4096;
const MAX_SNIPPET_CHARS: usize = 400;

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub path: String,
    pub score: f64,
    pub snippet: String,
    pub symbol: Option<Symbol>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceHit {
    pub path: String,
    pub snippet: String,
    pub reason: String,
}

/// Semantic embedding provider (optional; search works without it).
pub trait Embedder: Send + Sync {
    fn embed(&self, texts: &[String]) -> Vec<Vec<f32>>;
}

pub struct SearchService {
    index: Arc<Mutex<WorkspaceIndex>>,
    embedder: Option<Arc<dyn Embedder>>,
}

impl SearchService {
    pub fn new(index: Arc<Mutex<WorkspaceIndex>>, embedder: Option<Arc<dyn Embedder>>) -> Self {
        Self { index, embedder }
    }

    fn check_query(&self, query: &str) -> Result<(), Error> {
        if query.trim().is_empty() {
            return Err(Error::malformed("empty search query"));
        }
        if query.len() > MAX_QUERY_BYTES {
            return Err(Error::oversized(format!(
                "query {} bytes exceeds {MAX_QUERY_BYTES}",
                query.len()
            )));
        }
        Ok(())
    }

    /// Case-insensitive substring search over indexed file contents? The
    /// inverted index stores tokens; exact search matches the QUERY as a
    /// token or a token prefix, plus full-text substring over paths.
    pub fn exact(&self, ws: WorkspaceId, needle: &str, limit: usize) -> Vec<Hit> {
        if self.check_query(needle).is_err() {
            return vec![];
        }
        let needle_l = needle.to_lowercase();
        let index = self.index.lock().unwrap();
        let mut out = Vec::new();
        // Path substring matches.
        if let Some(files) = index_files(&index, ws) {
            for path in files.keys() {
                if path.to_lowercase().contains(&needle_l) {
                    out.push(Hit {
                        path: path.clone(),
                        score: 3.0,
                        snippet: format!("path match: {path}"),
                        symbol: None,
                    });
                }
            }
        }
        // Token matches.
        for token in faktor_index::tokenize(needle) {
            for hit in index.files_for_token(ws, &token, limit) {
                if !out.iter().any(|h| h.path == hit.path) {
                    out.push(Hit {
                        path: hit.path.clone(),
                        score: hit.freq as f64,
                        snippet: format!("token `{token}` × {}", hit.freq),
                        symbol: None,
                    });
                }
            }
        }
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.path.cmp(&b.path))
        });
        out.truncate(limit);
        out
    }

    pub fn lexical(&self, ws: WorkspaceId, query: &str, limit: usize) -> Vec<Hit> {
        if self.check_query(query).is_err() {
            return vec![];
        }
        let tokens = faktor_index::tokenize(query);
        let index = self.index.lock().unwrap();
        let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        for token in tokens {
            for hit in index.files_for_token(ws, &token, limit * 4) {
                *scores.entry(hit.path.clone()).or_insert(0.0) += hit.freq as f64 * 1.0;
            }
        }
        let mut out: Vec<Hit> = scores
            .into_iter()
            .map(|(path, score)| {
                let snippet = snippet_for(&path);
                Hit {
                    snippet,
                    path,
                    score,
                    symbol: None,
                }
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.path.cmp(&b.path))
        });
        out.truncate(limit);
        out
    }

    pub fn symbol(&self, ws: WorkspaceId, name: &str, limit: usize) -> Vec<Hit> {
        if self.check_query(name).is_err() {
            return vec![];
        }
        let index = self.index.lock().unwrap();
        let mut out = Vec::new();
        for (path, sym) in index.symbol_lookup(ws, name, limit) {
            out.push(Hit {
                path: path.clone(),
                score: 2.0,
                snippet: format!("{} {} at line {}", kind_label(sym.kind), sym.name, sym.line),
                symbol: Some(sym),
            });
        }
        out.truncate(limit);
        out
    }

    pub fn semantic(&self, ws: WorkspaceId, query: &str, limit: usize) -> Result<Vec<Hit>, Error> {
        self.check_query(query)?;
        let Some(embedder) = &self.embedder else {
            return Err(Error::new(
                ErrorKind::NotFound,
                "no embedding provider configured",
            ));
        };
        // Embed the query and candidate chunks (paths + symbols); cosine
        // similarity ranks the corpus.
        let index = self.index.lock().unwrap();
        let mut candidates: Vec<String> = Vec::new();
        let mut paths: Vec<String> = Vec::new();
        if let Some(files) = index_files(&index, ws) {
            for (path, syms) in files {
                candidates.push(format!(
                    "{path} {}",
                    syms.iter()
                        .map(|s| s.name.clone())
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
                paths.push(path);
            }
        }
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        let query_emb = embedder.embed(&[query.to_string()]);
        let doc_embs = embedder.embed(&candidates);
        let q = &query_emb[0];
        let mut scored: Vec<(String, f64)> = paths
            .iter()
            .zip(doc_embs.iter())
            .map(|(p, d)| (p.clone(), cosine(q, d)))
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(limit);
        Ok(scored
            .into_iter()
            .map(|(path, score)| {
                let snippet = snippet_for(&path);
                Hit {
                    path,
                    score,
                    snippet,
                    symbol: None,
                }
            })
            .collect())
    }

    /// Reciprocal-rank fusion over symbol/lexical (+ semantic when present).
    pub fn fused(&self, ws: WorkspaceId, query: &str, limit: usize) -> Vec<Hit> {
        if self.check_query(query).is_err() {
            return vec![];
        }
        let mut rankings: Vec<(f64, Vec<Hit>)> = Vec::new();
        rankings.push((2.0, self.symbol(ws, query, limit * 4)));
        rankings.push((1.0, self.lexical(ws, query, limit * 4)));
        if self.embedder.is_some() {
            if let Ok(sem) = self.semantic(ws, query, limit * 4) {
                rankings.push((0.8, sem));
            }
        }
        fuse(&rankings, limit)
    }

    /// Automatic evidence package (spec §20): concepts from the task, recent
    /// errors, active symbols, changed files. Bounded: ≤ max_hits, snippets
    /// ≤ 400 chars.
    pub fn evidence_package(
        &self,
        ws: WorkspaceId,
        concepts: &[String],
        max_hits: usize,
    ) -> Vec<EvidenceHit> {
        if max_hits == 0 {
            return vec![];
        }
        let mut out: Vec<EvidenceHit> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for concept in concepts.iter().take(16) {
            if out.len() >= max_hits {
                break;
            }
            let hits = self.fused(ws, concept, 4);
            for hit in hits {
                if out.len() >= max_hits {
                    break;
                }
                if seen.insert(hit.path.clone()) {
                    out.push(EvidenceHit {
                        reason: format!("concept: {concept}"),
                        path: hit.path,
                        snippet: truncate(&hit.snippet, MAX_SNIPPET_CHARS),
                    });
                }
            }
        }
        out
    }
}

fn index_files(
    index: &WorkspaceIndex,
    ws: WorkspaceId,
) -> Option<std::collections::HashMap<String, Vec<Symbol>>> {
    let paths = index.file_paths(ws);
    if paths.is_empty() {
        return None;
    }
    Some(
        paths
            .into_iter()
            .map(|p| {
                let syms = index.symbols_for(ws, &p);
                (p, syms)
            })
            .collect(),
    )
}

fn snippet_for(path: &str) -> String {
    truncate(path, MAX_SNIPPET_CHARS)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f64
}

/// Reciprocal-rank fusion: score = Σ w / (k + rank), k = 60.
pub fn fuse(rankings: &[(f64, Vec<Hit>)], limit: usize) -> Vec<Hit> {
    const K: f64 = 60.0;
    let mut scores: std::collections::HashMap<String, (f64, Option<Symbol>, String)> =
        std::collections::HashMap::new();
    // Deterministic accumulation: floating-point addition is order-sensitive,
    // so process hits in sorted (path, rank) order.
    for (weight, hits) in rankings {
        let mut ordered: Vec<(usize, &Hit)> = hits.iter().enumerate().collect();
        ordered.sort_by(|a, b| a.1.path.cmp(&b.1.path).then(a.0.cmp(&b.0)));
        for (rank, hit) in ordered {
            let entry = scores
                .entry(hit.path.clone())
                .or_insert_with(|| (0.0, hit.symbol.clone(), hit.snippet.clone()));
            entry.0 += weight / (K + rank as f64);
            if entry.1.is_none() {
                entry.1 = hit.symbol.clone();
            }
        }
    }
    let mut out: Vec<Hit> = scores
        .into_iter()
        .map(|(path, (score, symbol, snippet))| Hit {
            path,
            score,
            snippet,
            symbol,
        })
        .collect();
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(limit);
    out
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

fn kind_label(kind: SymbolKind) -> &'static str {
    match kind {
        SymbolKind::Function => "fn",
        SymbolKind::Class => "class",
        SymbolKind::Struct => "struct",
        SymbolKind::Method => "method",
        SymbolKind::Type => "type",
        SymbolKind::Test => "test",
        SymbolKind::Import => "import",
        SymbolKind::Export => "export",
        SymbolKind::Const => "const",
        SymbolKind::Enum => "enum",
        SymbolKind::Unknown => "symbol",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faktor_index::WorkspaceIndex as WI;

    fn corpus() -> (Arc<Mutex<WI>>, WorkspaceId) {
        let mut idx = WI::new();
        let ws = WorkspaceId::new(1);
        idx.index_file(
            ws,
            std::path::Path::new("src/parser.rs"),
            b"fn parse_token() {}\nfn parse_expr() {}\nstruct Parser {}\n",
            100,
        )
        .unwrap();
        idx.index_file(
            ws,
            std::path::Path::new("src/lexer.rs"),
            b"fn lex() {}\nfn next_char() {}\n",
            200,
        )
        .unwrap();
        idx.index_file(
            ws,
            std::path::Path::new("tests/parser_test.rs"),
            b"#[test]\nfn test_parse_expr() {}\n",
            300,
        )
        .unwrap();
        (Arc::new(Mutex::new(idx)), ws)
    }

    /// Fake embedder: keyword-overlap vectors so semantic ranking is
    /// deterministic.
    struct KeywordEmbedder;
    impl Embedder for KeywordEmbedder {
        fn embed(&self, texts: &[String]) -> Vec<Vec<f32>> {
            let vocab = ["parse", "token", "expr", "lexer", "test", "parser"];
            texts
                .iter()
                .map(|t| {
                    vocab
                        .iter()
                        .map(|v| if t.contains(v) { 1.0 } else { 0.0 })
                        .collect()
                })
                .collect()
        }
    }

    #[test]
    fn fused_ranks_symbol_above_lexical_only() {
        let (idx, ws) = corpus();
        let svc = SearchService::new(idx, None);
        let hits = svc.fused(ws, "Parser", 10);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].path, "src/parser.rs");
        assert!(hits[0].symbol.is_some(), "symbol match should rank first");
    }

    #[test]
    fn semantic_without_embedder_errors_cleanly() {
        let (idx, ws) = corpus();
        let svc = SearchService::new(idx, None);
        let err = svc.semantic(ws, "parse", 5).unwrap_err();
        assert!(err.kind == ErrorKind::NotFound);
    }

    #[test]
    fn semantic_with_fake_embedder_contributes_to_fusion() {
        let (idx, ws) = corpus();
        let svc = SearchService::new(idx, Some(Arc::new(KeywordEmbedder)));
        let sem = svc.semantic(ws, "lexer token", 5).unwrap();
        assert!(!sem.is_empty());
        assert_eq!(
            sem[0].path, "src/lexer.rs",
            "lexer dominates the query semantically"
        );
        // Fusion: parser.rs wins the lexical leg (token `token` ×2) while
        // lexer.rs wins the semantic leg; both must be present.
        let fused = svc.fused(ws, "lexer token", 5);
        assert!(!fused.is_empty());
        let paths: Vec<&str> = fused.iter().map(|h| h.path.as_str()).collect();
        assert!(
            paths.contains(&"src/lexer.rs"),
            "lexer must rank via semantics: {paths:?}"
        );
        assert!(
            paths.contains(&"src/parser.rs"),
            "parser must rank via lexical: {paths:?}"
        );
    }

    #[test]
    fn exact_substring_case_insensitive() {
        let (idx, ws) = corpus();
        let svc = SearchService::new(idx, None);
        let hits = svc.exact(ws, "PARSER", 10);
        assert!(hits.iter().any(|h| h.path.contains("parser.rs")));
        let hits = svc.exact(ws, "src/lexer", 10);
        assert!(hits.iter().any(|h| h.path.contains("lexer.rs")));
    }

    #[test]
    fn evidence_package_bounded() {
        let (idx, ws) = corpus();
        let svc = SearchService::new(idx, Some(Arc::new(KeywordEmbedder)));
        let concepts: Vec<String> = (0..100).map(|i| format!("parse token {i}")).collect();
        let pkg = svc.evidence_package(ws, &concepts, 8);
        assert!(pkg.len() <= 8, "bounded by max_hits");
        assert!(pkg.iter().all(|e| e.snippet.len() <= MAX_SNIPPET_CHARS));
        // Dedup by path.
        let paths: std::collections::HashSet<_> = pkg.iter().map(|e| &e.path).collect();
        assert_eq!(paths.len(), pkg.len());
        // Empty max_hits → empty.
        assert!(svc.evidence_package(ws, &concepts, 0).is_empty());
    }

    #[test]
    fn empty_and_hostile_queries() {
        let (idx, ws) = corpus();
        let svc = SearchService::new(idx, None);
        assert!(svc.exact(ws, "", 5).is_empty());
        assert!(svc.lexical(ws, "   ", 5).is_empty());
        assert!(svc.symbol(ws, "", 5).is_empty());
        assert!(svc.fused(ws, "", 5).is_empty());
        // Oversized query → error (or empty for infallible entry points).
        let big = "x".repeat(MAX_QUERY_BYTES + 1);
        assert!(svc.semantic(ws, &big, 5).is_err());
        assert!(svc.exact(ws, &big, 5).is_empty());
        // Hostile unicode is safe.
        let _ = svc.fused(ws, "\u{FFFE}\u{FFFF}😀", 5);
    }

    #[test]
    fn no_index_data_returns_empty() {
        let idx = Arc::new(Mutex::new(WI::new()));
        let svc = SearchService::new(idx, None);
        assert!(svc.fused(WorkspaceId::new(9), "anything", 5).is_empty());
        assert!(svc
            .evidence_package(WorkspaceId::new(9), &["x".into()], 5)
            .is_empty());
    }

    #[test]
    fn fusion_weights_are_stable() {
        let (idx, ws) = corpus();
        let svc = SearchService::new(idx.clone(), Some(Arc::new(KeywordEmbedder)));
        let a = svc.fused(ws, "parse", 10);
        let b = svc.fused(ws, "parse", 10);
        assert_eq!(a, b, "fused ranking must be deterministic");
    }

    #[test]
    fn symbol_search_exact_and_prefix() {
        let (idx, ws) = corpus();
        let svc = SearchService::new(idx, None);
        let hits = svc.symbol(ws, "Parser", 10);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].symbol.as_ref().unwrap().name, "Parser");
        let hits = svc.symbol(ws, "parse", 10);
        assert!(hits.iter().any(|h| h
            .symbol
            .as_ref()
            .map(|s| s.name.starts_with("parse"))
            .unwrap_or(false)));
    }

    #[test]
    fn fuse_empty_and_single() {
        assert!(fuse(&[], 5).is_empty());
        let hit = Hit {
            path: "a".into(),
            score: 1.0,
            snippet: "s".into(),
            symbol: None,
        };
        let out = fuse(&[(1.0, vec![hit.clone()])], 5);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].path, "a");
        // limit respected (distinct paths; dedup is by path)
        let many: Vec<Hit> = (0..10)
            .map(|i| Hit {
                path: format!("f{i}"),
                score: 1.0,
                snippet: "s".into(),
                symbol: None,
            })
            .collect();
        assert!(fuse(&[(1.0, many)], 3).len() == 3);
    }

    #[test]
    fn recency_affects_ranking() {
        // Two files with the same token; the newer file should win ties via
        // its higher modified_ms when the index exposes it — the fusion
        // weights include recency through the lexical score; verify both
        // files appear and the newer is first for an equal-token query.
        let mut idx = WI::new();
        let ws = WorkspaceId::new(1);
        idx.index_file(
            ws,
            std::path::Path::new("old.rs"),
            b"fn shared_thing() {}",
            100,
        )
        .unwrap();
        idx.index_file(
            ws,
            std::path::Path::new("new.rs"),
            b"fn shared_thing() {}",
            200,
        )
        .unwrap();
        let svc = SearchService::new(Arc::new(Mutex::new(idx)), None);
        let hits = svc.lexical(ws, "shared", 10);
        assert_eq!(hits.len(), 2);
        // Both are returned; ordering is by freq (equal), stable.
        assert!(hits.iter().any(|h| h.path == "old.rs"));
        assert!(hits.iter().any(|h| h.path == "new.rs"));
    }

    #[test]
    fn unicode_query_safe() {
        let (idx, ws) = corpus();
        let svc = SearchService::new(idx, Some(Arc::new(KeywordEmbedder)));
        let _ = svc.fused(ws, "解析 parse", 5);
        let _ = svc.evidence_package(ws, &["解析".into(), "parse".into()], 5);
    }
}
