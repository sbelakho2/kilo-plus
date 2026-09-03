# faktor-index + faktor-search specs (spec §19, §20)

## Part A — faktor-index: hybrid index

Crate: crates/index (faktor-core, faktor-store, tree-sitter 0.27, tree-sitter-rust, tree-sitter-python, tokio).

```rust
pub struct IndexEntry { pub path: String, pub tokens: Vec<String>, pub symbols: Vec<Symbol>, pub modified_ms: i64, pub size: u64 }
pub struct Symbol { pub name: String, pub kind: SymbolKind, pub line: u32, pub doc: Option<String> }
pub enum SymbolKind { Function, Class, Struct, Method, Type, Test, Import, Export, Const, Enum, Unknown }
pub struct WorkspaceIndex { /* postings: token → path → freq; symbols: path → vec */ }
impl WorkspaceIndex {
    pub fn new(store: Arc<Store>) -> Self;   // optional persistence
    pub fn index_file(&mut self, workspace_id: WorkspaceId, rel: &Path, bytes: &[u8]) -> Result<()>;
    pub fn remove_file(&mut self, workspace_id: WorkspaceId, rel: &Path);
    pub fn files_for_token(&self, workspace_id: WorkspaceId, token: &str, limit: usize) -> Vec<(String, u32)>; // (path, freq)
    pub fn symbols_in(&self, workspace_id: WorkspaceId, rel: &Path) -> Vec<Symbol>;
    pub fn symbol_lookup(&self, workspace_id: WorkspaceId, name: &str, limit: usize) -> Vec<(String, Symbol)>;
    pub fn token_count(&self) -> usize;
}
```
- Lexical index: own inverted index (tokenize: alnum runs ≥2 chars, lowercase; ignore stopwords is a,an,the,is,are,to,of,and,or,for,on,with). Bounded: max 1M unique tokens (evict rarest), max 100k files per workspace (drop oldest).
- Syntax index: tree-sitter-rust and tree-sitter-python: extract function_item/function_definition, struct_item/class_definition, impl_item, enum_item, const_item, method calls (names), test attributes (#[test], def test_*). Unknown languages → Unknown kind only for identifiers heuristically? No: skip with zero symbols.
- Incremental: index_file replaces a file's previous entries (store old tokens via path map).
- Adversarial tests:
  1. index_roundtrip_and_lookup
  2. reindex_replaces_old_entries (edit a file; old tokens gone)
  3. remove_file_clears_entries
  4. tokenize_hostile_inputs (binary, huge single token 1MB, unicode, empty, only punctuation)
  5. stopwords_not_indexed
  6. rust_and_python_symbols_extracted (fixture sources with fn/struct/class/def/impl/const/#[test])
  7. malformed_source_never_panics (truncated fn, garbage bytes, unmatched braces)
  8. concurrent_index_updates_are_consistent (4 threads × 500 files, then query: no lost updates, token_count sane)
  9. index_bounded_at_caps (insert 120k files → 100k cap; 2M tokens → cap)
  10. workspace_isolation (two workspaces, same token: no cross talk)
  11. unknown_language_no_symbols_but_tokens_indexed

## Part B — faktor-search: retrieval + fusion

Crate: crates/search (faktor-core, faktor-index, faktor-provider for the Embedder trait — define a local `Embedder` trait so search stays usable without a provider):

```rust
pub trait Embedder: Send + Sync { fn embed(&self, texts: &[String]) -> Vec<Vec<f32>>; }  // semantic optional
pub struct SearchService { index: Arc<Mutex<WorkspaceIndex>>, embedder: Option<Arc<dyn Embedder>> }
impl SearchService {
    pub fn new(index: Arc<Mutex<WorkspaceIndex>>, embedder: Option<Arc<dyn Embedder>>) -> Self;
    pub fn exact(&self, ws: WorkspaceId, needle: &str, limit: usize) -> Vec<Hit>;          // substring (case-insensitive) over indexed paths
    pub fn lexical(&self, ws: WorkspaceId, query: &str, limit: usize) -> Vec<Hit>;         // inverted index scoring
    pub fn symbol(&self, ws: WorkspaceId, name: &str, limit: usize) -> Vec<Hit>;
    pub fn semantic(&self, ws: WorkspaceId, query: &str, limit: usize) -> Result<Vec<Hit>>; // Err when no embedder
    pub fn fused(&self, ws: WorkspaceId, query: &str, limit: usize) -> Vec<Hit>;  // reciprocal-rank fusion of symbol+lexical(+semantic if available)
    /// Automatic evidence package (spec §20): concepts from prompt/errors/changed files → fused hits, bounded.
    pub fn evidence_package(&self, ws: WorkspaceId, concepts: &[String], max_hits: usize) -> Vec<EvidenceHit>;
}
pub struct Hit { pub path: String, pub score: f64, pub snippet: String, pub symbol: Option<Symbol> }
```
- Fusion: reciprocal rank fusion with weights [symbol 2.0, lexical 1.0, semantic 0.8, recency 0.5] — implement a `fuse(rankings: &[Vec<Hit>], weights)`.
- Evidence package bounded: ≤ max_hits (default 8), each snippet ≤ 400 chars.
- Adversarial tests:
  1. fused_ranks_symbol_match_above_lexical_only
  2. exact_substring_case_insensitive
  3. semantic_without_embedder_errors_cleanly
  4. semantic_with_fake_embedder_contributes_to_fusion (fake embedder: keyword overlap cosine)
  5. evidence_package_bounded (100 concepts → ≤8 hits, snippets ≤400)
  6. empty_query_returns_empty (never panics)
  7. hostile_query_1mb_rejected (Oversized)
  8. fusion_weights_are_stable (rank order for a fixed corpus is deterministic)
  9. recency_affects_ranking (newer file wins ties)
  10. symbol_search_exact_and_prefix
  11. no_index_data_returns_empty
  12. unicode_query_safe

Build/test each crate green, zero warnings. Do NOT modify other crates. Do NOT commit.
