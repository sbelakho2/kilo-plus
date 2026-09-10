//! Seed corpus sourcing (audit P0-75/P0-81): fuzz seeds come from the
//! repository's EXISTING fixtures, never from invented blobs.
//!
//! Two checked-in fixture corpora feed the deterministic campaign and the
//! `fuzz/seed-corpus.sh` generator:
//!
//! - `compat/kilo-v756/` — the frozen v7.5.6 wire goldens (compat DTOs,
//!   SSE frames, global event envelopes, provider lists, ...);
//! - `fixtures/providers/` — recorded provider stream bodies (OpenAI,
//!   Anthropic, Gemini, Ollama), the line-framing/SSE source shapes.
//!
//! Loading is lazy and process-wide; a missing directory yields no seeds
//! (the caller reports that honestly instead of failing the harness).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// One fixture file: repository-relative name + exact bytes.
#[derive(Debug, Clone)]
pub struct FixtureSeed {
    pub name: String,
    pub bytes: Vec<u8>,
}

const SEED_DIRS: &[&str] = &["compat/kilo-v756", "fixtures/providers"];

/// Repository root computed from this crate's manifest dir
/// (`tests/fuzz-seeds` -> `../..`).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("tests/fuzz-seeds always has a repository root")
        .to_path_buf()
}

fn push_json_seeds(dir: &Path, root: &Path, out: &mut Vec<FixtureSeed>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    for path in files {
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let name = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        out.push(FixtureSeed { name, bytes });
    }
}

/// Every fixture seed, sorted by path (deterministic across runs).
pub fn fixture_seeds() -> &'static [FixtureSeed] {
    static SEEDS: OnceLock<Vec<FixtureSeed>> = OnceLock::new();
    SEEDS.get_or_init(|| {
        let root = repo_root();
        let mut out = Vec::new();
        for dir in SEED_DIRS {
            push_json_seeds(&root.join(dir), &root, &mut out);
        }
        out
    })
}

/// One fixture by repository-relative name.
pub fn fixture_bytes(name: &str) -> Option<&'static [u8]> {
    fixture_seeds()
        .iter()
        .find(|s| s.name == name)
        .map(|s| s.bytes.as_slice())
}
