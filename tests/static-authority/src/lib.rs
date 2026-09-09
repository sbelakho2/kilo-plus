//! Static source-authority certification (audit 31/107-109).
//!
//! Three structural invariants are locked by scanning the repository's
//! *production* Rust sources (`crates/*/src`, test modules excluded):
//!
//! 1. **Child spawning** — `std::process::Command` /
//!    `tokio::process::Command` machinery exists ONLY in
//!    `crates/terminal` (the process supervisor) and `crates/pty` (the
//!    interactive-terminal platform launcher). Every other production crate
//!    must route children through the supervisor; the scans exit non-zero
//!    listing offenders with an empty default allowlist.
//! 2. **Outbound HTTP** — `reqwest::Client` construction and `.execute`
//!    calls exist ONLY inside the checked transport
//!    (`crates/provider/src/egress.rs`); every adapter send goes through
//!    the `HttpTransport` seam.
//! 3. **Durable atomic writes** — a temp-write + rename + fsync sequence
//!    (the fingerprint of a hand-rolled atomic file replacement) may only
//!    live in `crates/fs/src/atomic.rs`. Pre-existing grandfathered
//!    sequences (the CAS store, `faktor-fs`'s internal stream copy, the
//!    git worktree metadata save) are allowlisted **line-by-line** by exact
//!    content, so a NEW sequence anywhere is still listed loudly.
//!
//! Scanning methodology: per file, comments and string literals are masked
//! out and every `#[cfg(...)]`-gated item that can never compile in a
//! non-test build (`#[cfg(test)]`, `#[cfg(all(test, unix))]`, …) is
//! removed with brace-matched ranges, so markers in tests, docs or
//! examples can never certify production code. The machinery is itself
//! adversarially tested against synthetic sources.

#[cfg(test)]
mod scans {
    use std::path::Path;

    // ------------------------------------------------------------------
    // source lexing / cfg(test) stripping
    // ------------------------------------------------------------------

    /// Byte mask: `true` = semantic code (comments and string/char
    /// literals masked out, newlines preserved as irrelevant).
    fn code_mask(src: &str) -> Vec<bool> {
        let b = src.as_bytes();
        let n = b.len();
        let mut code = vec![true; n];
        let mut mask = |from: usize, to: usize| {
            let to = to.min(n);
            code[from..to].fill(false);
        };
        let mut i = 0usize;
        while i < n {
            if b[i] == b'/' && i + 1 < n && b[i + 1] == b'/' {
                let mut j = i;
                while j < n && b[j] != b'\n' {
                    j += 1;
                }
                mask(i, j);
                i = j;
            } else if b[i] == b'/' && i + 1 < n && b[i + 1] == b'*' {
                let mut j = i + 2;
                while j < n && !(b[j] == b'*' && j + 1 < n && b[j + 1] == b'/') {
                    j += 1;
                }
                if j < n {
                    j += 2;
                }
                mask(i, j);
                i = j;
            } else if (b[i] == b'r' || (b[i] == b'b' && i + 1 < n && b[i + 1] == b'r')) && {
                let mut k = i + if b[i] == b'b' { 2 } else { 1 };
                while k < n && b[k] == b'#' {
                    k += 1;
                }
                k < n && b[k] == b'"'
            } {
                // raw string r"…", r#"…"#, br#"…"#: ends at '"' + same
                // number of '#'. An unterminated raw string masks to EOF
                // (the file is not valid Rust anyway; being conservative
                // never certifies production code).
                let prefix = if b[i] == b'b' { 2 } else { 1 };
                let mut k = i + prefix;
                while k < n && b[k] == b'#' {
                    k += 1;
                }
                let hashes = k - (i + prefix);
                let mut j = k + 1; // skip the opening quote
                while j < n {
                    if b[j] == b'"' {
                        let mut h = 0usize;
                        while j + 1 + h < n && h < hashes && b[j + 1 + h] == b'#' {
                            h += 1;
                        }
                        if h == hashes {
                            j += 1 + hashes;
                            break;
                        }
                    }
                    j += 1;
                }
                mask(i, j);
                i = j;
            } else if b[i] == b'"' {
                let mut j = i + 1;
                while j < n {
                    if b[j] == b'\\' && j + 1 < n {
                        j += 2;
                    } else if b[j] == b'"' {
                        j += 1;
                        break;
                    } else {
                        j += 1;
                    }
                }
                mask(i, j);
                i = j;
            } else if b[i] == b'\'' {
                // char literal (masked) vs lifetime (kept): only mask when
                // a closing quote exists nearby.
                let mut j = i + 1;
                let mut closed = false;
                while j < n && j <= i + 12 {
                    if b[j] == b'\\' && j + 1 < n {
                        j += 2;
                        continue;
                    }
                    if b[j] == b'\'' {
                        closed = true;
                        break;
                    }
                    j += 1;
                }
                if closed {
                    mask(i, j + 1);
                    i = j + 1;
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        code
    }

    /// Three-valued evaluation of a `cfg(…)` predicate.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Tv {
        T,
        F,
        U,
    }

    fn cfg_eval(body: &str, test: bool, unknown: bool) -> Option<Tv> {
        let bytes = body.as_bytes();
        let n = bytes.len();
        let mut pos = 0usize;
        fn ident_end(bytes: &[u8], mut p: usize) -> usize {
            while p < bytes.len() && (bytes[p].is_ascii_alphanumeric() || bytes[p] == b'_') {
                p += 1;
            }
            p
        }
        fn parse(bytes: &[u8], pos: &mut usize, test: bool, unknown: bool) -> Option<Tv> {
            if bytes[*pos..].starts_with(b"not(") {
                *pos += 4;
                let v = parse(bytes, pos, test, unknown)?;
                if *pos >= bytes.len() || bytes[*pos] != b')' {
                    return None;
                }
                *pos += 1;
                return Some(match v {
                    Tv::T => Tv::F,
                    Tv::F => Tv::T,
                    Tv::U => Tv::U,
                });
            }
            for op in [&b"all("[..], &b"any("[..]] {
                if bytes[*pos..].starts_with(op) {
                    *pos += 4;
                    let mut vals = Vec::new();
                    loop {
                        vals.push(parse(bytes, pos, test, unknown)?);
                        if *pos >= bytes.len() {
                            return None;
                        }
                        if bytes[*pos] == b',' {
                            *pos += 1;
                        } else if bytes[*pos] == b')' {
                            *pos += 1;
                            break;
                        } else {
                            return None;
                        }
                    }
                    let is_all = op == &b"all("[..];
                    let mut saw_t = false;
                    let mut saw_f = false;
                    let mut saw_u = false;
                    for v in &vals {
                        match v {
                            Tv::T => saw_t = true,
                            Tv::F => saw_f = true,
                            Tv::U => saw_u = true,
                        }
                    }
                    return Some(if is_all {
                        if saw_f {
                            Tv::F
                        } else if saw_u {
                            Tv::U
                        } else {
                            Tv::T
                        }
                    } else if saw_t {
                        Tv::T
                    } else if saw_u {
                        Tv::U
                    } else {
                        Tv::F
                    });
                }
            }
            let id_end = ident_end(bytes, *pos);
            if id_end == *pos {
                return None;
            }
            let ident = std::str::from_utf8(&bytes[*pos..id_end]).ok()?;
            *pos = id_end;
            if *pos < bytes.len() && bytes[*pos] == b'=' {
                *pos += 1;
                if *pos >= bytes.len() || bytes[*pos] != b'"' {
                    return None;
                }
                *pos += 1;
                while *pos < bytes.len() && bytes[*pos] != b'"' {
                    *pos += 1;
                }
                if *pos >= bytes.len() {
                    return None;
                }
                *pos += 1;
            }
            if ident == "test" {
                return Some(if test { Tv::T } else { Tv::F });
            }
            Some(if unknown { Tv::T } else { Tv::F })
        }
        let v = parse(bytes, &mut pos, test, unknown)?;
        (pos == n).then_some(v)
    }

    /// Does the predicate mention the `test` key at all?
    fn cfg_mentions_test(body: &str) -> bool {
        let b = body.as_bytes();
        let mut i = 0usize;
        while i < b.len() {
            let mut j = i;
            while j < b.len() && (b[j].is_ascii_alphanumeric() || b[j] == b'_') {
                j += 1;
            }
            if j > i {
                let ident = &b[i..j];
                if ident == b"test" {
                    return true;
                }
                i = j;
            } else {
                i += 1;
            }
        }
        false
    }

    /// A `cfg(…)` item is test-gated when it mentions `test` and can NEVER
    /// be present in a non-test build (evaluated with the unknown keys at
    /// both extremes, since `not()` can flip either way).
    fn is_test_gated(attr_body: &str) -> bool {
        if !cfg_mentions_test(attr_body) {
            return false;
        }
        let compact: String = attr_body.chars().filter(|c| !c.is_whitespace()).collect();
        cfg_eval(&compact, false, true) == Some(Tv::F)
            && cfg_eval(&compact, false, false) == Some(Tv::F)
    }

    /// Kept (production) byte ranges: everything outside test-gated items.
    fn kept_ranges(src: &str, code: &[bool]) -> Vec<(usize, usize)> {
        let b = src.as_bytes();
        let n = b.len();
        let mut drops: Vec<(usize, usize)> = Vec::new();
        let mut i = 0usize;
        while i + 6 <= n {
            if b[i..].starts_with(b"#[cfg(") && code[i..i + 6].iter().all(|c| *c) {
                // find the matching ")]"
                let mut end = None;
                let mut j = i + 6;
                while j + 2 <= n {
                    if b[j] == b')' && b[j + 1] == b']' && code[j] && code[j + 1] {
                        end = Some(j + 2);
                        break;
                    }
                    j += 1;
                }
                let Some(after_attr) = end else {
                    break;
                };
                let body = std::str::from_utf8(&b[i + 6..after_attr - 2]).unwrap_or("");
                if !is_test_gated(body) {
                    i = after_attr;
                    continue;
                }
                // skip any further attributes attached to the same item
                let mut k = after_attr;
                loop {
                    while k < n && !code[k] {
                        k += 1;
                    }
                    if k < n && b[k..].starts_with(b"#[") {
                        let mut depth = 0usize;
                        let mut m = k;
                        while m < n {
                            if code[m] {
                                if b[m] == b'[' {
                                    depth += 1;
                                } else if b[m] == b']' {
                                    depth -= 1;
                                    if depth == 0 {
                                        m += 1;
                                        break;
                                    }
                                }
                            }
                            m += 1;
                        }
                        k = m;
                    } else {
                        break;
                    }
                }
                // locate the item: '{' block or ';' statement at depth 0
                let mut depth = 0usize;
                let mut item_end = None;
                while k < n {
                    if !code[k] {
                        k += 1;
                        continue;
                    }
                    match b[k] {
                        b'(' | b'[' => depth += 1,
                        b')' | b']' => depth = depth.saturating_sub(1),
                        b'{' if depth == 0 => {
                            let mut d = 0usize;
                            let mut m = k;
                            while m < n {
                                if code[m] {
                                    match b[m] {
                                        b'{' => d += 1,
                                        b'}' => {
                                            d -= 1;
                                            if d == 0 {
                                                item_end = Some(m + 1);
                                                break;
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                                m += 1;
                            }
                            break;
                        }
                        b';' if depth == 0 => {
                            item_end = Some(k + 1);
                            break;
                        }
                        _ => {}
                    }
                    k += 1;
                }
                match item_end {
                    Some(e) => {
                        drops.push((i, e));
                        i = e;
                    }
                    None => i = after_attr,
                }
            } else {
                i += 1;
            }
        }
        drops.sort_unstable();
        let mut kept = Vec::new();
        let mut pos = 0usize;
        for (a, z) in drops {
            if a > pos {
                kept.push((pos, a));
            }
            if z > pos {
                pos = z;
            }
        }
        if pos < n {
            kept.push((pos, n));
        }
        kept
    }

    struct File<'a> {
        rel: String,
        src: &'a str,
        code: Vec<bool>,
        kept: Vec<(usize, usize)>,
    }

    fn load(rel: &str) -> Option<File<'_>> {
        let root = repo_root();
        let path = root.join(rel);
        let src = std::fs::read_to_string(&path).ok()?;
        let code = code_mask(&src);
        let kept = kept_ranges(&src, &code);
        Some(File {
            rel: rel.to_string(),
            src: leak(&src),
            code,
            kept,
        })
    }

    /// Leak helper keeps `File` borrow-free; scans run once per test.
    fn leak(s: &str) -> &'static str {
        Box::leak(s.to_string().into_boxed_str())
    }

    fn repo_root() -> std::path::PathBuf {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = manifest
            .parent()
            .expect("tests/static-authority sits under tests/")
            .parent()
            .expect("tests/ sits under the repository root");
        assert!(
            root.join("crates").is_dir(),
            "repository root not found from {}",
            manifest.display()
        );
        root.to_path_buf()
    }

    /// Every `crates/<crate>/src/**/*.rs` file (test dirs excluded).
    fn walk_crate_sources() -> Vec<String> {
        let root = repo_root().join("crates");
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            let entries = match std::fs::read_dir(&dir) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let ft = match entry.file_type() {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if ft.is_dir() {
                    if matches!(name.as_ref(), "tests" | "examples" | "benches" | "target")
                        || name.starts_with('.')
                    {
                        continue;
                    }
                    stack.push(path);
                } else if ft.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
                    let rel = path
                        .strip_prefix(repo_root())
                        .unwrap_or(&path)
                        .display()
                        .to_string();
                    out.push(rel);
                }
            }
        }
        out.sort();
        out
    }

    /// Positions of `marker` on code lines inside kept (production)
    /// ranges.
    fn find_markers(f: &File<'_>, markers: &[&str]) -> Vec<(usize, String)> {
        let mut hits = Vec::new();
        for &m in markers {
            let mb = m.as_bytes();
            let mut pos = 0usize;
            while let Some(rel) = f.src[pos..].find(m) {
                let at = pos + rel;
                let in_kept = f.kept.iter().any(|(a, z)| at >= *a && at + mb.len() <= *z);
                let in_code = f.code[at..at + mb.len()].iter().all(|c| *c);
                if in_kept && in_code {
                    let line = line_of(f.src, at);
                    hits.push((line, trim_line(f.src, at)));
                }
                pos = at + mb.len();
            }
        }
        hits.sort_unstable();
        hits.dedup();
        hits
    }

    fn line_of(src: &str, at: usize) -> usize {
        src.as_bytes()[..at].iter().filter(|b| **b == b'\n').count() + 1
    }

    fn trim_line(src: &str, at: usize) -> String {
        let line = line_of(src, at);
        src.lines()
            .nth(line - 1)
            .map(|l| l.trim().to_string())
            .unwrap_or_default()
    }

    fn assert_no_offenders(scan: &str, offenders: &[String], scanned: usize, floor: usize) {
        assert!(scanned >= floor, "{scan}: scan walked nothing: {scanned}");
        assert!(
            offenders.is_empty(),
            "{scan} — source authority violations:\n  {}\n",
            offenders.join("\n  ")
        );
    }

    // ------------------------------------------------------------------
    // scan 1: production child spawning
    // ------------------------------------------------------------------

    /// `std::process::Command` / `tokio::process::Command` spawn machinery
    /// may exist in exactly two production homes: `crates/terminal` (the
    /// process supervisor, the single owner of children) and `crates/pty`
    /// (the interactive-terminal platform launcher wrapper). Platform
    /// launcher wrappers and tests are the ONLY listed exceptions — the
    /// default allowlist is empty, so any other production crate that
    /// starts spawning is listed loudly and fails the build.
    #[test]
    fn no_production_child_spawn_outside_terminal_and_pty_launcher() {
        const MARKERS: &[&str] = &[
            "std::process::Command",
            "tokio::process::Command",
            "process::Stdio",
            "Command::new",
            "Command::spawn",
            "CommandExt",
        ];
        let mut offenders = Vec::new();
        let mut scanned = 0usize;
        for rel in walk_crate_sources() {
            if rel.starts_with("crates/terminal/") || rel.starts_with("crates/pty/") {
                continue; // the supervisor crate + the pty launcher wrapper
            }
            let Some(f) = load(&rel) else {
                continue;
            };
            for (line, text) in find_markers(&f, MARKERS) {
                offenders.push(format!("{rel}:{line}: {text}"));
            }
            scanned += 1;
        }
        assert_no_offenders(
            "spawn scan: child spawn machinery outside crates/terminal and crates/pty \
             (every child must be routed through the ProcessSupervisor or the pty launcher)",
            &offenders,
            scanned,
            100,
        );
    }

    // ------------------------------------------------------------------
    // scan 2: production reqwest client egress
    // ------------------------------------------------------------------

    /// A raw `reqwest::Client` (construction or `.execute`) may exist ONLY
    /// inside the checked transport `crates/provider/src/egress.rs`.
    /// Adapter production code never names a client: every send goes
    /// through the `HttpTransport` seam, so the request-time destination
    /// gate applies to every provider. The default allowlist is empty.
    #[test]
    fn no_production_reqwest_client_outside_the_checked_transport() {
        const MARKERS: &[&str] = &["reqwest::Client", "Client::builder", ".execute("];
        let mut offenders = Vec::new();
        let mut scanned = 0usize;
        for rel in walk_crate_sources() {
            if rel == "crates/provider/src/egress.rs" {
                continue; // the ONE checked transport: PolicyCheckedHttpTransport
            }
            let Some(f) = load(&rel) else {
                continue;
            };
            let has_reqwest = !find_markers(&f, &["reqwest"]).is_empty();
            if !has_reqwest {
                continue; // no reqwest in production here: nothing to gate
            }
            for (line, text) in find_markers(&f, MARKERS) {
                offenders.push(format!("{rel}:{line}: {text}"));
            }
            scanned += 1;
        }
        assert_no_offenders(
            "reqwest scan: raw client construction/execute outside \
             crates/provider/src/egress.rs (every adapter send must go through the \
             HttpTransport seam)",
            &offenders,
            scanned,
            4,
        );
    }

    // ------------------------------------------------------------------
    // scan 3: new temp-write/rename/fsync sequences
    // ------------------------------------------------------------------

    /// A hand-rolled durable atomic write is the fingerprint of a
    /// temp-path write, a rename, and an fsync co-located in one file's
    /// production text. The one sanctioned home is
    /// `crates/fs/src/atomic.rs`. Pre-existing grandfathered sequences are
    /// allowlisted per exact line CONTENT (the egress-scan precedent), so
    /// the tree passes today but a NEW sequence — in any file, including
    /// grandfathered ones — is listed loudly. Default allowlist: empty.
    const ATOMIC_ANCHOR: &str = "crates/fs/src/atomic.rs";

    const ATOMIC_ALLOWLIST: &[(&str, &str)] = &[
        // crates/cas: content-addressed store writer (frozen layer below
        // crates/fs; its own documented temp+fsync+rename durability
        // contract, self-contained and seam-tested).
        (
            "crates/cas/src/lib.rs",
            "let tmp = self.tmp_path(\"stream\");",
        ),
        (
            "crates/cas/src/lib.rs",
            "fn tmp_path(&self, tag: &str) -> PathBuf {",
        ),
        ("crates/cas/src/lib.rs", "uuid::Uuid::new_v4()"),
        (
            "crates/cas/src/lib.rs",
            "let tmp = self.tmp_path(&hash.to_hex());",
        ),
        ("crates/cas/src/lib.rs", "match fs::rename(tmp, path) {"),
        ("crates/cas/src/lib.rs", "f.sync_all()?;"),
        ("crates/cas/src/lib.rs", "let _ = dir.sync_all();"),
        // crates/fs/src/lib.rs: the fs crate's own stream-copy writer
        // (copy_open_file) — same-crate internal helper that reuses the
        // atomic module's temp naming and fsync_parent.
        ("crates/fs/src/lib.rs", "uuid::Uuid::new_v4()"),
        (
            "crates/fs/src/lib.rs",
            "fs::rename(&tmp, target).map_err(|e| {",
        ),
        ("crates/fs/src/lib.rs", "out.sync_all()"),
        // crates/git: worktree metadata save (spec §33) — best-effort
        // .git-internal writer with its own unique-temp discipline.
        (
            "crates/git/src/lib.rs",
            "fn unique_meta_tmp_path(final_path: &Path) -> PathBuf {",
        ),
        (
            "crates/git/src/lib.rs",
            "let tmp = unique_meta_tmp_path(&path);",
        ),
        ("crates/git/src/lib.rs", ".create_new(true)"),
        ("crates/git/src/lib.rs", "std::fs::rename(&tmp, &path)?;"),
        (
            "crates/git/src/lib.rs",
            "f.sync_all()?; // fsync the file before it is published",
        ),
        ("crates/git/src/lib.rs", "if let Err(e) = d.sync_all() {"),
    ];

    #[test]
    fn no_new_atomic_write_sequence_outside_fs_atomic() {
        const TEMP: &[&str] = &[
            "nonce_temp",
            "tmp_path",
            "create_new(",
            "Uuid::new_v4(",
            "NamedTempFile",
            "tempdir()",
            "tempfile()",
        ];
        const RENAME: &[&str] = &["fs::rename", "std::fs::rename", ".rename("];
        const SYNC: &[&str] = &["sync_all", "sync_data", "fsync("];
        let mut offenders: Vec<String> = Vec::new();
        let mut scanned = 0usize;
        for rel in walk_crate_sources() {
            if rel == ATOMIC_ANCHOR {
                continue; // the one sanctioned home of the sequence
            }
            let Some(f) = load(&rel) else {
                continue;
            };
            let has_temp = !find_markers(&f, TEMP).is_empty();
            let has_rename = !find_markers(&f, RENAME).is_empty();
            let has_sync = !find_markers(&f, SYNC).is_empty();
            if !(has_temp && has_rename && has_sync) {
                continue;
            }
            for (line, text) in find_markers(
                &f,
                &TEMP
                    .iter()
                    .chain(RENAME)
                    .chain(SYNC)
                    .copied()
                    .collect::<Vec<_>>(),
            ) {
                let allowlisted = ATOMIC_ALLOWLIST
                    .iter()
                    .any(|(p, t)| *p == rel && *t == text);
                if !allowlisted {
                    offenders.push(format!("{rel}:{line}: {text}"));
                }
            }
            scanned += 1;
        }
        assert_no_offenders(
            "atomic-write scan: a temp-write/rename/fsync sequence exists outside \
             crates/fs/src/atomic.rs (route file-content replacement through \
             faktor_fs::atomic)",
            &offenders,
            scanned,
            3,
        );
    }

    // ------------------------------------------------------------------
    // adversarial tests of the machinery itself
    // ------------------------------------------------------------------

    #[test]
    fn cfg_stripper_removes_every_test_module_shape() {
        // Synthetic files: markers in test modules, raw strings, docs,
        // cfg(all(test, unix)) blocks must never survive as production.
        let src = r##"
//! docs with #[cfg(test)] and std::process::Command mentions
use std::process::Command as C;
fn production_spawn() { let _c = Command::new("x"); }
#[cfg(all(test, unix))]
mod unix_tests {
    fn spawn() { let _ = std::process::Command::new("t"); }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn t() { let _ = std::process::Command::new("t"); }
    const RAW: &str = r#"std::process::Command fake"#;
}
#[cfg(test)]
use std::sync::OnceLock;
#[cfg(test)]
static SEAM: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
fn after() { let _ = C::new("y"); }
"##;
        let code = code_mask(src);
        let kept = kept_ranges(src, &code);
        let mut prod = String::new();
        for (a, z) in kept {
            prod.push_str(&src[a..z]);
        }
        assert!(prod.contains("fn production_spawn"));
        assert!(prod.contains("fn after"));
        assert!(
            !prod.contains("mod unix_tests") && !prod.contains("mod tests"),
            "test modules leaked into production text:\n{prod}"
        );
        assert!(
            !prod.contains("std::process::Command::new(\"t\")"),
            "markers in tests leaked"
        );
        assert!(!prod.contains("OnceLock"), "cfg(test) items leaked");
        // Raw-string and doc-comment mentions are masked (not code).
        assert!(code_mask(r##"let s = r#"std::process::Command"#;"##)
            .iter()
            .any(|c| !c));
    }

    #[test]
    fn cfg_stripper_keeps_cfg_platform_blocks() {
        // cfg(unix)/cfg(not(unix))/cfg(any(test, unix)) production shapes
        // must NOT be stripped, and pure `cfg(any(test, unix))` items that
        // also exist on unix production are kept (conservative direction).
        let src = r##"
#[cfg(unix)]
fn unix_alive() { let _ = std::process::Command::new("ps"); }
#[cfg(not(unix))]
fn win_alive() { let _ = std::process::Command::new("tasklist"); }
#[cfg(any(test, unix))]
fn both() {}
#[cfg(not(test))]
fn prod_only() {}
"##;
        let code = code_mask(src);
        let kept = kept_ranges(src, &code);
        let prod: String = kept.iter().map(|(a, z)| &src[*a..*z]).collect();
        for needle in ["unix_alive", "win_alive", "both", "prod_only"] {
            assert!(prod.contains(needle), "{needle} must survive");
        }
    }

    #[test]
    fn scan_floors_guard_against_an_empty_walk() {
        // The scans assert a minimum file count; prove the walk finds the
        // real tree (a silently skipped tree would pass vacuously).
        let files = walk_crate_sources();
        assert!(files.len() >= 100, "walk too small: {}", files.len());
        assert!(files.contains(&"crates/security/src/lib.rs".to_string()));
        assert!(files.contains(&"crates/provider/src/egress.rs".to_string()));
        assert!(
            files
                .iter()
                .all(|f| !f.starts_with("crates/") || f.contains("/src/")),
            "only crate src trees may certify production code"
        );
    }

    #[test]
    fn allowlist_lines_that_no_longer_exist_are_dead_entries() {
        // Grandfathered allowlist entries must keep matching real lines:
        // a dead entry means the sequence was removed or moved, and the
        // entry should be cleaned up or the move audited.
        for (rel, text) in ATOMIC_ALLOWLIST {
            let root = repo_root();
            let src = std::fs::read_to_string(root.join(rel))
                .unwrap_or_else(|_| panic!("allowlisted file missing: {rel}"));
            let found = src.lines().any(|l| l.trim() == *text);
            assert!(
                found,
                "stale atomic-write allowlist entry {rel}: {text:?} — the grandfathered \
                 sequence moved or was removed; update the allowlist"
            );
        }
    }

    /// Synthetic-file negative tests: prove each scan FIRES on a violation
    /// (the machinery is not vacuously green).
    fn synthetic_file(rel: &str, src: &str) -> File<'static> {
        let code = code_mask(src);
        let kept = kept_ranges(src, &code);
        File {
            rel: rel.to_string(),
            src: leak(src),
            code,
            kept,
        }
    }

    #[test]
    fn spawn_scan_fires_on_a_violating_crate() {
        const MARKERS: &[&str] = &[
            "std::process::Command",
            "tokio::process::Command",
            "process::Stdio",
            "Command::new",
            "Command::spawn",
            "CommandExt",
        ];
        let f = synthetic_file(
            "crates/git/src/lib.rs",
            "fn run() { let c = std::process::Command::new(\"git\"); }\n",
        );
        let hits = find_markers(&f, MARKERS);
        assert!(!hits.is_empty(), "a production spawn must be flagged");
        // The sanctioned homes are the only silent files.
        for allowed in ["crates/terminal/src/lib.rs", "crates/pty/src/unix.rs"] {
            let f = synthetic_file(
                allowed,
                "fn run() { let _ = std::process::Command::new(\"x\"); }\n",
            );
            let hits = find_markers(&f, MARKERS);
            assert!(!hits.is_empty(), "marker detection must still work");
        }
        // A spawn buried in a #[cfg(test)] module must NOT fire.
        let f = synthetic_file(
            "crates/git/src/lib.rs",
            "fn ok() {}\n#[cfg(test)] mod tests {\n  fn t() { let _ = std::process::Command::new(\"x\"); }\n}\n",
        );
        assert!(find_markers(&f, MARKERS).is_empty());
    }

    #[test]
    fn reqwest_scan_fires_on_a_violating_adapter() {
        const MARKERS: &[&str] = &["reqwest::Client", "Client::builder", ".execute("];
        let f = synthetic_file(
            "crates/openai/src/lib.rs",
            "use reqwest::Client;\npub fn send() { let c = Client::new(); let _ = c.execute(r); }\n",
        );
        let has_reqwest = !find_markers(&f, &["reqwest"]).is_empty();
        assert!(has_reqwest);
        let hits = find_markers(&f, MARKERS);
        assert!(
            hits.iter()
                .any(|(_, t)| t.contains("Client::new") || t.contains(".execute(")),
            "raw client code in an adapter must be flagged: {hits:?}"
        );
        // A file whose production text never mentions reqwest is not gated.
        let f = synthetic_file(
            "crates/store/src/lib.rs",
            "fn q() { conn.execute(\"SELECT 1\", []).unwrap(); }\n",
        );
        assert!(find_markers(&f, &["reqwest"]).is_empty());
    }

    #[test]
    fn atomic_scan_fires_on_a_new_hand_rolled_sequence() {
        const TEMP: &[&str] = &[
            "nonce_temp",
            "tmp_path",
            "create_new(",
            "Uuid::new_v4(",
            "NamedTempFile",
            "tempdir()",
            "tempfile()",
        ];
        const RENAME: &[&str] = &["fs::rename", "std::fs::rename", ".rename("];
        const SYNC: &[&str] = &["sync_all", "sync_data", "fsync("];
        // A brand-new dance in a clean crate: temp write + fsync + rename.
        let f = synthetic_file(
            "crates/agent/src/lib.rs",
            "fn save() {\n  let tmp = format!(\"x{}\", uuid::Uuid::new_v4());\n  let mut f = std::fs::File::create(&tmp).unwrap();\n  f.sync_all().unwrap();\n  fs::rename(&tmp, path).unwrap();\n}\n",
        );
        let fams = [
            find_markers(&f, TEMP),
            find_markers(&f, RENAME),
            find_markers(&f, SYNC),
        ];
        assert!(fams.iter().all(|h| !h.is_empty()), "{fams:?}");
        let mut offenders: Vec<String> = Vec::new();
        for fam in &fams {
            for (line, text) in fam {
                let allowlisted = ATOMIC_ALLOWLIST
                    .iter()
                    .any(|(p, t)| *p == f.rel && *t == *text);
                if !allowlisted {
                    offenders.push(format!("{}:{line}: {text}", f.rel));
                }
            }
        }
        assert!(
            !offenders.is_empty(),
            "a fresh atomic-write sequence must be listed loudly"
        );
        assert!(
            offenders
                .iter()
                .any(|o| o.contains("crates/agent/src/lib.rs")),
            "the offender must name the violating file: {offenders:?}"
        );
    }
}
