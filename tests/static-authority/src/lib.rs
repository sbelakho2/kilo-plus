//! Static source-authority certification (audit 31/107-109).
//!
//! Six structural invariants are locked by scanning the repository's
//! *production* Rust sources (`crates/*/src`, test modules and out-of-line
//! `#[cfg(test)] mod` bodies excluded):
//!
//! 1. **Child spawning** — `std::process::Command` /
//!    `tokio::process::Command` machinery exists ONLY in
//!    `crates/terminal` (the process supervisor) and `crates/pty` (the
//!    interactive-terminal platform launcher). Every other production crate
//!    must route children through the supervisor; the scans exit non-zero
//!    listing offenders with an empty default allowlist. An INDIRECT
//!    `ResolvedCommand`/`.lower*` -> process-`Command` conversion (lowering a
//!    command spec and then spawning it directly) is detected by a
//!    co-location scan even when no spawn marker names the type directly.
//! 2. **Outbound HTTP** — `reqwest::Client` construction and `.execute`
//!    calls exist ONLY inside the checked transport
//!    (`crates/provider/src/egress.rs`); every adapter send goes through
//!    the `HttpTransport` seam.
//! 3. **Durable atomic writes** — the forbidden patterns are checked
//!    INDEPENDENTLY (no fsync marker required): any `std::fs::rename` /
//!    `fs::rename` / `tokio::fs::rename` / `NamedTempFile::persist`, and any
//!    write-family call targeting a temp path, must either live in
//!    `crates/fs/src/atomic.rs` or match a small exact line allowlist.
//!    Pre-existing grandfathered sequences (the CAS store, `faktor-fs`'s
//!    internal stream copy, the git worktree metadata save, the index
//!    generation publish, the startup backup finalize) are allowlisted
//!    **line-by-line** by exact content, so a NEW sequence anywhere is
//!    still listed loudly.
//! 4. **ONE semantic-provider registry authority** (audits 48-54/58/59/83) —
//!    production code constructs `SemanticProviderRegistry::new` ONLY in
//!    the agent crate's fallback constructor and the CLI graph builder;
//!    the native server introspection surface can never construct a
//!    parallel registry (it inspects `deps.semantic` only).
//! 5. **ONE durable evidence authority** — `DurableEvidenceAuthority::for_store`
//!    exists exactly ONCE in production (`crates/agent/src/runtime.rs`,
//!    the runtime whose allocation the daemon graph clones and the native
//!    server receives); no CLI/server site may construct a parallel
//!    authority over the same rows.
//! 6. **Constructor-site authority** — `DurableBudgetLedger::new` and
//!    `ProcessSupervisor::new` additionally have documented production
//!    sites with exact per-file occurrence counts: a new (or stale)
//!    construction site anywhere is a red test, never a review nit.
//!
//! Scanning methodology: per file, comments and string literals are masked
//! out and every `#[cfg(...)]`-gated item that can never compile in a
//! non-test build (`#[cfg(test)]`, `#[cfg(all(test, unix))]`, …) is
//! removed with brace-matched ranges, so markers in tests, docs or
//! examples can never certify production code. Out-of-line
//! `#[cfg(test)] mod` bodies (`tests.rs`, `*_tests.rs`) are skipped by the
//! production scans. The machinery is itself adversarially tested against
//! synthetic sources.

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

    /// Out-of-line `#[cfg(test)] mod` bodies are test-only by declaration
    /// (their parent files carry `#[cfg(test)] mod <name>;`), so production
    /// scans skip them — a marker there can never certify or violate a
    /// production surface.
    fn is_out_of_line_test(rel: &str) -> bool {
        let name = rel.rsplit('/').next().unwrap_or(rel);
        name == "tests.rs" || name.ends_with("_tests.rs")
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

    /// Byte offsets of `marker` on code lines inside kept (production)
    /// ranges (the offset analogue of [`find_markers`], for scans that need
    /// to inspect the enclosing expression).
    fn find_marker_offsets(f: &File<'_>, marker: &str) -> Vec<usize> {
        let mb = marker.as_bytes();
        let mut out = Vec::new();
        let mut pos = 0usize;
        while let Some(rel) = f.src[pos..].find(marker) {
            let at = pos + rel;
            let in_kept = f.kept.iter().any(|(a, z)| at >= *a && at + mb.len() <= *z);
            let in_code = f.code[at..at + mb.len()].iter().all(|c| *c);
            if in_kept && in_code {
                out.push(at);
            }
            pos = at + mb.len();
        }
        out
    }

    /// True when `.await` appears within `window` bytes after `at` — the
    /// shape of the manager's async read wrappers (a synchronous store read
    /// on a Tokio worker has no await in its own expression).
    fn awaited_within(f: &File<'_>, at: usize, window: usize) -> bool {
        let end = (at + window).min(f.src.len());
        f.src[at..end].contains(".await")
    }

    /// The audit-13 tripwire: production `crates/agent` code must never run
    /// a bounded read synchronously. Offenders are the explicitly rejected
    /// shapes — `store().provider_call_prefix_rows`, `store().cost_task_row`,
    /// `store().get_task`, and any `messages_backwards_bounded` call whose
    /// expression is not awaited (the sync `SessionHandle` read) — while the
    /// `SessionManager` async wrappers (`provider_prefix_history`,
    /// `budget_view`, `task`, awaited `messages_backwards_bounded`) pass.
    fn agent_sync_store_read_offenders(f: &File<'_>) -> Vec<String> {
        let mut offenders = Vec::new();
        for marker in [
            "provider_call_prefix_rows",
            "cost_task_row",
            ".store().get_task",
        ] {
            for (line, text) in find_markers(f, &[marker]) {
                offenders.push(format!(
                    "{}:{line}: {text}  [synchronous store read on the turn path; \
                     submit it through the SessionManager's bounded read pool]",
                    f.rel
                ));
            }
        }
        for at in find_marker_offsets(f, "messages_backwards_bounded") {
            if !awaited_within(f, at, 400) {
                offenders.push(format!(
                    "{}:{}: un-awaited messages_backwards_bounded (synchronous \
                     SessionHandle read; use the awaited SessionManager wrapper)",
                    f.rel,
                    line_of(f.src, at)
                ));
            }
        }
        offenders
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

    /// Files whose plain `use std::process::Command` import is the
    /// documented environment authority, never a spawn site.
    const SPAWN_NAME_IMPORT_ALLOWLIST: &[&str] = &["crates/core/src/command.rs"];

    /// `std::process::Command` / `tokio::process::Command` spawn machinery
    /// may exist in exactly two production homes: `crates/terminal` (the
    /// process supervisor, the single owner of children) and `crates/pty`
    /// (the interactive-terminal platform launcher wrapper). One additional
    /// file may NAME `std::process::Command` without constructing or
    /// spawning one: `crates/core/src/command.rs`, the environment authority
    /// whose `EnvSpec::apply` configures a caller-provided `Command` — only
    /// the `use` line is excused, every construction/spawn marker there
    /// still fires. Platform launcher wrappers and tests are the ONLY listed
    /// exceptions — the default allowlist is empty, so any other production
    /// crate that starts spawning is listed loudly and fails the build.
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
                if text.contains("use std::process::Command")
                    && SPAWN_NAME_IMPORT_ALLOWLIST.contains(&rel.as_str())
                {
                    continue;
                }
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

    /// The INDIRECT spawn shape: a production file that LOWERS a
    /// `CommandSpec`/`ResolvedCommand` (or names the resolved type) and
    /// ALSO contains process-`Command` construction/spawn markers. The
    /// direct scan catches the literal markers; this scan additionally
    /// keeps flagging the conversion when a lowered program value flows
    /// into a spawn the marker list would otherwise describe as legit.
    fn indirect_resolved_spawn_offenders(f: &File<'_>) -> Vec<String> {
        let lowered = find_markers(f, &["ResolvedCommand", ".lower(", ".lower_with("]);
        if lowered.is_empty() {
            return Vec::new();
        }
        find_markers(
            f,
            &[
                "std::process::Command",
                "tokio::process::Command",
                "process::Command",
                "Command::new",
                "Command::spawn",
                "CommandExt",
            ],
        )
        .into_iter()
        .filter(|(_, text)| {
            !(text.contains("use std::process::Command")
                && SPAWN_NAME_IMPORT_ALLOWLIST.contains(&f.rel.as_str()))
        })
        .map(|(line, text)| {
            format!(
                "{}:{line}: {text}  [indirect ResolvedCommand -> process spawn]",
                f.rel
            )
        })
        .collect()
    }

    #[test]
    fn no_indirect_resolved_command_spawn_outside_terminal_and_pty() {
        let mut offenders = Vec::new();
        let mut scanned = 0usize;
        for rel in walk_crate_sources() {
            if rel.starts_with("crates/terminal/")
                || rel.starts_with("crates/pty/")
                || is_out_of_line_test(&rel)
            {
                continue;
            }
            let Some(f) = load(&rel) else {
                continue;
            };
            offenders.extend(indirect_resolved_spawn_offenders(&f));
            scanned += 1;
        }
        assert_no_offenders(
            "indirect-spawn scan: a file lowers a ResolvedCommand and then constructs/spawns \
             a process Command (route the resolved program+argv through the ProcessSupervisor)",
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
    // scan 3: hand-rolled temp-write / rename sequences
    // ------------------------------------------------------------------

    /// The forbidden patterns are checked INDEPENDENTLY — no fsync marker
    /// is required, and the historic temp+rename+sync triangle is NOT the
    /// detection shape. Any production `std::fs::rename` / `fs::rename` /
    /// `tokio::fs::rename` / `NamedTempFile::persist`, and any write-family
    /// call whose target names a temp path (`tmp`/`NamedTempFile` on the
    /// same line), is an offender unless its exact line content is
    /// allowlisted. The one sanctioned home is `crates/fs/src/atomic.rs`;
    /// the grandfathered sequences below are documented per exact line, so
    /// any NEW sequence (including inside a grandfathered file) is listed
    /// loudly. Default allowlist: empty.
    const ATOMIC_ANCHOR: &str = "crates/fs/src/atomic.rs";

    const ATOMIC_RENAME_MARKERS: &[&str] = &[
        "std::fs::rename",
        "fs::rename",
        "tokio::fs::rename",
        "NamedTempFile::persist",
    ];

    const ATOMIC_WRITE_MARKERS: &[&str] = &[
        "fs::write",
        "std::fs::write",
        "tokio::fs::write",
        "File::create",
        "OpenOptions",
        "write_all",
    ];

    const ATOMIC_ALLOWLIST: &[(&str, &str)] = &[
        // crates/cas: content-addressed store writer (frozen layer below
        // crates/fs; its own documented temp+fsync+rename durability
        // contract, self-contained and seam-tested).
        (
            "crates/cas/src/lib.rs",
            "let file = fs::File::create(&tmp)?;",
        ),
        (
            "crates/cas/src/lib.rs",
            "let mut f = fs::File::create(&tmp)?;",
        ),
        ("crates/cas/src/lib.rs", "match fs::rename(tmp, path) {"),
        // crates/fs/src/lib.rs: the fs crate's own stream-copy writer
        // (copy_open_file) — same-crate internal helper that reuses the
        // atomic module's temp naming and fsync_parent.
        (
            "crates/fs/src/lib.rs",
            "let mut out = fs::File::create(&tmp)",
        ),
        (
            "crates/fs/src/lib.rs",
            "fs::rename(&tmp, target).map_err(|e| {",
        ),
        // crates/git: worktree metadata save (spec §33) — best-effort
        // .git-internal writer with its own unique-temp discipline.
        ("crates/git/src/lib.rs", "std::fs::rename(&tmp, &path)?;"),
        // crates/index: durable generation publish — the scratch file was
        // written + fsynced by write_scratch; this single rename makes the
        // generation visible, and the torn-publish heal owns a crash
        // between the rename and the row commit.
        (
            "crates/index/src/service.rs",
            "if let Err(e) = fs::rename(&scratch, &gen_path) {",
        ),
        // crates/cli main: startup backup finalize — `store.backup_to`
        // wrote the `.db.tmp-*` snapshot; this rename publishes it and the
        // stale-tmp sweeper owns crash residue.
        (
            "crates/cli/src/main.rs",
            "if let Err(e) = std::fs::rename(&tmp, &dest) {",
        ),
    ];

    /// The temp-write half of scan 3, independent of any fsync: a
    /// write-family call whose LINE also targets a temp path. The temp
    /// check reads the raw line (not the code mask), because the canonical
    /// shape is a string-literal path such as `".thing.tmp"`.
    fn atomic_temp_write_hits(f: &File<'_>) -> Vec<(usize, String)> {
        let mut hits = Vec::new();
        for (line, text) in find_markers(f, ATOMIC_WRITE_MARKERS) {
            let raw = f.src.lines().nth(line - 1).unwrap_or_default();
            if raw.contains("tmp") || raw.contains("NamedTempFile") {
                hits.push((line, text));
            }
        }
        // `NamedTempFile::persist` in its associated/literal form AND the
        // instance form (`line ... NamedTempFile ... .persist(`) are both
        // independent forbidden patterns.
        for (line, text) in find_markers(f, &["NamedTempFile"]) {
            let raw = f.src.lines().nth(line - 1).unwrap_or_default();
            if raw.contains(".persist(") {
                hits.push((line, text));
            }
        }
        hits.sort_unstable();
        hits.dedup();
        hits
    }

    /// Every scan-3 offender of one production file: rename/persist
    /// patterns plus temp-path writes, minus the exact-line allowlist.
    fn atomic_write_offenders(f: &File<'_>) -> Vec<String> {
        let mut hits = find_markers(f, ATOMIC_RENAME_MARKERS);
        hits.extend(atomic_temp_write_hits(f));
        hits.sort_unstable();
        hits.dedup();
        let mut offenders = Vec::new();
        for (line, text) in hits {
            let allowlisted = ATOMIC_ALLOWLIST
                .iter()
                .any(|(p, t)| *p == f.rel && *t == text);
            if !allowlisted {
                offenders.push(format!("{}:{line}: {text}", f.rel));
            }
        }
        offenders
    }

    #[test]
    fn no_hand_rolled_atomic_write_outside_fs_atomic_and_the_exact_allowlist() {
        let mut offenders: Vec<String> = Vec::new();
        let mut scanned = 0usize;
        for rel in walk_crate_sources() {
            if rel == ATOMIC_ANCHOR || is_out_of_line_test(&rel) {
                continue; // the one sanctioned home; out-of-line test bodies
            }
            let Some(f) = load(&rel) else {
                continue;
            };
            offenders.extend(atomic_write_offenders(&f));
            scanned += 1;
        }
        assert_no_offenders(
            "atomic-write scan: a rename / NamedTempFile persist / temp-path write \
             exists outside crates/fs/src/atomic.rs (route file-content replacement \
             through faktor_fs::atomic)",
            &offenders,
            scanned,
            3,
        );
    }

    // ------------------------------------------------------------------
    // scan 4: daemon-authority constructor sites (documented, exact counts)
    // ------------------------------------------------------------------

    /// Each daemon-lifetime constructor has exactly the DOCUMENTED
    /// production sites, with exact per-file occurrence counts:
    ///
    /// - `DurableEvidenceAuthority::for_store`: ONE site — the agent
    ///   runtime, whose allocation the daemon graph clones into
    ///   `DaemonGraph::evidence` and the native server receives. No
    ///   CLI/server site may build a parallel authority over the same rows.
    /// - `DurableBudgetLedger::new`: the CLI graph's ONE ledger, the
    ///   per-task/per-session view constructions in the session crate, the
    ///   orchestrator's two cap-seeding sites, and the server crate's
    ///   `ServerDeps::new` embedded/test seam (the daemon assembles through
    ///   `new_with` over the graph's ledger).
    /// - `SemanticProviderRegistry::new`: the agent fallback constructor
    ///   (embedded/test hosts) and the CLI graph builder; the native
    ///   introspection surface inspects `deps.semantic` only.
    /// - `ProcessSupervisor::new`: the two daemon entries
    ///   (`build_daemon`, `build_daemon_with_mcp_inner`), each handing the
    ///   ONE supervisor into the core builder, plus the terminal crate's own
    ///   `shared()` ephemeral fallback.
    ///
    /// A new (or stale) site anywhere is a red test, never a review nit.
    /// Out-of-line `#[cfg(test)] mod` bodies are skipped (test-only by
    /// declaration).
    const CONSTRUCTOR_SITES: &[(&str, &[(&str, usize)])] = &[
        (
            "DurableEvidenceAuthority::for_store",
            &[("crates/agent/src/runtime.rs", 1)],
        ),
        (
            "DurableBudgetLedger::new",
            &[
                ("crates/cli/src/main.rs", 1),
                ("crates/orchestrator/src/task_executor.rs", 2),
                ("crates/server/src/api.rs", 1),
                ("crates/session/src/manager.rs", 1),
                ("crates/session/src/task.rs", 2),
            ],
        ),
        (
            "SemanticProviderRegistry::new",
            &[
                ("crates/agent/src/lib.rs", 1),
                ("crates/cli/src/graph.rs", 1),
            ],
        ),
        (
            "ProcessSupervisor::new",
            &[
                ("crates/cli/src/main.rs", 2),
                // The terminal crate's own `ProcessSupervisor::shared`
                // fallback (env-var hook registry / crate-level tests): an
                // ephemeral per-process supervisor, documented in place.
                ("crates/terminal/src/lib.rs", 1),
            ],
        ),
    ];

    /// Every production file that constructs `marker`, with the exact
    /// marker hits (line + text).
    fn marker_sites(marker: &str) -> Vec<(String, Vec<(usize, String)>)> {
        let mut out = Vec::new();
        for rel in walk_crate_sources() {
            if is_out_of_line_test(&rel) {
                continue;
            }
            let Some(f) = load(&rel) else {
                continue;
            };
            let hits = find_markers(&f, &[marker]);
            if !hits.is_empty() {
                out.push((rel, hits));
            }
        }
        out
    }

    #[test]
    fn daemon_authority_constructors_have_exactly_the_documented_sites() {
        for (marker, documented) in CONSTRUCTOR_SITES.iter().copied() {
            let seen = marker_sites(marker);
            // Every documented site must exist with EXACTLY the documented
            // count (a stale allowlist entry is a red test).
            for (file, want) in documented.iter().copied() {
                let found = seen
                    .iter()
                    .find(|(rel, _)| rel.as_str() == file)
                    .map(|(_, hits)| hits.len())
                    .unwrap_or(0);
                assert_eq!(
                    found, want,
                    "{marker}: documented production site {file} must occur exactly \
                     {want} time(s), found {found}"
                );
            }
            // No undocumented production site may construct the authority.
            let mut offenders = Vec::new();
            for (rel, hits) in &seen {
                if documented.iter().any(|(file, _)| *file == rel.as_str()) {
                    continue;
                }
                for (line, text) in hits {
                    offenders.push(format!("{rel}:{line}: {text}"));
                }
            }
            assert!(
                offenders.is_empty(),
                "{marker} outside its documented production sites:\n  {}\n",
                offenders.join("\n  ")
            );
        }
        // The evidence authority is the strictest case (exactly one site):
        // assert the marker really is scanned, so the allowlist can never
        // pass vacuously.
        assert_eq!(
            marker_sites("DurableEvidenceAuthority::for_store")
                .iter()
                .map(|(_, hits)| hits.len())
                .sum::<usize>(),
            1,
            "production DurableEvidenceAuthority::for_store occurrences must be exactly 1"
        );
    }

    // ------------------------------------------------------------------
    // scan 5: no synchronous store reads in the production agent runtime
    // ------------------------------------------------------------------

    /// Audit 13: the production `crates/agent` turn path reads through the
    /// `SessionManager`'s bounded async read pool, never synchronously on a
    /// Tokio worker. The explicitly rejected shapes are
    /// `store().provider_call_prefix_rows`, `store().cost_task_row`,
    /// `store().get_task` and a sync (un-awaited)
    /// `messages_backwards_bounded`; the manager async wrappers are the
    /// allowance, and their ADOPTION is asserted too (an empty allowlist
    /// scan would pass vacuously). Test modules, comments and strings are
    /// stripped by the shared machinery.
    #[test]
    fn no_synchronous_store_reads_in_the_production_agent_runtime() {
        let mut offenders = Vec::new();
        let mut scanned = 0usize;
        let mut adopted = 0usize;
        for rel in walk_crate_sources() {
            if !rel.starts_with("crates/agent/") {
                continue;
            }
            let Some(f) = load(&rel) else {
                continue;
            };
            scanned += 1;
            offenders.extend(agent_sync_store_read_offenders(&f));
            // The async wrappers are really adopted by the production agent
            // (history, budget, prefix at minimum): the allowance is
            // demonstrated, not assumed.
            if !find_markers(
                &f,
                &[
                    "messages_backwards_bounded",
                    "budget_view(",
                    "provider_prefix_history(",
                ],
            )
            .is_empty()
            {
                adopted += 1;
            }
        }
        assert_no_offenders(
            "sync-read scan: the production agent runtime must submit every bounded read \
             (history/budget/task/prefix/verification/memory) through the SessionManager's \
             async read pool",
            &offenders,
            scanned,
            5,
        );
        assert!(
            adopted >= 1,
            "the async read wrappers must be adopted by the production agent runtime \
             (otherwise this scan certifies nothing)"
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
    fn atomic_scan_fires_on_the_write_then_rename_regression_fixture() {
        // The audit regression fixture: a temp-path write followed by a
        // rename, with NO fsync call anywhere. The fsync is not the
        // detection signal; both patterns are independently forbidden.
        let fixture = "fn bad(){std::fs::write(\".thing.tmp\",b\"x\"); \
                       std::fs::rename(\".thing.tmp\",\"thing\");}\n";
        let f = synthetic_file("crates/git/src/lib.rs", fixture);
        let offenders = atomic_write_offenders(&f);
        assert!(
            !offenders.is_empty(),
            "the write-then-rename fixture must fail the scan"
        );
        assert!(
            offenders.iter().any(|o| o.contains("std::fs::rename")),
            "the rename must be listed: {offenders:?}"
        );
        // The temp-path write alone fires too (no rename needed).
        let f = synthetic_file(
            "crates/agent/src/lib.rs",
            "fn w() { std::fs::write(\"cache.tmp\", b\"x\").unwrap(); }\n",
        );
        assert!(
            !atomic_write_offenders(&f).is_empty(),
            "a temp-path write with no rename/fsync must be listed"
        );
        // A rename with no temp path is independently forbidden.
        let f = synthetic_file(
            "crates/agent/src/lib.rs",
            "fn m() { std::fs::rename(\"a\", \"b\").unwrap(); }\n",
        );
        assert!(
            !atomic_write_offenders(&f).is_empty(),
            "a rename with no temp path must be listed independently"
        );
        // NamedTempFile::persist (instance form) is independently forbidden.
        let f = synthetic_file(
            "crates/agent/src/lib.rs",
            "fn p() { NamedTempFile::new()?.persist(\"out\").unwrap(); }\n",
        );
        assert!(
            !atomic_write_offenders(&f).is_empty(),
            "NamedTempFile persist must be listed independently"
        );
        // A plain non-temp write is not an atomic-write sequence.
        let f = synthetic_file(
            "crates/agent/src/lib.rs",
            "fn w() { std::fs::write(\"cache.bin\", b\"x\").unwrap(); }\n",
        );
        assert!(atomic_write_offenders(&f).is_empty());
        // The exact allowlist excuses the grandfathered line only.
        let f = synthetic_file("crates/cas/src/lib.rs", "match fs::rename(tmp, path) {\n");
        assert!(
            atomic_write_offenders(&f).is_empty(),
            "the exact allowlisted line must pass"
        );
    }

    #[test]
    fn sync_read_scan_fires_on_sync_reads_and_allows_the_async_wrappers() {
        // A sync SessionHandle-style read (no await) fires.
        let f = synthetic_file(
            "crates/agent/src/runtime.rs",
            "fn t(h: &H) { let _ = h.messages_backwards_bounded(None, 4, 64).unwrap(); }\n",
        );
        let hits = agent_sync_store_read_offenders(&f);
        assert!(
            hits.iter()
                .any(|h| h.contains("messages_backwards_bounded")),
            "an un-awaited window read must be flagged: {hits:?}"
        );
        // The manager async wrapper (awaited) passes.
        let f = synthetic_file(
            "crates/agent/src/runtime.rs",
            "async fn t(m: &M) { let _ = m.messages_backwards_bounded(1, None, 4, 64).await.unwrap(); }\n",
        );
        assert!(
            agent_sync_store_read_offenders(&f).is_empty(),
            "the awaited manager wrapper must pass"
        );
        // Direct sync store reads fire for every rejected shape.
        for (src, needle) in [
            (
                "fn t() { let _ = self.deps.session.store().get_task(s, t); }\n",
                "get_task",
            ),
            (
                "fn t() { let _ = self.deps.session.store().cost_task_row(s, t); }\n",
                "cost_task_row",
            ),
            (
                "fn t() { let _ = self.deps.session.store().provider_call_prefix_rows(s); }\n",
                "provider_call_prefix_rows",
            ),
        ] {
            let f = synthetic_file("crates/agent/src/runtime.rs", src);
            let hits = agent_sync_store_read_offenders(&f);
            assert!(
                hits.iter().any(|h| h.contains(needle)),
                "{needle} must be flagged: {hits:?}"
            );
        }
        // A read buried in a #[cfg(test)] module must NOT fire.
        let f = synthetic_file(
            "crates/agent/src/runtime.rs",
            "#[cfg(test)] mod tests {\n  fn t() { let _ = h.messages_backwards_bounded(None, 1, 1); }\n}\n",
        );
        assert!(agent_sync_store_read_offenders(&f).is_empty());
    }

    #[test]
    fn constructor_marker_scan_ignores_test_gated_mentions() {
        for marker in [
            "DurableEvidenceAuthority::for_store",
            "DurableBudgetLedger::new",
            "SemanticProviderRegistry::new",
            "ProcessSupervisor::new",
        ] {
            let f = synthetic_file(
                "crates/agent/src/runtime.rs",
                &format!("fn production() {{ let _ = {marker}(a, b); }}\n"),
            );
            assert!(
                !find_markers(&f, &[marker]).is_empty(),
                "{marker} must be detected in production text"
            );
            let f = synthetic_file(
                "crates/agent/src/runtime.rs",
                &format!("#[cfg(test)] mod tests {{\n  fn t() {{ let _ = {marker}(a, b); }}\n}}\n"),
            );
            assert!(
                find_markers(&f, &[marker]).is_empty(),
                "{marker} in a test module must never certify or violate production"
            );
        }
    }

    #[test]
    fn indirect_spawn_scan_fires_on_lower_then_spawn_and_passes_the_supervisor_seam() {
        let f = synthetic_file(
            "crates/cli/src/tools.rs",
            "fn run(spec: CommandSpec) -> std::io::Result<()> {\n  let resolved = spec.lower()?;\n  let mut c = std::process::Command::new(resolved.program);\n  c.args(resolved.args);\n  c.spawn()?;\n  Ok(())\n}\n",
        );
        let hits = indirect_resolved_spawn_offenders(&f);
        assert!(
            !hits.is_empty(),
            "lower-then-spawn must be flagged: {hits:?}"
        );
        // Lowering and handing program/argv to the supervisor seam (no
        // process-Command anywhere) is the sanctioned shape.
        let f = synthetic_file(
            "crates/cli/src/tools.rs",
            "fn run(spec: CommandSpec, sup: &ProcessSupervisor) {\n  let resolved = spec.lower()?;\n  let _ = sup.spawn(SpawnConfig { cmd: resolved.program, args: resolved.args });\n}\n",
        );
        assert!(indirect_resolved_spawn_offenders(&f).is_empty());
        // Lowering alone (no spawn marker) never fires.
        let f = synthetic_file(
            "crates/git/src/lib.rs",
            "fn lower(spec: CommandSpec) { let _ = spec.lower(); }\n",
        );
        assert!(indirect_resolved_spawn_offenders(&f).is_empty());
        // The helper detects the shape anywhere; the scan TEST excludes the
        // terminal/pty homes (the supervisor owns its process layer).
        let f = synthetic_file(
            "crates/terminal/src/lib.rs",
            "fn spawn(cmd: ResolvedCommand) { let _ = std::process::Command::new(cmd.program); }\n",
        );
        assert!(!indirect_resolved_spawn_offenders(&f).is_empty());
    }
}
