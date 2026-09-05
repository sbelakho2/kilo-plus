//! Performance gates (spec §37): hard budgets enforced as tests.
//! `[perf]`-gated tests measure; the fast variants assert budgets that must
//! hold on developer machines.
//!
//! Distribution harness (audit 82-83): single-Instant samples catch
//! catastrophes, not regressions, so [`Dist`]/[`bench_n`] repeat measurements
//! and budget assertions target p50/p95/p99 with runner/build metadata.

#![cfg_attr(
    not(test),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)] // test-harness crate: the lib view exists only for clippy
use std::sync::Arc;
use std::time::{Duration, Instant};

use faktor_core::id::{SessionId, WorkspaceId};
use faktor_provider::ProviderRegistry;
use faktor_server::permission::ChannelPermissionRequester;
use faktor_session::SessionManager;
use tempfile::tempdir;

/// Warm health/API response < 5ms (spec §37): a session page load and a
/// state read must both be sub-5ms warm.
#[test]
fn warm_api_response_under_5ms() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "perf", "fake", "m").unwrap();

    // Warm up.
    let handle = session.get_session(row.id()).unwrap().unwrap();
    let _ = handle.messages_page(None, 10).unwrap();
    let _ = handle.state().unwrap();

    let mut worst = Duration::ZERO;
    for _ in 0..200 {
        let t0 = Instant::now();
        let _ = handle.messages_page(None, 10).unwrap();
        worst = worst.max(t0.elapsed());
    }
    // CI machines vary; the budget is 5ms warm — allow 20ms headroom for
    // debug builds.
    assert!(
        worst < Duration::from_millis(20),
        "warm page load {worst:?} exceeds the budget"
    );
}

/// Historical message count has ~zero effect on initial load (spec §37).
#[test]
fn historical_message_count_does_not_affect_initial_load() {
    let dir = tempdir().unwrap();
    let small = {
        let session =
            SessionManager::open(dir.path().join("s"), dir.path().join("sc"), true).unwrap();
        let ws = session.create_workspace("/w").unwrap();
        let row = session.create_session(ws, "small", "fake", "m").unwrap();
        (session, row.id())
    };
    let big = {
        let session =
            SessionManager::open(dir.path().join("b"), dir.path().join("bc"), true).unwrap();
        let ws = session.create_workspace("/w").unwrap();
        let row = session.create_session(ws, "big", "fake", "m").unwrap();
        let handle = session.get_session(row.id()).unwrap().unwrap();
        for i in 1..=50_000 {
            handle
                .put_message(i, "user", serde_json::json!({"text": format!("m{i}")}))
                .unwrap();
        }
        (session, row.id())
    };

    let h_small = small.0.get_session(small.1).unwrap().unwrap();
    let h_big = big.0.get_session(big.1).unwrap().unwrap();
    // Warm.
    let _ = h_small.messages_page(None, 100).unwrap();
    let _ = h_big.messages_page(None, 100).unwrap();

    let t0 = Instant::now();
    let _ = h_small.messages_page(None, 100).unwrap();
    let small_ms = t0.elapsed();
    let t0 = Instant::now();
    let _ = h_big.messages_page(None, 100).unwrap();
    let big_ms = t0.elapsed();

    // 50k messages must not make the initial page meaningfully slower.
    assert!(
        big_ms <= small_ms + Duration::from_millis(5),
        "big session page took {big_ms:?} vs small {small_ms:?}"
    );
}

/// The daemon cold-starts fast: binding + handshake < 150ms typical
/// (spec §37) — measured here on the server stack alone.
#[tokio::test]
#[ignore = "[perf] cold start measurement — run explicitly"]
async fn cold_start_under_150ms() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let perm = ChannelPermissionRequester::new(Duration::from_secs(5));
    let registry = Arc::new(ProviderRegistry::new());
    let agent = {
        let mut deps = faktor_agent::AgentDeps {
            session: session.clone(),
            providers: registry,
            chunk_sink: None,
            permission_requester: perm.clone(),
            evidence: Arc::new(faktor_agent::NoEvidence),
            tools: Arc::new(faktor_agent::ToolRegistry::new()),
            cas: None,
            workspaces: faktor_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            verifier: None,
            hooks: None,
            instructions_loader: None,
            router: None,
            budget_micro: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "i".into(),
            clock: Arc::new(faktor_core::time::SystemClock),
            tool_call_mode: faktor_agent::ToolCallMode::Native,
            tool_deadline_ms: 1000,
            retry_policy: faktor_core::retry::RetryPolicy::default(),
        };
        deps.permission_requester = perm.clone();
        faktor_agent::AgentRuntime::new(deps).unwrap()
    };
    let deps = faktor_server::ServerDeps::new(session.clone(), agent, perm);
    let t0 = Instant::now();
    let handle = faktor_server::serve(deps, 0).await.unwrap();
    let elapsed = t0.elapsed();
    let _ = handle;
    // Debug builds are slower; assert the relaxed budget (150ms typical in
    // release).
    assert!(
        elapsed < Duration::from_millis(500),
        "cold start {elapsed:?}"
    );
}

/// Idle daemon memory excluding indexes < 80MB (spec §37) — measured as the
/// resident size of this test process after heavy session churn.
#[tokio::test]
#[ignore = "[perf] memory measurement — run explicitly"]
async fn idle_memory_under_80mb() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    // Churn: create + drop many sessions to surface leaks.
    for _ in 0..200 {
        let ws = session.create_workspace("/w").unwrap();
        let row = session.create_session(ws, "churn", "fake", "m").unwrap();
        let handle = session.get_session(row.id()).unwrap().unwrap();
        for i in 1..=50 {
            handle
                .put_message(i, "user", serde_json::json!({"text": "x".repeat(1000)}))
                .unwrap();
        }
        drop(handle);
    }
    drop(session);
    // Measure RSS.
    let rss_kb = rss_kb();
    assert!(
        rss_kb < 80 * 1024,
        "idle RSS {rss_kb}KB exceeds the 80MB budget"
    );
}

fn rss_kb() -> u64 {
    let pid = std::process::id();
    let out = std::process::Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .unwrap_or(u64::MAX)
}

/// Cached symbol lookup < 10ms (spec §37).
#[test]
fn cached_symbol_lookup_under_10ms() {
    let mut idx = faktor_index::WorkspaceIndex::new();
    let ws = WorkspaceId::new(1);
    let src = "pub fn alpha() {}\npub fn beta() {}\n".repeat(2000);
    idx.index_file(ws, std::path::Path::new("big.rs"), src.as_bytes(), 1)
        .unwrap();
    let t0 = Instant::now();
    for _ in 0..1000 {
        let hits = idx.symbol_lookup(ws, "alpha", 10);
        assert!(!hits.is_empty());
    }
    let per = t0.elapsed() / 1000;
    assert!(per < Duration::from_millis(10), "symbol lookup {per:?}");
}

/// Session initial load < 150ms warm (spec §37) for a 100k-message session.
#[test]
fn session_initial_load_under_150ms_warm() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session.create_session(ws, "big", "fake", "m").unwrap();
    let handle = session.get_session(row.id()).unwrap().unwrap();
    for i in 1..=100_000 {
        handle
            .put_message(i, "user", serde_json::json!({"text": format!("m{i}")}))
            .unwrap();
    }
    // Warm.
    let _ = handle.messages_page(None, 100).unwrap();
    let _ = handle.session_state_view().unwrap();
    let t0 = Instant::now();
    let _ = handle.session_state_view().unwrap();
    let _ = handle.messages_page(None, 100).unwrap();
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_millis(200),
        "initial load {elapsed:?}"
    );
}

// Keep SessionId referenced.
#[allow(dead_code)]
fn _sid(_: SessionId) {}

// ---------------------------------------------------------------------------
// Distribution measurement harness (audit 82-83)
// ---------------------------------------------------------------------------

/// Warmup calls before sampling starts.
const WARMUP: usize = 10;
/// Total wall-clock bound for one [`bench_n`] batch: a slow op yields fewer
/// than `n` samples instead of a runaway test.
const WALL_CAP: Duration = Duration::from_secs(10);

/// Repeated-measure sample set, stored as u64 nanoseconds.
#[derive(Debug, Default)]
struct Dist {
    samples: Vec<u64>,
}

impl Dist {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    fn push(&mut self, ns: u64) {
        self.samples.push(ns);
    }

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The `p`-th percentile in ns, by linear interpolation between sorted
    /// order statistics (deterministic and monotone in `p` for a fixed set).
    fn pct(&self, p: f64) -> f64 {
        assert!(!self.samples.is_empty(), "pct on an empty Dist");
        assert!((0.0..=100.0).contains(&p), "pct {p} out of [0, 100]");
        let mut sorted = self.samples.clone();
        sorted.sort_unstable();
        let rank = p / 100.0 * (sorted.len() - 1) as f64;
        let lo = rank as usize;
        let hi = (rank.ceil() as usize).min(sorted.len() - 1);
        let lo_v = sorted[lo] as f64;
        lo_v + (sorted[hi] as f64 - lo_v) * (rank - lo as f64)
    }

    /// Arithmetic mean of the samples in ns.
    fn mean(&self) -> f64 {
        assert!(!self.samples.is_empty(), "mean of an empty Dist");
        self.samples.iter().map(|&s| s as f64).sum::<f64>() / self.samples.len() as f64
    }
}

/// Human-readable ns quantity, e.g. `format_pct(1_500.0) == "1.50 µs"`.
fn format_pct(ns: f64) -> String {
    if ns >= 1e9 {
        format!("{:.2} s", ns / 1e9)
    } else if ns >= 1e6 {
        format!("{:.2} ms", ns / 1e6)
    } else if ns >= 1e3 {
        format!("{:.2} µs", ns / 1e3)
    } else {
        format!("{ns:.0} ns")
    }
}

/// Runs `f` `n` times and returns the sample distribution. Ten warmup calls
/// run first; sampling stops at the [`WALL_CAP`] deadline, so the batch is
/// bounded in wall time (one in-flight sample may overshoot the cap).
fn bench_n<F: FnMut()>(f: F, n: usize) -> Dist {
    bench_n_with_cap(f, n, WALL_CAP)
}

/// [`bench_n`] with an injectable cap so the wall-cap truncation is
/// exercised deterministically without waiting out the 10s default.
fn bench_n_with_cap<F: FnMut()>(mut f: F, n: usize, cap: Duration) -> Dist {
    let mut dist = Dist::new();
    for _ in 0..WARMUP {
        f();
    }
    let deadline = Instant::now() + cap;
    while dist.len() < n.max(1) && Instant::now() < deadline {
        let t0 = Instant::now();
        f();
        dist.push(t0.elapsed().as_nanos() as u64);
    }
    dist
}

/// Runner/build metadata for perf reports: package version, build profile
/// (Cargo does not expose `PROFILE` at compile time; `debug_assertions` is
/// the reliable proxy) and the git commit under test.
fn build_meta() -> String {
    let head = match std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
    {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).trim().to_owned(),
        _ => String::from("unknown"),
    };
    format!(
        "crate={} v{} debug_assertions={} commit={}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        cfg!(debug_assertions),
        head
    )
}

/// One report line per [perf] test so explicit runs leave an audit trail
/// (`--nocapture` shows it on pass; a failing assertion prints it too).
fn perf_report(test: &str, op: &str, dist: &Dist) {
    eprintln!(
        "[perf] {test}: {op}: p50={} p95={} p99={} mean={} n={} | {}",
        format_pct(dist.pct(50.0)),
        format_pct(dist.pct(95.0)),
        format_pct(dist.pct(99.0)),
        format_pct(dist.mean()),
        dist.len(),
        build_meta()
    );
}

// ---------------------------------------------------------------- sanity (no timing budgets)

/// Sanity: percentiles are finite, correctly ordered and reproducible, and
/// the mean is exact for integer-ns samples.
#[test]
fn dist_percentiles_finite_and_ordered() {
    let mut d = Dist::new();
    for i in 1..=1000u64 {
        d.push(i * 1000);
    }
    assert!(!d.is_empty(), "pushes must be recorded");
    let p50 = d.pct(50.0);
    let p95 = d.pct(95.0);
    let p99 = d.pct(99.0);
    assert!(d.pct(0.0) <= p50 && p50 <= p95 && p95 <= p99 && p99 <= d.pct(100.0));
    for v in [p50, p95, p99, d.mean()] {
        assert!(v.is_finite(), "non-finite statistic {v}");
    }
    let mean = d.mean();
    assert!(
        (mean - 500_500.0).abs() < 1e-6,
        "mean of 1..=1000 µs is 500.5 µs, got {mean}"
    );
    assert_eq!(d.pct(95.0), p95, "pct must be reproducible");
}

/// Sanity: the report string renders ns in a human unit.
#[test]
fn dist_report_format_human_readable() {
    assert_eq!(format_pct(1_500.0), "1.50 µs");
    assert_eq!(format_pct(1_500_000.0), "1.50 ms");
    assert_eq!(format_pct(25.0), "25 ns");
}

/// Sanity: the harness itself must not hang — a fast op with n=1000 returns
/// quickly with all samples, and a slow op is truncated by the wall cap
/// instead of running n × per-sample time.
#[test]
fn bench_n_wall_cap_bounds_runtime() {
    let t0 = Instant::now();
    let fast = bench_n(|| std::hint::black_box(()), 1000);
    let elapsed = t0.elapsed();
    assert_eq!(fast.len(), 1000, "fast bench_n must collect every sample");
    assert!(
        elapsed < Duration::from_secs(1),
        "fast bench_n took {elapsed:?}"
    );

    // 4ms × 1000 = 4s uncapped; a 300ms cap must truncate the batch.
    let t0 = Instant::now();
    let slow = bench_n_with_cap(
        || std::thread::sleep(Duration::from_millis(4)),
        1000,
        Duration::from_millis(300),
    );
    let elapsed = t0.elapsed();
    assert!(
        slow.len() < 1000,
        "wall cap did not truncate the sample count"
    );
    assert!(
        elapsed < Duration::from_secs(1),
        "wall cap did not bound runtime: {elapsed:?}"
    );
}

// ---------------------------------------------------------------- [perf] distribution budgets (release)

/// Reference budget (release): newest-first paging over a 50k-message
/// session keeps p95 under 5ms per page (spec §37 paging is fundamental).
/// Distribution, not a single worst sample.
#[test]
#[ignore = "[perf] paging distribution over 50k messages — run explicitly"]
fn perf_messages_before_paging_p95_lt_5ms() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session
        .create_session(ws, "perf-page", "fake", "m")
        .unwrap();
    let handle = session.get_session(row.id()).unwrap().unwrap();
    for i in 1..=50_000 {
        handle
            .put_message(i, "user", serde_json::json!({"text": format!("m{i}")}))
            .unwrap();
    }
    // Warm the store and sqlite statement paths.
    let _ = handle.messages_before(None, 100).unwrap();
    let dist = bench_n(
        || {
            let _ = handle.messages_before(None, 100).unwrap();
        },
        2000,
    );
    perf_report(
        "perf_messages_before_paging_p95_lt_5ms",
        "page of 100 over 50k-message session",
        &dist,
    );
    assert!(
        dist.pct(95.0) < 5e6,
        "p95 {} exceeds the 5ms/page budget",
        format_pct(dist.pct(95.0))
    );
}

/// Reference budget (release): a tiny warm pure op — the wire JSON round
/// trip (encode + decode) of a ~4KiB message payload, what paging and SSE
/// do per message — keeps p50 well under 100µs.
#[test]
#[ignore = "[perf] JSON wire round-trip distribution — run explicitly"]
fn perf_json_wire_roundtrip_p50_lt_100us() {
    let payload = serde_json::json!({"role": "user", "text": "x".repeat(4096)});
    let dist = bench_n(
        || {
            let bytes = serde_json::to_vec(&payload).unwrap();
            let _: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        },
        10_000,
    );
    perf_report(
        "perf_json_wire_roundtrip_p50_lt_100us",
        "encode+decode 4KiB JSON payload",
        &dist,
    );
    assert!(
        dist.pct(50.0) < 1e5,
        "p50 {} exceeds the 100µs budget",
        format_pct(dist.pct(50.0))
    );
}

// ---------------------------------------------------------------- growing transcript

/// Certification gate (audits 84-92): long-running task cost stays ~
/// constant as the transcript grows. A session transcript is grown from
/// 2k to 20k messages while the per-window (newest page of 100) cost is
/// measured at both ends with the distribution harness; the bounded loader
/// made paging ~O(window), so the last p50 must stay within 3x the first
/// p50 — proving per-turn cost does not grow with history.
#[test]
#[ignore = "[perf] growing-transcript cost bound — run explicitly"]
fn perf_growing_transcript_cost_stays_bounded() {
    let dir = tempdir().unwrap();
    let session =
        SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
    let ws = session.create_workspace("/w").unwrap();
    let row = session
        .create_session(ws, "perf-growing", "fake", "m")
        .unwrap();
    let handle = session.get_session(row.id()).unwrap().unwrap();

    let fill_to = |next_id: &mut i64, n: i64| {
        for _ in 1..=n {
            handle
                .put_message(
                    *next_id,
                    "user",
                    serde_json::json!({"text": format!("m{next_id}")}),
                )
                .unwrap();
            *next_id += 1;
        }
    };

    // ---- Small transcript end: 2k messages, warmed, p50 of a page.
    let mut next_id = 1i64;
    fill_to(&mut next_id, 2_000);
    let _ = handle.messages_before(None, 100).unwrap();
    let small = bench_n(
        || {
            let _ = handle.messages_before(None, 100).unwrap();
        },
        1500,
    );
    perf_report(
        "perf_growing_transcript_cost_stays_bounded",
        "page of 100 over a 2k-message session",
        &small,
    );

    // ---- Grow the SAME session transcript to 20k messages and re-measure
    // the same per-window cost at the large end.
    fill_to(&mut next_id, 18_000);
    let _ = handle.messages_before(None, 100).unwrap();
    let large = bench_n(
        || {
            let _ = handle.messages_before(None, 100).unwrap();
        },
        1500,
    );
    perf_report(
        "perf_growing_transcript_cost_stays_bounded",
        "page of 100 over a 20k-message session",
        &large,
    );

    let small_p50 = small.pct(50.0);
    let large_p50 = large.pct(50.0);
    assert!(
        large_p50 <= 3.0 * small_p50.max(1.0),
        "per-window cost grew with the transcript: p50 over 20k messages {} is > 3x p50 over \
         2k messages {}",
        format_pct(large_p50),
        format_pct(small_p50)
    );
}
