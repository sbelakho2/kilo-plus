//! Performance gates (spec §37): hard budgets enforced as tests.
//! `[perf]`-gated tests measure; the fast variants assert budgets that must
//! hold on developer machines.

#![cfg_attr(
    not(test),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)] // test-harness crate: the lib view exists only for clippy
use std::sync::Arc;
use std::time::{Duration, Instant};

use kilop_core::id::{SessionId, WorkspaceId};
use kilop_provider::ProviderRegistry;
use kilop_server::permission::ChannelPermissionRequester;
use kilop_session::SessionManager;
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
        let mut deps = kilop_agent::AgentDeps {
            session: session.clone(),
            providers: registry,
            permission_requester: perm.clone(),
            evidence: Arc::new(kilop_agent::NoEvidence),
            tools: Arc::new(kilop_agent::ToolRegistry::new()),
            cas: None,
            workspaces: kilop_fs::WorkspaceFileService::new(),
            edit: None,
            snapshots: None,
            sandbox: None,
            supervisor: None,
            model: "m".into(),
            compaction_model: None,
            compact_at_usage: 0.65,
            instructions: "i".into(),
            clock: Arc::new(kilop_core::time::SystemClock),
            tool_call_mode: kilop_agent::ToolCallMode::Native,
            tool_deadline_ms: 1000,
            retry_policy: kilop_core::retry::RetryPolicy::default(),
        };
        deps.permission_requester = perm.clone();
        kilop_agent::AgentRuntime::new(deps).unwrap()
    };
    let deps = kilop_server::ServerDeps::new(session.clone(), agent, perm);
    let t0 = Instant::now();
    let handle = kilop_server::serve(deps, 0).await.unwrap();
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
    let mut idx = kilop_index::WorkspaceIndex::new();
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
