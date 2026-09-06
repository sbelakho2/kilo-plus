//! Real-model runs: the ACTUAL daemon (subprocess `faktor-cli serve`)
//! against a REAL provider. `#[ignore]`-gated: these runs need provider
//! keys and a built daemon binary, cost money, and are NOT part of the
//! normal test run (`cargo test` skips them).
//!
//! How to run (see README.md in this crate for the full protocol):
//!
//! ```sh
//! cargo build -p faktor-cli                 # or: cargo build --release -p faktor-cli
//! FAKTOR_BENCH_PROVIDER=anthropic \
//! FAKTOR_BENCH_MODEL=<model-id> \
//! FAKTOR_BENCH_API_KEY=<key> \
//! FAKTOR_BENCH_ONLY=rust-reverse-words \
//! cargo test -p faktor-tests-coding-benchmark --test real -- --ignored --nocapture
//! ```
//!
//! Optional: FAKTOR_BENCH_BIN (explicit daemon path), FAKTOR_BENCH_BASE_URL,
//! FAKTOR_BENCH_NETWORK, FAKTOR_BENCH_TASK_TIMEOUT_S (default 600),
//! FAKTOR_BENCH_VERIFY_TIMEOUT_S / FAKTOR_BENCH_VERIFY_OUTPUT_CAP,
//! FAKTOR_BENCH_SUMMARY_PROMPT=0, FAKTOR_BENCH_TAG, FAKTOR_BENCH_ONLY,
//! FAKTOR_BENCH_REPORT_PATH (write the JSON-lines report there).
//!
//! A missing provider/model/key or daemon binary is a DOCUMENTED SKIP
//! (the test passes with the skip note on stderr), never a failure — so
//! `--ignored` stays safe on machines without keys.
//!
//! `real_daemon_mechanics_with_stub_provider` drives the same daemon flow
//! against a LOCAL STUB OpenAI-compatible endpoint (no keys, no money):
//! it proves spawn/auth/session/prompt/poll/abort/readings/verify over the
//! real binary end to end.

use faktor_tests_coding_benchmark::corpus;
use faktor_tests_coding_benchmark::daemon::{run_corpus_real, DaemonBenchConfig};
use faktor_tests_coding_benchmark::Corpus;

#[test]
#[ignore = "[real-model] drives the actual daemon against a real provider (needs FAKTOR_BENCH_PROVIDER/MODEL/API_KEY + a built faktor-cli binary); run explicitly, never in CI"]
fn real_model_corpus_run() {
    let (cfg, note) = match DaemonBenchConfig::from_env() {
        Ok(Some(x)) => x,
        Ok(None) => {
            eprintln!(
                "SKIP: real-model env not configured \
                 (FAKTOR_BENCH_PROVIDER / FAKTOR_BENCH_MODEL / FAKTOR_BENCH_API_KEY); \
                 run with -- --ignored and the documented env vars"
            );
            return;
        }
        Err(e) => panic!("real-model configuration error: {e}"),
    };
    eprintln!("{note}");
    let compile_hint = option_env!("CARGO_BIN_EXE_faktor-cli");
    let corpus: Corpus = corpus::load_corpus(&faktor_tests_coding_benchmark::corpus_dir())
        .expect("checked-in corpus must load");
    let report = run_corpus_real(&cfg, &corpus, compile_hint);

    for line in report.to_json_lines().lines() {
        println!("{line}");
    }
    let agg = serde_json::to_string_pretty(&report.aggregate).unwrap();
    println!("{agg}");
    for note in &report.notes {
        eprintln!("note: {note}");
    }
    if let Ok(path) = std::env::var("FAKTOR_BENCH_REPORT_PATH") {
        let body = format!(
            "{}\n{}\n",
            report.to_json_lines(),
            serde_json::to_string(&report.aggregate).unwrap()
        );
        std::fs::write(&path, body).unwrap_or_else(|e| panic!("write report {path}: {e}"));
        eprintln!("report written to {path}");
    }
    assert_eq!(
        report.results.len(),
        corpus.tasks.len(),
        "every corpus task must produce exactly one row"
    );
}

/// Minimal OpenAI-compatible chat-completions stub used by the mechanics
/// test below: answers every request with one canned text completion (no
/// tool calls), JSON or SSE depending on the request's `stream` flag, and
/// counts the requests it served.
fn spawn_stub_provider() -> (
    std::net::SocketAddr,
    std::sync::Arc<std::sync::atomic::AtomicUsize>,
) {
    use std::io::{BufRead, Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").expect("stub binds");
    let addr = listener.local_addr().expect("stub addr");
    let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let hits2 = hits.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            hits2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut reader = std::io::BufReader::new(stream.try_clone().expect("clone"));
            let mut request = String::new();
            loop {
                let mut line = String::new();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if line == "\r\n" || line == "\n" {
                            break;
                        }
                        request.push_str(&line);
                    }
                }
            }
            let content_length: usize = request
                .lines()
                .find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.trim()
                        .eq_ignore_ascii_case("content-length")
                        .then(|| v.trim().parse().ok())
                        .flatten()
                })
                .unwrap_or(0);
            let mut body = vec![0u8; content_length];
            let _ = reader.read_exact(&mut body);
            let body = String::from_utf8_lossy(&body).to_string();
            let streaming = body.contains("\"stream\":true") || body.contains("\"stream\": true");
            let content = "The fix is in place and the repository test suite passes. \
                 Final summary:\n- crit-01: PASS — suite green\n- crit-02: PASS\n\
                 - crit-03: PASS\n- crit-04: PASS";
            let payload = if streaming {
                let chunk = serde_json::json!({
                    "id": "stub-1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "default",
                    "choices": [{
                        "index": 0,
                        "delta": { "role": "assistant", "content": content },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30 }
                });
                format!(
                    "data: {}\n\ndata: [DONE]\n\n",
                    serde_json::to_string(&chunk).expect("json")
                )
            } else {
                serde_json::to_string(&serde_json::json!({
                    "id": "stub-1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "default",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": content },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30 }
                }))
                .expect("json")
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.len(),
                payload
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    (addr, hits)
}

#[test]
#[ignore = "[real-model mechanics] drives the ACTUAL daemon end to end against a STUB openai-compatible provider (no keys, no money); needs a built faktor-cli binary; run explicitly, never in CI"]
fn real_daemon_mechanics_with_stub_provider() {
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use faktor_tests_coding_benchmark::daemon::{
        resolve_binary, run_task_real_into, write_config_file, DaemonBenchConfig, ProviderKind,
    };
    use faktor_tests_coding_benchmark::verify::VerifyOptions;
    use faktor_tests_coding_benchmark::{corpus, fsutil};

    let Some(bin) = resolve_binary(option_env!("CARGO_BIN_EXE_faktor-cli")) else {
        eprintln!("SKIP: no daemon binary (cargo build -p faktor-cli, or set FAKTOR_BENCH_BIN)");
        return;
    };
    let (addr, hits) = spawn_stub_provider();
    let base_url = format!("http://{addr}/v1");
    let cfg = DaemonBenchConfig {
        provider: ProviderKind::OpenAi,
        model: "default".into(),
        api_key_env: "FAKTOR_BENCH_API_KEY".into(),
        base_url: Some(base_url.clone()),
        network_rows: vec![],
        task_timeout: Duration::from_secs(90),
        verify: VerifyOptions::default(),
        append_criteria: true,
        tag: "stub-mechanics".into(),
    };

    let workspace = tempfile::tempdir().expect("tempdir");
    let copy = workspace.path().join("repo");
    let task =
        corpus::load_task(&faktor_tests_coding_benchmark::corpus_dir().join("rust-reverse-words"))
            .expect("rust task loads");
    fsutil::copy_tree(&task.dir, &copy).expect("workspace copy");

    let data = tempfile::tempdir().expect("tempdir");
    let config_file = write_config_file(data.path(), &cfg).expect("config file");
    let result = run_task_real_into(&cfg, &bin, &task, &copy, data.path(), &config_file)
        .expect("the stub run must complete without a hard failure");
    eprintln!(
        "row: {}",
        serde_json::to_string(&result).expect("row serializes")
    );

    eprintln!("stub served {} model requests", hits.load(Ordering::SeqCst));
    assert!(
        hits.load(Ordering::SeqCst) >= 1,
        "the daemon must have called the stub provider"
    );
    assert!(result.skipped.is_none(), "{:?}", result.skipped);
    assert!(
        !result.verified,
        "the stub never edits files: verify must fail on the pristine copy"
    );
    assert!(result.attempts >= 1, "at least one durable turn recorded");
    assert!(result.wall_ms > 0, "wall time must be measured");
    assert_eq!(result.criteria_total, task.criteria.len());
    // The stub's canned final summary names every criterion key.
    assert_eq!(
        result.criteria_met, result.criteria_total,
        "criteria_met {} vs total {}",
        result.criteria_met, result.criteria_total
    );
    let verify = result.verify.as_ref().expect("verify ran");
    assert_ne!(verify.exit, Some(0), "pristine repo still fails its suite");
    let tail = String::from_utf8_lossy(&verify.tail);
    assert!(
        tail.contains("FAILED") || tail.contains("error"),
        "unexpected verify tail: {tail}"
    );
}
