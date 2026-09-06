//! Normal-mode smoke suite: the FAKE, in-process, no-model harness.
//!
//! Nothing here spawns the daemon and nothing contacts a provider. The
//! tests prove the harness mechanics against the pristine (buggy) corpus:
//! corpus loading + adversarial rejection, immutable corpus copies,
//! toolchain skip-vs-fail gates, `verify.sh` execution (timeout/cap/kill
//! are unit-tested in the `verify` module), deterministic criteria
//! scoring, and the cost aggregation. Pristine repos MUST fail
//! verification; a synthetic pass-task proves the success path of the
//! scorer. All tests run in seconds and are deterministic.

use std::time::Duration;

use faktor_tests_coding_benchmark::{
    corpus, corpus_dir, run_verify_sh, verify::VerifyOptions, Corpus, CorpusError, CriterionResult,
    TaskResult,
};
use tempfile::tempdir;

fn load_checked_in_corpus() -> Corpus {
    corpus::load_corpus(&corpus_dir()).expect("the checked-in corpus must load")
}

fn quick_verify() -> VerifyOptions {
    VerifyOptions {
        timeout: Duration::from_secs(300),
        ..VerifyOptions::default()
    }
}

/// The fake no-model driver: copy the pristine task to a temp workspace,
/// run its own `verify.sh`, score an empty final summary (no model ran),
/// and produce the deterministic row the report aggregates. This is the
/// whole "fake in-process provider mode that does NOT solve repos" path.
fn fake_no_model_run(task: &corpus::Task) -> TaskResult {
    let workspace = tempdir().expect("tempdir");
    let copy = workspace.path().join("repo");
    faktor_tests_coding_benchmark::fsutil::copy_tree(&task.dir, &copy).expect("copy tree");
    let verify = run_verify_sh(&copy, quick_verify());
    let criteria = faktor_tests_coding_benchmark::score::score_summary(&task.criteria, "");
    TaskResult {
        task_id: task.id.clone(),
        lang: task.lang.to_string(),
        skipped: None,
        verified: verify.verified(),
        wall_ms: 0,
        timed_out: false,
        attempts: 0,
        spend_micro: 0,
        tokens_total: 0,
        criteria_met: faktor_tests_coding_benchmark::score::criteria_met(&criteria),
        criteria_total: task.criteria.len(),
        criteria,
        record_status: None,
        verify: Some(verify),
        note: Some("fake no-model harness run (no daemon, no provider)".into()),
    }
}

#[test]
fn checked_in_corpus_is_valid_and_small() {
    let corpus = load_checked_in_corpus();
    assert!(!corpus.tasks.is_empty());
    assert!(
        corpus.tasks.len() <= 12,
        "corpus must stay <= 12 tasks, has {}",
        corpus.tasks.len()
    );
    let langs: Vec<&str> = corpus.tasks.iter().map(|t| t.lang.as_str()).collect();
    for expected in ["rust", "c", "python", "go", "typescript", "java"] {
        assert!(
            langs.contains(&expected),
            "corpus must cover {expected}, has {langs:?}"
        );
    }
    let mut ids = corpus.task_ids();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), corpus.tasks.len(), "task ids must be unique");
    for task in &corpus.tasks {
        assert!(
            !task.task_md.trim().is_empty(),
            "{}: task.md empty",
            task.id
        );
        assert!(
            task.criteria.len() >= 3,
            "{}: criteria.md must carry >= 3 criteria",
            task.id
        );
        assert!(task.file_count >= 2, "{}: repo too thin", task.id);
        assert!(
            task.file_count <= 64,
            "{}: {}-file task exceeds the 64-file cap",
            task.id,
            task.file_count
        );
        let ids: Vec<&str> = task.criteria.iter().map(|c| c.key.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            ids.len(),
            sorted.len(),
            "{}: duplicate criterion keys",
            task.id
        );
    }
}

#[test]
fn pristine_repos_fail_their_own_verification() {
    let corpus = load_checked_in_corpus();
    let mut ran = 0;
    let mut skipped = 0;
    for task in &corpus.tasks {
        match faktor_tests_coding_benchmark::toolchain::require_toolchain(task.lang) {
            Ok(()) => {}
            Err(reason) => {
                // Missing toolchain = documented skip, never a failure.
                skipped += 1;
                eprintln!("skip {}: {reason}", task.id);
                continue;
            }
        }
        let row = fake_no_model_run(task);
        let verify = row.verify.as_ref().expect("verify ran");
        assert!(
            !row.verified,
            "{}: the PRISTINE repo must fail its own suite (bug present)",
            task.id
        );
        assert!(
            !verify.timed_out,
            "{}: verify timed out (environment too slow or verify.sh hangs)",
            task.id
        );
        assert!(
            verify.exit.is_some(),
            "{}: verify must exit with a status, not a signal",
            task.id
        );
        assert_ne!(
            verify.exit,
            Some(0),
            "{}: pristine verify.sh must exit non-zero",
            task.id
        );
        assert_eq!(
            row.criteria_met, 0,
            "{}: no model ran, no criteria met",
            task.id
        );
        ran += 1;
    }
    assert!(
        ran >= 1,
        "at least one task must have run on this machine (toolchains present)"
    );
    eprintln!("ran {ran} pristine tasks, skipped {skipped} for missing toolchains");
}

#[test]
fn scorer_success_path_and_aggregation_are_deterministic() {
    // A synthetic PASSING task (created at test time, never in the
    // checked-in corpus) proves the verified-success branch end to end.
    let dir = tempdir().unwrap();
    let task_dir = dir.path().join("python-synthetic-pass");
    std::fs::create_dir_all(&task_dir).unwrap();
    std::fs::write(task_dir.join("task.md"), "Synthetic: always true.").unwrap();
    std::fs::write(
        task_dir.join("criteria.md"),
        "- crit-01: the trivial check passes\n- crit-02: the summary names this key\n",
    )
    .unwrap();
    std::fs::write(task_dir.join("verify.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    let task = corpus::load_task(&task_dir).expect("synthetic task loads");

    let verify = run_verify_sh(&task_dir, quick_verify());
    assert!(verify.verified());

    // Model summary naming the keys → criteria met.
    let summary = "Finished. crit-01: PASS — trivial. crit-02: PASS — named here.";
    let scored = faktor_tests_coding_benchmark::score::score_summary(&task.criteria, summary);
    assert_eq!(
        faktor_tests_coding_benchmark::score::criteria_met(&scored),
        2
    );

    let pass_row = TaskResult {
        task_id: task.id.clone(),
        lang: "python".into(),
        skipped: None,
        verified: verify.verified(),
        wall_ms: 5,
        timed_out: false,
        attempts: 2,
        spend_micro: 250,
        tokens_total: 900,
        criteria: scored.clone(),
        criteria_met: faktor_tests_coding_benchmark::score::criteria_met(&scored),
        criteria_total: task.criteria.len(),
        record_status: None,
        verify: Some(verify),
        note: None,
    };
    // The other row: a no-model run of a REAL buggy corpus task — the
    // pristine repo still fails its suite, so the aggregate sees one
    // verified and one unverified task.
    let corpus = load_checked_in_corpus();
    let fail_row = fake_no_model_run(corpus.task("rust-reverse-words").expect("rust task"));
    assert!(
        !fail_row.verified,
        "no-model run never fixes the planted bug"
    );

    let report = faktor_tests_coding_benchmark::report::CorpusReport::new(vec![
        pass_row.clone(),
        fail_row.clone(),
    ]);
    let a = &report.aggregate;
    assert_eq!(a.tasks_ran, 2);
    assert_eq!(a.tasks_skipped, 0);
    assert_eq!(a.verified, 1);
    assert_eq!(a.spend_micro, 250);
    assert_eq!(a.cost_to_verified_micro, Some(250));
    assert_eq!(a.criteria_met, 2);
    assert!(report.to_json_lines().lines().count() == 2);

    // Determinism: identical inputs → identical rows.
    let again =
        faktor_tests_coding_benchmark::report::CorpusReport::new(vec![pass_row, fail_row.clone()]);
    let a_json = serde_json::to_string(&report.aggregate).unwrap();
    let again_json = serde_json::to_string(&again.aggregate).unwrap();
    assert_eq!(a_json, again_json, "aggregation must be deterministic");
}

#[test]
fn corpus_bytes_are_immutable_across_harness_runs() {
    let before = faktor_tests_coding_benchmark::fsutil::snapshot_tree(&corpus_dir()).unwrap();
    let corpus = load_checked_in_corpus();
    let rust_task = corpus.task("rust-reverse-words").expect("rust task");
    let row = fake_no_model_run(rust_task);
    assert!(!row.verified);
    let after = faktor_tests_coding_benchmark::fsutil::snapshot_tree(&corpus_dir()).unwrap();
    assert_eq!(
        before, after,
        "a harness run must never write the checked-in corpus"
    );
}

#[test]
fn hostile_corpus_entries_are_rejected() {
    // Invalid task ids / missing metadata / oversized metadata.
    let root = tempdir().unwrap();
    std::fs::create_dir_all(root.path().join("bad-id!")).unwrap();
    let err = corpus::load_corpus(root.path()).unwrap_err();
    assert!(matches!(err, CorpusError::InvalidTaskId(_)), "{err}");

    let root = tempdir().unwrap();
    let task = root.path().join("rust-x");
    std::fs::create_dir_all(&task).unwrap();
    std::fs::write(task.join("task.md"), "task").unwrap();
    // criteria.md and verify.sh missing.
    let err = corpus::load_corpus(root.path()).unwrap_err();
    assert!(matches!(err, CorpusError::MissingFile { .. }), "{err}");

    // Oversized metadata is rejected before any harness work.
    let root = tempdir().unwrap();
    let task = root.path().join("rust-x");
    std::fs::create_dir_all(&task).unwrap();
    std::fs::write(task.join("task.md"), vec![b'a'; 70 * 1024]).unwrap();
    std::fs::write(task.join("criteria.md"), "- crit-01: a\n").unwrap();
    std::fs::write(task.join("verify.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    let err = corpus::load_corpus(root.path()).unwrap_err();
    assert!(matches!(err, CorpusError::Oversized { .. }), "{err}");

    // Hostile criteria content.
    let root = tempdir().unwrap();
    let task = root.path().join("python-x");
    std::fs::create_dir_all(&task).unwrap();
    std::fs::write(task.join("task.md"), "task").unwrap();
    std::fs::write(task.join("criteria.md"), "crit-01: missing the dash\n").unwrap();
    std::fs::write(task.join("verify.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    let err = corpus::load_corpus(root.path()).unwrap_err();
    assert!(matches!(err, CorpusError::Criteria(_)), "{err}");

    // A corpus whose language prefix is unknown is rejected.
    let root = tempdir().unwrap();
    let task = root.path().join("car-bench");
    std::fs::create_dir_all(&task).unwrap();
    std::fs::write(task.join("task.md"), "task").unwrap();
    std::fs::write(task.join("criteria.md"), "- crit-01: a\n").unwrap();
    std::fs::write(task.join("verify.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    assert!(corpus::load_corpus(root.path()).is_err());
}

#[cfg(unix)]
#[test]
fn hostile_symlink_corpus_entries_are_rejected() {
    // A symlink INSIDE a task tree could escape the workspace copy.
    let outside = tempdir().unwrap();
    std::fs::write(outside.path().join("secret.txt"), "x").unwrap();
    let root = tempdir().unwrap();
    let task = root.path().join("rust-x");
    std::fs::create_dir_all(&task).unwrap();
    std::fs::write(task.join("task.md"), "task").unwrap();
    std::fs::write(task.join("criteria.md"), "- crit-01: a\n").unwrap();
    std::fs::write(task.join("verify.sh"), "#!/bin/sh\nexit 0\n").unwrap();
    std::os::unix::fs::symlink(outside.path().join("secret.txt"), task.join("link.txt")).unwrap();
    let err = corpus::load_corpus(root.path()).unwrap_err();
    assert!(matches!(err, CorpusError::Symlink { .. }), "{err}");
}

#[test]
fn real_mode_planning_without_env_is_a_documented_skip() {
    // No real-model config → every task is a documented skip; the plan is
    // deterministic and never touches the daemon.
    let corpus = load_checked_in_corpus();
    let plan = faktor_tests_coding_benchmark::daemon::plan_corpus(None, &corpus, None);
    assert_eq!(plan.len(), corpus.tasks.len());
    for (id, gate) in plan {
        let reason = gate.expect_err(&format!("{id} must skip without real-model env"));
        assert!(
            reason.contains("FAKTOR_BENCH"),
            "{id}: unexpected skip reason {reason:?}"
        );
    }
}

#[test]
fn daemon_config_json_is_strict_and_pinned() {
    // Pure config-file construction (no daemon, no env): the pinned
    // provider/model and the sandbox override must serialize into exactly
    // the daemon's strict JSON surface.
    use faktor_tests_coding_benchmark::daemon::{DaemonBenchConfig, ProviderKind};
    let dir = tempdir().unwrap();
    let cfg = DaemonBenchConfig {
        provider: ProviderKind::OpenAi,
        model: "gpt-x".into(),
        api_key_env: "FAKTOR_BENCH_API_KEY".into(),
        base_url: Some("http://127.0.0.1:8080/v1".into()),
        network_rows: vec!["http://127.0.0.1:8080".into()],
        task_timeout: Duration::from_secs(60),
        verify: quick_verify(),
        append_criteria: true,
        tag: "unit".into(),
    };
    let path = faktor_tests_coding_benchmark::daemon::write_config_file(dir.path(), &cfg).unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["config_version"], 1);
    assert_eq!(v["model"], "gpt-x");
    assert_eq!(v["routing_mode"]["pinned"]["provider"], "bench");
    assert_eq!(v["routing_mode"]["pinned"]["model"], "gpt-x");
    assert_eq!(v["providers"][0]["kind"], "open_ai");
    assert_eq!(v["providers"][0]["base_url"], "http://127.0.0.1:8080/v1");
    assert_eq!(v["sandbox"]["network"][0], "http://127.0.0.1:8080");
    assert!(v["tasks"]["shadow_mutation"] == serde_json::Value::Bool(false));
}

#[test]
fn criterion_results_render_deterministically() {
    // The serialized row of a scored criterion is stable.
    let r = CriterionResult {
        key: "crit-01".into(),
        summary_hit: true,
        record_pass: false,
    };
    let json = serde_json::to_string(&r).unwrap();
    let back: CriterionResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back, r);
}
