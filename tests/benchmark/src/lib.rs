//! faktor-tests-benchmark — the frozen benchmarking corpus + harness substrate
//! (audit items 26-27).
//!
//! Freezes the 11-task mixed corpus (`corpus.json`, embedded at compile time),
//! the deterministic three-policy cost + verification harness *metrics*, and
//! the offline report substrate so future live runs and same-model comparison
//! lanes have a stable base. The harness is deterministic and offline: live
//! runs are env-gated behind `FAKTOR_BENCH_LIVE=1` (see [`live_mode`] and the
//! `#[ignore]`-gated `live_benchmark_requires_env` skeleton test). Router
//! *policy* (AlwaysFrontier / Economy / MaxQuality routing) deliberately lives
//! in the economy crate, not here; this crate owns only corpus + metrics +
//! report + gate.
//!
//! Deps are minimal on purpose: serde/serde_json and `faktor-core` (used for
//! the kernel identity stamped into reports, so a recorded lane is always
//! attributable to the frozen runtime it ran on). The loader is
//! adversarial-first: unknown JSON fields are rejected at parse time and
//! structural violations (wrong task count, duplicate ids, empty prompts,
//! oversized prompts, empty or oversized criteria lists) are rejected by
//! [`TaskCorpus::validate`], which collects every violation instead of
//! stopping at the first one.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use faktor_core::{PROTOCOL_V756, UX_BASELINE, VERSION};
use serde::{Deserialize, Serialize};

/// The frozen corpus size. A task may be added only by changing this constant
/// *and* the audit mirror in the tests; anything else fails loudly.
pub const CORPUS_TASK_COUNT: usize = 11;

/// Hard bound on prompt length, measured in characters (not bytes): a hostile
/// multi-byte payload may not smuggle a longer prompt past the bound.
pub const MAX_PROMPT_CHARS: usize = 4000;

/// Hard bound on the number of acceptance criteria per task.
pub const MAX_CRITERIA_ITEMS: usize = 8;

/// Environment variable that flips the harness into live mode.
pub const LIVE_ENV: &str = "FAKTOR_BENCH_LIVE";

const CORPUS_JSON: &str = include_str!("corpus.json");

/// What must pass for a task result to count as verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub enum ExpectedVerification {
    #[serde(rename = "build+test")]
    BuildAndTest,
    #[serde(rename = "build")]
    Build,
    #[serde(rename = "none")]
    None,
}

impl ExpectedVerification {
    pub const fn as_str(self) -> &'static str {
        match self {
            ExpectedVerification::BuildAndTest => "build+test",
            ExpectedVerification::Build => "build",
            ExpectedVerification::None => "none",
        }
    }
}

impl fmt::Display for ExpectedVerification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for ExpectedVerification {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "build+test" => Ok(ExpectedVerification::BuildAndTest),
            "build" => Ok(ExpectedVerification::Build),
            "none" => Ok(ExpectedVerification::None),
            other => Err(format!("unknown expected_verification `{other}`")),
        }
    }
}

/// One frozen benchmark task. `deny_unknown_fields` makes hostile JSON (an
/// unexpected extra field, a typo'd key) a hard parse error rather than
/// silently parsed-and-dropped input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskDescriptor {
    pub id: String,
    pub category: String,
    pub prompt: String,
    pub acceptance_criteria: Vec<String>,
    pub changed_files_hint: Vec<String>,
    pub expected_verification: ExpectedVerification,
    pub deceptive: bool,
}

impl TaskDescriptor {
    pub fn prompt_len(&self) -> usize {
        self.prompt.chars().count()
    }

    pub fn is_deceptive(&self) -> bool {
        self.deceptive
    }
}

/// The frozen corpus: the embedded `corpus.json` parsed and validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCorpus {
    pub tasks: Vec<TaskDescriptor>,
}

impl TaskCorpus {
    /// Parse and validate. Parse failures (including unknown fields) and
    /// structural violations both surface here; use [`CorpusError::into_json`]
    /// or pattern-matching to tell them apart.
    pub fn load_json(json: &str) -> Result<Self, CorpusError> {
        let corpus: TaskCorpus =
            serde_json::from_str(json).map_err(|e| CorpusError::Json(e.to_string()))?;
        corpus.validate()?;
        Ok(corpus)
    }

    /// Load the embedded frozen corpus.
    pub fn load_embedded() -> Result<Self, CorpusError> {
        Self::load_json(CORPUS_JSON)
    }

    /// Structural validation, collecting **every** violation in one pass so a
    /// multi-way-corrupted corpus is rejected with the full problem list.
    pub fn validate(&self) -> Result<(), CorpusError> {
        let mut problems: Vec<String> = Vec::new();
        let mut seen: HashSet<&str> = HashSet::with_capacity(self.tasks.len());
        let mut duplicates: Vec<String> = Vec::new();

        if self.tasks.len() != CORPUS_TASK_COUNT {
            problems.push(format!(
                "corpus must contain exactly {CORPUS_TASK_COUNT} tasks, found {}",
                self.tasks.len()
            ));
        }

        for (index, task) in self.tasks.iter().enumerate() {
            if task.id.is_empty() {
                problems.push(format!("task at index {index} has an empty id"));
            } else if !seen.insert(task.id.as_str()) {
                duplicates.push(task.id.clone());
            }

            if task.category.is_empty() {
                problems.push(format!("task `{}` has an empty category", task.id));
            }

            let prompt_len = task.prompt_len();
            if prompt_len == 0 {
                problems.push(format!("task `{}` has an empty prompt", task.id));
            } else if prompt_len > MAX_PROMPT_CHARS {
                problems.push(format!(
                    "task `{}` prompt is {prompt_len} chars, exceeds the bound of {MAX_PROMPT_CHARS}",
                    task.id
                ));
            }

            if task.acceptance_criteria.is_empty() {
                problems.push(format!("task `{}` has no acceptance criteria", task.id));
            }
            if task.acceptance_criteria.len() > MAX_CRITERIA_ITEMS {
                problems.push(format!(
                    "task `{}` has {} criteria, exceeds the bound of {MAX_CRITERIA_ITEMS}",
                    task.id,
                    task.acceptance_criteria.len()
                ));
            }
            for (criterion_index, criterion) in task.acceptance_criteria.iter().enumerate() {
                if criterion.is_empty() {
                    problems.push(format!(
                        "task `{}` criterion at index {criterion_index} is empty",
                        task.id
                    ));
                }
            }
        }

        if !duplicates.is_empty() {
            problems.push(format!(
                "duplicate task ids: {}",
                duplicates
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        if problems.is_empty() {
            Ok(())
        } else {
            Err(CorpusError::Validation(problems))
        }
    }

    pub fn by_id(&self, id: &str) -> Option<&TaskDescriptor> {
        self.tasks.iter().find(|task| task.id == id)
    }

    pub fn deceptive_tasks(&self) -> impl Iterator<Item = &TaskDescriptor> {
        self.tasks.iter().filter(|task| task.deceptive)
    }
}

/// A load/validation failure. `Json` wraps a serde parse error (unknown
/// fields, bad enum values, missing keys); `Validation` carries every
/// structural violation found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CorpusError {
    Json(String),
    Validation(Vec<String>),
}

impl CorpusError {
    pub fn json_message(&self) -> Option<&str> {
        match self {
            CorpusError::Json(message) => Some(message),
            CorpusError::Validation(_) => None,
        }
    }

    pub fn validation_problems(&self) -> Option<&[String]> {
        match self {
            CorpusError::Json(_) => None,
            CorpusError::Validation(problems) => Some(problems),
        }
    }
}

impl fmt::Display for CorpusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CorpusError::Json(message) => write!(f, "corpus JSON parse error: {message}"),
            CorpusError::Validation(problems) => {
                write!(f, "corpus validation failed ({}):", problems.len())?;
                for problem in problems {
                    write!(f, "\n  - {problem}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CorpusError {}

/// One completed benchmark run on one (task, model-lane) pair. All counters
/// are unsigned and only ever move up through the `record_*` / `add_*`
/// helpers (saturating), so arithmetic can neither underflow nor wrap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RunMetrics {
    /// Tasks whose result passed the expected verification.
    pub verified_ok: u32,
    /// Tasks that declared success without passing verification.
    pub false_completions: u32,
    /// Total model cost, in micro-units of the configured price.
    pub total_cost_micro: u64,
    /// Total wall time of the lane, milliseconds.
    pub wall_ms: u64,
    /// Tool invocations that failed and were retried or abandoned.
    pub tool_failures: u32,
    /// Model calls that had to be re-issued (rate limits, drops, malformed).
    pub model_retries: u32,
    /// Manual operator steering events (aborts, resumes, edits).
    pub interventions: u32,
}

impl RunMetrics {
    fn inc_u32(counter: &mut u32) {
        *counter = counter.saturating_add(1);
    }

    /// Monotonic recorders; each can only raise its counter.
    pub fn record_verified(&mut self) {
        Self::inc_u32(&mut self.verified_ok);
    }

    pub fn record_false_completion(&mut self) {
        Self::inc_u32(&mut self.false_completions);
    }

    pub fn record_tool_failure(&mut self) {
        Self::inc_u32(&mut self.tool_failures);
    }

    pub fn record_model_retry(&mut self) {
        Self::inc_u32(&mut self.model_retries);
    }

    pub fn record_intervention(&mut self) {
        Self::inc_u32(&mut self.interventions);
    }

    pub fn add_cost_micro(&mut self, amount: u64) {
        self.total_cost_micro = self.total_cost_micro.saturating_add(amount);
    }

    pub fn add_wall_ms(&mut self, amount: u64) {
        self.wall_ms = self.wall_ms.saturating_add(amount);
    }
}

/// The audit ranking, as a strict lexicographic chain over five signals:
///
/// 1. `verified_ok` — descending (verified correctness dominates everything),
/// 2. `false_completions` — ascending (fewer unverified success claims),
/// 3. `interventions` — ascending (autonomy; steering is a quality signal),
/// 4. `total_cost_micro` — ascending (cost only ever breaks a tie that
///    correctness and autonomy left open),
/// 5. `wall_ms` — ascending (final tiebreak).
///
/// Returns `false` for equal runs (`tool_failures` / `model_retries` are
/// report-only statistics that never enter the ranking). See also
/// [`RunMetrics::lexicographic_rank`], the compact 4-key snapshot of this
/// chain.
pub fn better_than(left: &RunMetrics, right: &RunMetrics) -> bool {
    match left.verified_ok.cmp(&right.verified_ok) {
        Ordering::Greater => return true,
        Ordering::Less => return false,
        Ordering::Equal => {}
    }
    match left.false_completions.cmp(&right.false_completions) {
        Ordering::Less => return true,
        Ordering::Greater => return false,
        Ordering::Equal => {}
    }
    match left.interventions.cmp(&right.interventions) {
        Ordering::Less => return true,
        Ordering::Greater => return false,
        Ordering::Equal => {}
    }
    match left.total_cost_micro.cmp(&right.total_cost_micro) {
        Ordering::Less => return true,
        Ordering::Greater => return false,
        Ordering::Equal => {}
    }
    left.wall_ms.cmp(&right.wall_ms) == Ordering::Less
}

/// RunMetrics is deliberately not `Ord`: the ranking is defined on the five
/// signals above, not on the whole struct, so expose it as a method pair.
impl RunMetrics {
    /// Is `self` ranked strictly better than `other` per the audit chain?
    pub fn better_than(&self, other: &Self) -> bool {
        better_than(self, other)
    }
}

/// The compact, sortable, **persistent** ranking key for cross-lane
/// comparison. Sorted ascending, a smaller key is a better run:
///
/// `(255 − verified_ok, false_completions, total_cost_micro, wall_ms)`
///
/// This is the audit chain minus the `interventions` signal, which lives in
/// [`better_than`] as the autonomy tiebreak between correctness and cost but
/// is deliberately not part of the four-key persisted record; `verified_ok`
/// saturates at 255 (the corpus is 11 tasks; 255 is unreachable).
pub fn lexicographic_rank(metrics: &RunMetrics) -> (u8, u32, u64, u64) {
    let verified = u8::try_from(metrics.verified_ok).unwrap_or(u8::MAX);
    (
        u8::MAX - verified,
        metrics.false_completions,
        metrics.total_cost_micro,
        metrics.wall_ms,
    )
}

impl RunMetrics {
    /// The compact persistent key; see [`lexicographic_rank`].
    pub fn lexicographic_rank(&self) -> (u8, u32, u64, u64) {
        lexicographic_rank(self)
    }
}

/// Offline by default; live benchmark lanes only activate when
/// `FAKTOR_BENCH_LIVE=1` is set. `cargo test -p faktor-tests-benchmark` in CI
/// never runs live lanes: the live skeleton test is `#[ignore]`-gated and the
/// gate itself returns `false` here, so nothing executes.
pub fn live_mode() -> bool {
    std::env::var(LIVE_ENV).is_ok_and(|value| value == "1")
}

/// Render the frozen corpus as a Markdown table skeleton. Live lanes append
/// per-run metric rows below the table; the corpus half of the report is
/// stable across lanes and runs so diffs stay readable.
pub fn render_corpus_markdown(corpus: &TaskCorpus) -> String {
    let mut out = String::new();
    out.push_str("# Faktor benchmark corpus\n\n");
    out.push_str(&format!(
        "Frozen 11-task corpus. Kernel {VERSION} (protocol {PROTOCOL_V756}, baseline {UX_BASELINE}). \
         Live lanes run only with {LIVE_ENV}=1.\n\n"
    ));
    out.push_str("| id | category | verification | deceptive | criteria | prompt chars |\n");
    out.push_str("| --- | --- | --- | --- | ---: | ---: |\n");
    for task in &corpus.tasks {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            task.id,
            task.category,
            task.expected_verification,
            if task.deceptive { "yes" } else { "no" },
            task.acceptance_criteria.len(),
            task.prompt_len(),
        ));
    }
    out.push_str("\n<!-- per-lane RunMetrics rows plug in below this table -->\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The audit list, in order — a frozen mirror the corpus must match
    /// exactly (same 11 ids, same order, same categories).
    const AUDIT_IDS: [&str; CORPUS_TASK_COUNT] = [
        "tiny_code_fix",
        "simple_search",
        "medium_bug",
        "large_refactor",
        "repo_exploration",
        "test_failure_diagnosis",
        "multi_file_feature",
        "ui_change",
        "long_horizon",
        "parallelizable",
        "misleading",
    ];

    const AUDIT_CATEGORIES: [&str; CORPUS_TASK_COUNT] = [
        "tiny code fix",
        "simple search/explanation",
        "medium bug",
        "large refactor",
        "repository exploration",
        "test failure diagnosis",
        "multi-file feature",
        "UI change",
        "long-horizon task",
        "parallelizable task",
        "intentionally misleading task",
    ];

    /// Parse the embedded corpus into a `Value` so tests can corrupt a copy
    /// without touching the frozen original.
    fn corrupted() -> serde_json::Value {
        serde_json::from_str(CORPUS_JSON).expect("embedded corpus must be valid JSON")
    }

    #[test]
    fn embedded_corpus_is_valid_and_mirrors_the_audit_list_exactly() {
        let corpus = TaskCorpus::load_embedded().expect("frozen corpus must load");
        assert_eq!(corpus.tasks.len(), CORPUS_TASK_COUNT);

        let ids: Vec<&str> = corpus.tasks.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(
            ids, AUDIT_IDS,
            "corpus order/ids drifted from the audit list"
        );

        for (task, expected_category) in corpus.tasks.iter().zip(AUDIT_CATEGORIES) {
            assert_eq!(task.category, expected_category);
        }

        for task in &corpus.tasks {
            assert!(!task.prompt.is_empty(), "task {} prompt empty", task.id);
            assert!(task.prompt_len() <= MAX_PROMPT_CHARS);
            assert!(!task.acceptance_criteria.is_empty());
            assert!(task.acceptance_criteria.len() <= MAX_CRITERIA_ITEMS);
            assert!(
                !task.changed_files_hint.is_empty(),
                "task {} must carry a changed-files hint",
                task.id
            );
        }

        let deceptive: Vec<&str> = corpus.deceptive_tasks().map(|t| t.id.as_str()).collect();
        assert_eq!(
            deceptive,
            vec!["misleading"],
            "exactly the misleading task is deceptive"
        );

        let verification_kinds: HashSet<ExpectedVerification> = corpus
            .tasks
            .iter()
            .map(|t| t.expected_verification)
            .collect();
        assert_eq!(
            verification_kinds,
            HashSet::from([
                ExpectedVerification::BuildAndTest,
                ExpectedVerification::Build,
                ExpectedVerification::None,
            ]),
            "the corpus must exercise every verification policy"
        );

        for id in AUDIT_IDS {
            assert!(corpus.by_id(id).is_some(), "by_id must find {id}");
        }
        assert!(corpus.by_id("no_such_task").is_none());
    }

    #[test]
    fn hostile_unknown_fields_are_hard_parse_errors() {
        let mut value = corrupted();
        value["tasks"][0]["bribe_field"] = serde_json::json!("make the loader ignore this");
        let hostile = serde_json::to_string(&value).unwrap();
        let err = TaskCorpus::load_json(&hostile).unwrap_err();
        assert!(
            err.json_message().is_some(),
            "unknown task field must fail at parse: {err}"
        );
        assert!(err.json_message().unwrap().contains("bribe_field"));
    }

    #[test]
    fn hostile_enum_and_missing_keys_are_hard_parse_errors() {
        let mut bad_enum = corrupted();
        bad_enum["tasks"][0]["expected_verification"] = serde_json::json!("build+test+extras");
        let err = TaskCorpus::load_json(&serde_json::to_string(&bad_enum).unwrap()).unwrap_err();
        assert!(
            err.json_message().is_some(),
            "bad enum must fail at parse: {err}"
        );

        let mut missing_prompt = corrupted();
        missing_prompt["tasks"][0]
            .as_object_mut()
            .unwrap()
            .remove("prompt");
        let err =
            TaskCorpus::load_json(&serde_json::to_string(&missing_prompt).unwrap()).unwrap_err();
        assert!(
            err.json_message().is_some(),
            "missing prompt must fail at parse: {err}"
        );

        let mut missing_tasks_key = corrupted();
        missing_tasks_key.as_object_mut().unwrap().remove("tasks");
        let err =
            TaskCorpus::load_json(&serde_json::to_string(&missing_tasks_key).unwrap()).unwrap_err();
        assert!(
            err.json_message().is_some(),
            "missing tasks key must fail: {err}"
        );
    }

    #[test]
    fn duplicated_ids_are_rejected_with_the_offenders_named() {
        let mut value = corrupted();
        value["tasks"][10]["id"] = serde_json::json!("tiny_code_fix");
        let err = TaskCorpus::load_json(&serde_json::to_string(&value).unwrap()).unwrap_err();
        let problems = err
            .validation_problems()
            .expect("must be a validation error");
        assert!(
            problems
                .iter()
                .any(|p| p.contains("duplicate task ids") && p.contains("tiny_code_fix")),
            "problems: {problems:?}"
        );
    }

    #[test]
    fn wrong_corpus_size_is_rejected_either_direction() {
        let mut one_fewer = corrupted();
        one_fewer["tasks"].as_array_mut().unwrap().pop();
        let err = TaskCorpus::load_json(&serde_json::to_string(&one_fewer).unwrap()).unwrap_err();
        let problems = err
            .validation_problems()
            .expect("must be a validation error");
        assert!(
            problems
                .iter()
                .any(|p| p.contains("exactly 11 tasks") && p.contains("10")),
            "problems: {problems:?}"
        );

        let mut one_more = corrupted();
        let mut intruder = corrupted()["tasks"][0].clone();
        intruder["id"] = serde_json::json!("unfrozen_extra_task");
        intruder["category"] = serde_json::json!("corpus drift");
        one_more["tasks"].as_array_mut().unwrap().push(intruder);
        let err = TaskCorpus::load_json(&serde_json::to_string(&one_more).unwrap()).unwrap_err();
        let problems = err
            .validation_problems()
            .expect("must be a validation error");
        assert!(
            problems
                .iter()
                .any(|p| p.contains("exactly 11 tasks") && p.contains("12")),
            "problems: {problems:?}"
        );
    }

    #[test]
    fn oversized_prompt_is_rejected_and_multi_violations_are_all_reported() {
        let mut value = corrupted();

        // Byte-heavy but char-counted: 2001 × 3-byte chars is 6003 bytes yet
        // only 2001 chars — under the 4000-char bound, so it must PASS. This
        // pins the bound semantics to characters, not bytes.
        let wide = "\u{20ac}".repeat(2001);
        value["tasks"][0]["prompt"] = serde_json::Value::String(wide);
        let ok = TaskCorpus::load_json(&serde_json::to_string(&value).unwrap()).unwrap();
        assert_eq!(ok.tasks[0].prompt_len(), 2001);

        // Now the real hostile payloads, all at once: every violation must be
        // reported, not just the first one found.
        let mut multi = corrupted();
        multi["tasks"][0]["prompt"] = serde_json::Value::String("x".repeat(MAX_PROMPT_CHARS + 1));
        multi["tasks"][1]["prompt"] = serde_json::Value::String(String::new());
        multi["tasks"][2]["acceptance_criteria"] = serde_json::Value::Array(Vec::new());
        let too_many_criteria = (0..=MAX_CRITERIA_ITEMS)
            .map(|i| format!("criterion {i}"))
            .collect::<Vec<_>>();
        multi["tasks"][3]["acceptance_criteria"] = serde_json::Value::Array(
            too_many_criteria
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        );
        multi["tasks"][4]["acceptance_criteria"] = serde_json::json!(["ok", ""]);

        let err = TaskCorpus::load_json(&serde_json::to_string(&multi).unwrap()).unwrap_err();
        let problems = err
            .validation_problems()
            .expect("must be a validation error");
        assert!(
            problems
                .iter()
                .any(|p| p.contains("tiny_code_fix") && p.contains("4001")),
            "problems: {problems:?}"
        );
        assert!(
            problems
                .iter()
                .any(|p| p.contains("simple_search") && p.contains("empty prompt")),
            "problems: {problems:?}"
        );
        assert!(
            problems
                .iter()
                .any(|p| p.contains("medium_bug") && p.contains("no acceptance criteria")),
            "problems: {problems:?}"
        );
        assert!(
            problems
                .iter()
                .any(|p| p.contains("large_refactor") && p.contains("9 criteria")),
            "problems: {problems:?}"
        );
        assert!(
            problems
                .iter()
                .any(|p| p.contains("repo_exploration") && p.contains("empty")),
            "problems: {problems:?}"
        );
    }

    #[test]
    fn better_than_matrix_verified_correctness_beats_cheap_runs() {
        let verified = RunMetrics {
            verified_ok: 1,
            total_cost_micro: 1_000_000,
            ..RunMetrics::default()
        };
        let cheap_fail = RunMetrics {
            verified_ok: 0,
            total_cost_micro: 1,
            ..RunMetrics::default()
        };
        assert!(verified.better_than(&cheap_fail));
        assert!(!cheap_fail.better_than(&verified));
    }

    #[test]
    fn better_than_matrix_fewer_false_completions_wins_after_verified() {
        let honest = RunMetrics {
            verified_ok: 2,
            false_completions: 0,
            total_cost_micro: 9_000_000,
            ..RunMetrics::default()
        };
        let liar = RunMetrics {
            verified_ok: 2,
            false_completions: 1,
            total_cost_micro: 1,
            ..RunMetrics::default()
        };
        assert!(
            honest.better_than(&liar),
            "correctness first, cost never rescues a liar"
        );
        assert!(!liar.better_than(&honest));
    }

    #[test]
    fn better_than_matrix_cost_is_a_tiebreak_only_after_correctness() {
        let cheaper = RunMetrics {
            verified_ok: 1,
            false_completions: 0,
            total_cost_micro: 400,
            ..RunMetrics::default()
        };
        let pricier = RunMetrics {
            verified_ok: 1,
            false_completions: 0,
            total_cost_micro: 900,
            ..RunMetrics::default()
        };
        assert!(
            cheaper.better_than(&pricier),
            "equal correctness: lower cost wins"
        );
        assert!(!pricier.better_than(&cheaper));

        let expensive_verified = RunMetrics {
            verified_ok: 2,
            false_completions: 1,
            total_cost_micro: 900,
            ..RunMetrics::default()
        };
        let cheap_failed = RunMetrics {
            verified_ok: 1,
            false_completions: 0,
            total_cost_micro: 1,
            ..RunMetrics::default()
        };
        assert!(
            expensive_verified.better_than(&cheap_failed),
            "verified count dominates: cost must not rescue a lower-verified run"
        );
        assert!(!cheap_failed.better_than(&expensive_verified));
    }

    #[test]
    fn better_than_matrix_interventions_rank_before_cost_wall_is_final_tiebreak() {
        let autonomous_pricier = RunMetrics {
            verified_ok: 1,
            interventions: 0,
            total_cost_micro: 100_000,
            wall_ms: 50,
            ..RunMetrics::default()
        };
        let steered_cheaper = RunMetrics {
            verified_ok: 1,
            interventions: 4,
            total_cost_micro: 1,
            wall_ms: 50,
            ..RunMetrics::default()
        };
        assert!(
            autonomous_pricier.better_than(&steered_cheaper),
            "autonomy (interventions) is ranked ahead of cost"
        );
        assert!(!steered_cheaper.better_than(&autonomous_pricier));

        let fast = RunMetrics {
            verified_ok: 1,
            total_cost_micro: 10,
            wall_ms: 5,
            ..RunMetrics::default()
        };
        let slow = RunMetrics {
            verified_ok: 1,
            total_cost_micro: 10,
            wall_ms: 500,
            ..RunMetrics::default()
        };
        assert!(fast.better_than(&slow), "wall time is the final tiebreak");
        assert!(!slow.better_than(&fast));
        assert!(
            !fast.better_than(&fast),
            "never strictly better than itself"
        );
    }

    #[test]
    fn better_than_is_a_strict_total_order_over_the_ranking_signals() {
        // Exhaustive grid over the five ranking keys: for any two DISTINCT
        // runs exactly one direction must hold, and transitivity must never
        // be violated (report-only stats are pinned so equality is exact).
        let mut runs = Vec::new();
        for verified in 0..=2u32 {
            for false_completions in 0..=1u32 {
                for interventions in 0..=1u32 {
                    for cost in [1u64, 50, 900] {
                        for wall in [1u64, 50, 900] {
                            runs.push(RunMetrics {
                                verified_ok: verified,
                                false_completions,
                                interventions,
                                total_cost_micro: cost,
                                wall_ms: wall,
                                tool_failures: 0,
                                model_retries: 0,
                            });
                        }
                    }
                }
            }
        }
        assert_eq!(runs.len(), 3 * 2 * 2 * 3 * 3);

        for left in &runs {
            assert!(!left.better_than(left), "irreflexive");
            for right in &runs {
                if left == right {
                    continue;
                }
                let left_wins = left.better_than(right);
                let right_wins = right.better_than(left);
                assert!(
                    left_wins ^ right_wins,
                    "distinct runs must be strictly ordered one way: {left:?} vs {right:?}"
                );
            }
        }

        for a in &runs {
            for b in &runs {
                for c in &runs {
                    if a.better_than(b) && b.better_than(c) {
                        assert!(a.better_than(c), "transitivity violated: {a:?} {b:?} {c:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn rank_tuple_agrees_with_better_than_when_interventions_tie() {
        // `lexicographic_rank` is the audit chain without the interventions
        // key, so for equal interventions the tuple order and the full chain
        // must agree exactly.
        let mut runs = Vec::new();
        for verified in 0..=3u32 {
            for false_completions in [0u32, 1, 2] {
                for cost in [1u64, 100, 100_000] {
                    for wall in [1u64, 100] {
                        runs.push(RunMetrics {
                            verified_ok: verified,
                            false_completions,
                            total_cost_micro: cost,
                            wall_ms: wall,
                            ..RunMetrics::default()
                        });
                    }
                }
            }
        }
        for left in &runs {
            for right in &runs {
                assert_eq!(
                    left.better_than(right),
                    left.lexicographic_rank() < right.lexicographic_rank(),
                    "rank must mirror better_than at equal interventions: {left:?} {right:?}"
                );
            }
        }

        let saturated = RunMetrics {
            verified_ok: u32::MAX,
            ..RunMetrics::default()
        };
        assert_eq!(
            saturated.lexicographic_rank().0,
            0,
            "verified saturates into slot 0"
        );
    }

    #[test]
    fn metric_counters_saturate_and_are_strictly_monotonic() {
        let mut metrics = RunMetrics::default();
        assert_eq!(metrics.verified_ok, 0);
        metrics.record_verified();
        metrics.record_false_completion();
        metrics.record_tool_failure();
        metrics.record_model_retry();
        metrics.record_intervention();
        metrics.add_cost_micro(150);
        metrics.add_wall_ms(25);
        assert_eq!(
            metrics,
            RunMetrics {
                verified_ok: 1,
                false_completions: 1,
                total_cost_micro: 150,
                wall_ms: 25,
                tool_failures: 1,
                model_retries: 1,
                interventions: 1,
            }
        );

        // Hammer every counter to its ceiling: the next record must saturate,
        // never wrap and never panic — no underflow, no overflow.
        let mut hammered = RunMetrics {
            verified_ok: u32::MAX - 1,
            false_completions: u32::MAX,
            tool_failures: u32::MAX,
            model_retries: u32::MAX,
            interventions: u32::MAX,
            total_cost_micro: u64::MAX - 1,
            wall_ms: u64::MAX - 1,
        };
        hammered.record_verified();
        hammered.record_false_completion();
        hammered.record_tool_failure();
        hammered.record_model_retry();
        hammered.record_intervention();
        hammered.add_cost_micro(2);
        hammered.add_wall_ms(2);
        assert_eq!(hammered.verified_ok, u32::MAX);
        assert_eq!(hammered.false_completions, u32::MAX);
        assert_eq!(hammered.tool_failures, u32::MAX);
        assert_eq!(hammered.model_retries, u32::MAX);
        assert_eq!(hammered.interventions, u32::MAX);
        assert_eq!(hammered.total_cost_micro, u64::MAX);
        assert_eq!(hammered.wall_ms, u64::MAX);

        // Monotonic: any prefix of recording never exceeds a later prefix.
        let mut growing = RunMetrics::default();
        let mut last = growing;
        for step in 1..=50u64 {
            if step % 5 == 0 {
                growing.record_verified();
            }
            if step % 7 == 0 {
                growing.record_false_completion();
            }
            growing.add_cost_micro(step);
            growing.add_wall_ms(step);
            assert!(
                last.verified_ok <= growing.verified_ok
                    && last.false_completions <= growing.false_completions
                    && last.total_cost_micro <= growing.total_cost_micro
                    && last.wall_ms <= growing.wall_ms
            );
            last = growing;
        }
        // Commutative accumulation: same total regardless of interleaving.
        let mut ordered = RunMetrics::default();
        for step in 1..=50u64 {
            ordered.add_cost_micro(step);
            ordered.add_wall_ms(step);
        }
        assert_eq!(ordered.total_cost_micro, 50 * 51 / 2);
        assert_eq!(ordered.wall_ms, 50 * 51 / 2);
    }

    #[test]
    fn markdown_report_renders_all_eleven_rows() {
        let corpus = TaskCorpus::load_embedded().expect("frozen corpus must load");
        let rendered = render_corpus_markdown(&corpus);

        assert!(rendered.starts_with("# Faktor benchmark corpus"));
        assert!(
            rendered.contains(VERSION),
            "report must stamp the kernel version"
        );
        assert!(rendered.contains(PROTOCOL_V756) && rendered.contains(UX_BASELINE));

        let data_rows: Vec<&str> = rendered
            .lines()
            .filter(|line| line.starts_with("| ") && !line.starts_with("| ---"))
            .collect();
        // Header row + 11 data rows; the separator column row is excluded.
        assert_eq!(data_rows.len(), 1 + CORPUS_TASK_COUNT);

        for task in &corpus.tasks {
            let row = data_rows
                .iter()
                .find(|line| line.contains(&format!("| {} |", task.id)))
                .unwrap_or_else(|| panic!("missing row for task {}", task.id));
            assert!(row.contains(&task.category));
            assert!(row.contains(task.expected_verification.as_str()));
            assert!(row.contains(&task.acceptance_criteria.len().to_string()));
        }

        // Deceptive tasks are flagged in the report so human readers are not
        // misled by a task whose whole point is to be misleading.
        assert!(rendered
            .lines()
            .any(|line| line.contains("misleading") && line.contains("| yes |")));
        assert_eq!(
            rendered
                .lines()
                .filter(|line| line.contains("| yes |"))
                .count(),
            1
        );
    }

    #[test]
    fn live_mode_parses_only_the_exact_gate_value() {
        // The gate semantics are pinned here without touching process env
        // (which would race parallel tests): only "1" flips live mode on.
        for (value, expected) in [("1", true), ("0", false), ("true", false), ("", false)] {
            assert_eq!(parse_gate(value), expected, "gate value `{value}`");
        }
    }

    /// Pure extraction of the gate rule; [`live_mode`] applies it to the env.
    fn parse_gate(value: &str) -> bool {
        value == "1"
    }

    /// Live-lane skeleton. `#[ignore]`d: CI runs it never, and even a manual
    /// `--ignored` run executes nothing unless `FAKTOR_BENCH_LIVE=1` is set.
    /// The live lane body (per-task × per-model runs on fresh worktrees,
    /// `RunMetrics` collection, markdown report emission) plugs in here and is
    /// deliberately left unwired: routing policy belongs to the economy crate.
    #[test]
    #[ignore = "live benchmark lanes are env-gated by FAKTOR_BENCH_LIVE=1"]
    fn live_benchmark_requires_env() {
        if !live_mode() {
            eprintln!("FAKTOR_BENCH_LIVE != 1: live lanes disabled, running nothing");
            return;
        }
        // Live mode is on: smoke-check the substrate the lanes will consume.
        // Live per-model lanes on isolated worktrees plug in below; this test
        // stays offline-safe (no network, no providers) by design.
        let corpus = TaskCorpus::load_embedded().expect("frozen corpus must load");
        eprintln!(
            "live lanes active: {} frozen tasks, deceptive={}, kernel {VERSION}",
            corpus.tasks.len(),
            corpus.deceptive_tasks().count(),
        );
    }
}
