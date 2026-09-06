//! Real-daemon benchmark driver (subprocess mode).
//!
//! Drives the ACTUAL daemon — the built `faktor-cli serve` binary — over
//! its native HTTP API (never the RouterService, never in-process
//! shortcuts): one fresh daemon per task, one fresh session whose durable
//! workspace root is a temp COPY of the task repository, the task prompt
//! issued through the daemon's own prompt path, then the repository-native
//! `verify.sh` run in that workspace copy by the harness.
//!
//! The daemon is spawned with a pinned provider/model from the
//! `FAKTOR_BENCH_*` environment (see `README.md`); without them the whole
//! corpus is a documented skip. The harness also plays the role of the UI
//! permission channel (deterministic auto-allow of every pending
//! permission/question — the sandbox policy still gates what a command may
//! do). Real runs are `#[ignore]`-gated and require provider keys; they
//! are NOT part of the normal test run.

use crate::process::spawn_killable as spawn_daemon_child;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::corpus::{Corpus, Task};
use crate::fsutil;
use crate::report::{CorpusReport, TaskResult};
use crate::score::{corroborate_records, criteria_met, score_summary, RecordCriteria};
use crate::toolchain::require_toolchain;
use crate::verify::{run_verify_sh, VerifyOptions};

/// Wait budgets of the driver itself (daemon startup/readiness are fast;
/// the per-task model budget is `task_timeout`).
const STARTUP_TIMEOUT: Duration = Duration::from_secs(60);
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const POLL_INTERVAL: Duration = Duration::from_millis(400);
const HTTP_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_SUMMARY_CHARS: usize = 32 * 1024;
/// Probe bound for the durable verification-record endpoint (the numeric
/// task id of a session is not exposed on the read surface; the endpoint
/// is session-scoped and workspace-guarded server-side, so probing the
/// first ids is safe and bounded).
const MAX_TASK_ID_PROBE: u64 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    Google,
    DeepSeek,
    Gateway,
    Ollama,
}

impl ProviderKind {
    pub fn parse(s: &str) -> Option<ProviderKind> {
        match s.to_ascii_lowercase().as_str() {
            "anthropic" => Some(ProviderKind::Anthropic),
            "openai" | "open_ai" => Some(ProviderKind::OpenAi),
            "google" | "gemini" => Some(ProviderKind::Google),
            "deepseek" => Some(ProviderKind::DeepSeek),
            "gateway" => Some(ProviderKind::Gateway),
            "ollama" => Some(ProviderKind::Ollama),
            _ => None,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::OpenAi => "openai",
            ProviderKind::Google => "google",
            ProviderKind::DeepSeek => "deepseek",
            ProviderKind::Gateway => "gateway",
            ProviderKind::Ollama => "ollama",
        }
    }

    /// Kinds whose adapter accepts a custom base URL.
    pub const fn accepts_base_url(self) -> bool {
        matches!(
            self,
            ProviderKind::OpenAi
                | ProviderKind::DeepSeek
                | ProviderKind::Gateway
                | ProviderKind::Ollama
        )
    }

    /// Only Ollama is keyless by design.
    pub const fn needs_key(self) -> bool {
        !matches!(self, ProviderKind::Ollama)
    }
}

/// Everything a real run needs, resolved from the environment (or built
/// explicitly by tests). Immutable per corpus run.
#[derive(Debug, Clone)]
pub struct DaemonBenchConfig {
    pub provider: ProviderKind,
    pub model: String,
    /// Name of the env var the daemon reads the API key from. The key
    /// itself never appears in any file or log.
    pub api_key_env: String,
    /// Custom endpoint for kinds that accept one (drives the config's
    /// `base_url` and the daemon sandbox network rows).
    pub base_url: Option<String>,
    /// Extra sandbox network allowlist rows.
    pub network_rows: Vec<String>,
    /// Per-task model wall budget.
    pub task_timeout: Duration,
    pub verify: VerifyOptions,
    /// Append the immutable criteria block to the prompt and require the
    /// final summary to name each `crit-NN` key.
    pub append_criteria: bool,
    /// Free-form run tag recorded in report notes.
    pub tag: String,
}

impl DaemonBenchConfig {
    pub const ENV_PROVIDER: &'static str = "FAKTOR_BENCH_PROVIDER";
    pub const ENV_MODEL: &'static str = "FAKTOR_BENCH_MODEL";
    pub const ENV_API_KEY_ENV: &'static str = "FAKTOR_BENCH_API_KEY_ENV";
    pub const ENV_API_KEY: &'static str = "FAKTOR_BENCH_API_KEY";
    pub const ENV_BASE_URL: &'static str = "FAKTOR_BENCH_BASE_URL";
    pub const ENV_NETWORK: &'static str = "FAKTOR_BENCH_NETWORK";
    pub const ENV_BIN: &'static str = "FAKTOR_BENCH_BIN";
    pub const ENV_TASK_TIMEOUT_S: &'static str = "FAKTOR_BENCH_TASK_TIMEOUT_S";
    pub const ENV_SUMMARY: &'static str = "FAKTOR_BENCH_SUMMARY_PROMPT";
    pub const ENV_TAG: &'static str = "FAKTOR_BENCH_TAG";
    pub const ENV_ONLY: &'static str = "FAKTOR_BENCH_ONLY";

    /// Resolve the configuration from the process environment.
    ///
    /// `Ok(None)` = no provider/model configured → the corpus run must be
    /// skipped with the returned note. `Err` = hard configuration error.
    pub fn from_env() -> Result<Option<(DaemonBenchConfig, String)>, String> {
        let Ok(provider_raw) = std::env::var(Self::ENV_PROVIDER) else {
            return Ok(None);
        };
        let Some(provider) = ProviderKind::parse(&provider_raw) else {
            return Err(format!(
                "unknown {}={provider_raw:?} (anthropic|openai|google|deepseek|gateway|ollama)",
                Self::ENV_PROVIDER
            ));
        };
        let model = std::env::var(Self::ENV_MODEL)
            .map_err(|_| format!("{} is required for real-model runs", Self::ENV_MODEL))?;
        if model.is_empty() || model.chars().count() > 128 {
            return Err(format!("{} must be 1..=128 chars", Self::ENV_MODEL));
        }
        let api_key_env = std::env::var(Self::ENV_API_KEY_ENV).unwrap_or_else(|_| {
            if provider.needs_key() {
                Self::ENV_API_KEY.to_string()
            } else {
                String::new()
            }
        });
        if provider.needs_key() && std::env::var(&api_key_env).map_or(true, |v| v.trim().is_empty())
        {
            return Ok(None);
        }
        if !provider.needs_key() && !api_key_env.is_empty() {
            return Err(format!(
                "provider {} is keyless; {} must not be set",
                provider.as_str(),
                Self::ENV_API_KEY_ENV
            ));
        }
        let base_url = match std::env::var(Self::ENV_BASE_URL) {
            Ok(v) if !v.trim().is_empty() => {
                if !provider.accepts_base_url() {
                    return Err(format!(
                        "provider {} does not accept {} (no custom-endpoint adapter)",
                        provider.as_str(),
                        Self::ENV_BASE_URL
                    ));
                }
                Some(v)
            }
            _ => None,
        };
        let network_rows: Vec<String> = match std::env::var(Self::ENV_NETWORK) {
            Ok(v) if !v.trim().is_empty() => v.split(',').map(|s| s.trim().to_string()).collect(),
            _ => Vec::new(),
        };
        let task_timeout = std::env::var(Self::ENV_TASK_TIMEOUT_S)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or(Duration::from_secs(600));
        let append_criteria = std::env::var(Self::ENV_SUMMARY)
            .map(|v| v != "0")
            .unwrap_or(true);
        let tag = std::env::var(Self::ENV_TAG).unwrap_or_default();
        let note = format!(
            "real-model run: provider={} model={}",
            provider.as_str(),
            model
        );
        Ok(Some((
            DaemonBenchConfig {
                provider,
                model,
                api_key_env,
                base_url,
                network_rows,
                task_timeout,
                verify: VerifyOptions::default().with_env(),
                append_criteria,
                tag,
            },
            note,
        )))
    }
}

/// Resolve the daemon binary. Precedence: `FAKTOR_BENCH_BIN` env, the
/// compile-time `CARGO_BIN_EXE_faktor-cli` hint (cargo sets it when it
/// builds the faktor-cli bin for this crate's test targets), then the
/// workspace target dirs next to this crate.
pub fn resolve_binary(compile_hint: Option<&str>) -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var(DaemonBenchConfig::ENV_BIN) {
        let p = PathBuf::from(explicit);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Some(hint) = compile_hint {
        let p = PathBuf::from(hint);
        if p.is_file() {
            return Some(p);
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    for profile in ["release", "debug"] {
        let p = manifest
            .join("..")
            .join("..")
            .join("target")
            .join(profile)
            .join("faktor-cli");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

/// The prompt handed to the model: the immutable task.md plus the fixed
/// summary contract embedding the immutable criteria lines.
fn build_prompt(task: &Task, cfg: &DaemonBenchConfig) -> String {
    let mut prompt = task.task_md.clone();
    if cfg.append_criteria {
        prompt.push_str("\n\n## Acceptance criteria\n");
        prompt.push_str(
            "The benchmark judges this run against the criteria below. \
             Finish by writing a FINAL SUMMARY that names each key verbatim, \
             one line per criterion, in this exact shape:\n",
        );
        for c in &task.criteria {
            prompt.push_str(&format!("- {}: PASS or FAIL — one evidence line\n", c.key));
        }
        prompt.push_str("\nThe repository's own test suite remains the objective gate.\n");
    }
    prompt
}

// ---------------------------------------------------------------- HTTP

pub(crate) struct Client {
    inner: reqwest::blocking::Client,
    base: String,
    password: String,
}

impl Client {
    fn new(base: String, password: String) -> Result<Client, String> {
        let inner = reqwest::blocking::Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .map_err(|e| format!("http client: {e}"))?;
        Ok(Client {
            inner,
            base,
            password,
        })
    }

    fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, String> {
        let url = format!("{}{}", self.base, path);
        let mut req = self
            .inner
            .request(method.clone(), &url)
            .header("Authorization", bearer(&self.password));
        if let Some(b) = body {
            req = req.json(b);
        }
        let resp = req.send().map_err(|e| format!("{method} {path}: {e}"))?;
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        if !status.is_success() {
            let snippet: String = body.chars().take(2048).collect();
            return Err(format!("{method} {path}: HTTP {status}: {snippet}"));
        }
        if body.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&body).map_err(|e| format!("{method} {path}: bad JSON: {e}"))
    }

    fn get(&self, path: &str) -> Result<Value, String> {
        self.request(reqwest::Method::GET, path, None)
    }

    fn post(&self, path: &str, body: &Value) -> Result<Value, String> {
        self.request(reqwest::Method::POST, path, Some(body))
    }
}

fn bearer(password: &str) -> String {
    format!("Bearer {password}")
}

// ---------------------------------------------------------------- daemon

/// One spawned daemon. Dropping it kills the whole daemon process group
/// and reaps the child.
pub(crate) struct DaemonProcess {
    child: Option<Child>,
    /// Reader thread of the daemon's stdout lines (ends at EOF once the
    /// daemon dies).
    _reader: Option<std::thread::JoinHandle<()>>,
    /// Bounded tail of the daemon's stderr (diagnostics for spawn errors).
    _stderr: Arc<std::sync::Mutex<String>>,
}

impl DaemonProcess {
    /// Spawn `faktor-cli serve --port 0` over the given data dir and
    /// config file; waits for the frozen startup line and returns the
    /// authenticated client against the bound port.
    pub(crate) fn spawn(
        bin: &Path,
        data_dir: &Path,
        config_file: &Path,
    ) -> Result<(DaemonProcess, Client), String> {
        let password = random_hex();
        let mut command = Command::new(bin);
        command
            .arg("serve")
            .arg("--port")
            .arg("0")
            .arg("--data-dir")
            .arg(data_dir)
            .arg("--config")
            .arg(config_file)
            .env("FAKTOR_SERVER_PASSWORD", &password)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = spawn_daemon_child(&mut command)
            .map_err(|e| format!("failed to spawn daemon {bin:?}: {e}"))?;
        let stdout = child.stdout.take().expect("stdout piped");
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let reader = std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let stderr_tail = Arc::new(std::sync::Mutex::new(String::new()));
        if let Some(stderr) = child.stderr.take() {
            let tail = stderr_tail.clone();
            std::thread::spawn(move || {
                use std::io::BufRead;
                for line in std::io::BufReader::new(stderr).lines() {
                    let Ok(line) = line else { break };
                    let mut buf = tail.lock().expect("stderr tail poisoned");
                    buf.push_str(&line);
                    buf.push('\n');
                    let keep = 16 * 1024;
                    let overflow = buf.len().saturating_sub(keep);
                    if overflow > 0 {
                        buf.drain(..overflow);
                    }
                }
            });
        }
        let base_url = match wait_for_startup_line(&mut child, &rx) {
            Ok(u) => u,
            Err(e) => {
                let tail = stderr_tail.lock().expect("stderr tail poisoned");
                let diag: String = tail.chars().take(4096).collect();
                return Err(format!("{e}; daemon stderr: {diag}"));
            }
        };
        let client = Client::new(base_url, password)?;
        Ok((
            DaemonProcess {
                child: Some(child),
                _reader: Some(reader),
                _stderr: stderr_tail,
            },
            client,
        ))
    }

    /// Wait until `/native/ready` answers `{"ready": true}` (bounded).
    pub(crate) fn wait_ready(&mut self, client: &Client) -> Result<(), String> {
        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            match client.get("/native/ready") {
                Ok(v) => {
                    if v.get("ready").and_then(Value::as_bool) == Some(true) {
                        return Ok(());
                    }
                }
                Err(e) => {
                    if !e.contains("HTTP") && Instant::now() >= deadline {
                        return Err(format!("daemon never became ready: {e}"));
                    }
                }
            }
            if self
                .child
                .as_mut()
                .and_then(|c| c.try_wait().ok())
                .flatten()
                .is_some()
            {
                return Err("daemon exited during startup".into());
            }
            if Instant::now() >= deadline {
                return Err("daemon never became ready (timeout)".into());
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

impl DaemonProcess {
    /// Bounded stderr diagnostics tail (debug aid).
    pub(crate) fn stderr_tail(&self) -> String {
        self._stderr.lock().map(|t| t.clone()).unwrap_or_default()
    }
}

impl Drop for DaemonProcess {
    fn drop(&mut self) {
        if let Some(child) = self.child.as_mut() {
            kill_group(child);
            let _ = child.wait();
        }
    }
}

fn wait_for_startup_line(child: &mut Child, rx: &Receiver<String>) -> Result<String, String> {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(line) => {
                let needle = "http://127.0.0.1:";
                if let Some(pos) = line.find(needle) {
                    let port: String = line[pos + needle.len()..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    if let Ok(port) = port.parse::<u16>() {
                        if port > 0 {
                            return Ok(format!("http://127.0.0.1:{port}"));
                        }
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if let Ok(Some(status)) = child.try_wait() {
                    return Err(format!("daemon exited before the startup line ({status})"));
                }
                if Instant::now() >= deadline {
                    return Err("daemon never printed the startup line".into());
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err("daemon stdout closed before the startup line".into());
            }
        }
    }
}

#[cfg(unix)]
fn kill_group(child: &mut Child) {
    let pid = child.id() as i32;
    // SAFETY: pgid == pid because the daemon was spawned with
    // process_group(0) (spawn_killable). ESRCH = already gone.
    unsafe {
        libc::killpg(pid, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_group(child: &mut Child) {
    let _ = child.kill();
}
/// 64-hex ephemeral password (localhost-only, per-run; the daemon
/// compares constant-time and never logs it).
fn random_hex() -> String {
    use std::hash::{BuildHasher, Hash, Hasher};
    let a = std::collections::hash_map::RandomState::new();
    let mut h = a.build_hasher();
    std::thread::current().id().hash(&mut h);
    std::process::id().hash(&mut h);
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .hash(&mut h);
    format!("{:016x}{:016x}", h.finish(), h.finish())
}

// ---------------------------------------------------------------- JSON reads

/// Recursive, normalized field find (tolerant of snake_case/camelCase
/// drift and nesting, bounded depth): the wire surface is strict
/// server-side, but the scorer reads defensively so future additive
/// fields never break it.
fn find_field<'a>(v: &'a Value, name: &str, depth: usize) -> Option<&'a Value> {
    if depth == 0 {
        return None;
    }
    let norm = |s: &str| -> String {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(|c| c.to_lowercase())
            .collect()
    };
    let want = norm(name);
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                if norm(k) == want {
                    return Some(val);
                }
            }
            for val in map.values() {
                if let Some(found) = find_field(val, name, depth - 1) {
                    return Some(found);
                }
            }
            None
        }
        Value::Array(items) => {
            for item in items {
                if let Some(found) = find_field(item, name, depth - 1) {
                    return Some(found);
                }
            }
            None
        }
        _ => None,
    }
}

/// The latest assistant/agent message text (the final summary), bounded.
fn assistant_summary(messages: &Value) -> String {
    let arr = messages
        .get("messages")
        .and_then(Value::as_array)
        .or_else(|| messages.as_array());
    let Some(arr) = arr else {
        return String::new();
    };
    for msg in arr {
        let role = msg
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        if !matches!(role.as_str(), "assistant" | "agent" | "model") {
            continue;
        }
        let mut text = String::new();
        if let Some(parts) = msg.get("parts").and_then(Value::as_array) {
            for part in parts {
                let kind = part
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                if matches!(kind.as_str(), "text" | "summary") {
                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                        text.push('\n');
                    }
                }
            }
        }
        if text.is_empty() {
            for key in ["text", "content"] {
                if let Some(t) = msg.get(key).and_then(Value::as_str) {
                    text.push_str(t);
                    break;
                }
            }
        }
        text.truncate(MAX_SUMMARY_CHARS);
        return text;
    }
    String::new()
}

fn parse_records(v: &Value) -> Vec<RecordCriteria> {
    let Some(records) = v.get("records").and_then(Value::as_array) else {
        return Vec::new();
    };
    records
        .iter()
        .map(|r| RecordCriteria {
            status: r
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            criteria: r
                .get("criteria")
                .and_then(Value::as_array)
                .map(|rows| {
                    rows.iter()
                        .filter_map(|c| {
                            Some((
                                c.get("criterionKey").and_then(Value::as_str)?.to_string(),
                                c.get("passed").and_then(Value::as_bool).unwrap_or(false),
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default(),
        })
        .collect()
}

// ---------------------------------------------------------------- task run

struct Readings {
    summary: String,
    attempts: u64,
    spend_micro: u64,
    tokens_total: u64,
    records: Vec<RecordCriteria>,
    record_status: Option<String>,
}

fn collect_readings(client: &Client, session_id: &str) -> Readings {
    let mut attempts = 0u64;
    let mut spend_micro = 0u64;

    if let Ok(turns) = client.get(&format!("/native/session/{session_id}/turns")) {
        attempts = turns.as_array().map(|a| a.len() as u64).unwrap_or(0);
    }
    let mut ledger_spend = None;
    let mut ledger_tokens = None;
    if let Ok(tasks) = client.get(&format!("/native/session/{session_id}/tasks")) {
        if let Some(entry) = tasks.as_array().and_then(|a| a.first()) {
            ledger_spend = find_field(entry, "spentCostMicro", 5).and_then(Value::as_u64);
            ledger_tokens = find_field(entry, "spentTokens", 5).and_then(Value::as_u64);
            if let Some(turns) = find_field(entry, "spentTurns", 5).and_then(Value::as_u64) {
                if turns > attempts {
                    attempts = turns;
                }
            }
        }
    }
    if let Some(s) = ledger_spend {
        spend_micro = s;
    } else if let Ok(usage) = client.get("/native/usage") {
        spend_micro = find_field(&usage, "spent", 6)
            .and_then(Value::as_u64)
            .unwrap_or(0);
    }
    let tokens_total = ledger_tokens.unwrap_or(0);

    let mut records = Vec::new();
    for probe in 1..=MAX_TASK_ID_PROBE {
        let path = format!("/native/session/{session_id}/tasks/{probe}/verification");
        if let Ok(v) = client.get(&path) {
            records.extend(parse_records(&v));
        }
    }
    records.sort_by(|a, b| b.status.cmp(&a.status));
    let record_status = records.first().map(|r| r.status.clone());

    let summary = client
        .get(&format!(
            "/session/messages?session_id={session_id}&limit=100"
        ))
        .map(|v| assistant_summary(&v))
        .unwrap_or_default();

    Readings {
        summary,
        attempts,
        spend_micro,
        tokens_total,
        records,
        record_status,
    }
}

fn native_abort(client: &Client, session_id: &str) {
    let _ = client.post(
        &format!("/native/session/{session_id}/abort"),
        &json!({ "session_id": session_id }),
    );
}

/// Resolve the pending permissions/questions the session raised (the
/// harness is the UI permission channel: deterministic allow). Unknown
/// bodies and races (already-resolved ids) are ignored.
fn auto_allow(client: &Client, session_id: &str) {
    if let Ok(list) = client.get(&format!("/permission/list?session_id={session_id}")) {
        if let Some(items) = list.get("permissions").and_then(Value::as_array) {
            for item in items {
                if let Some(id) = item.get("id").and_then(Value::as_str) {
                    let _ = client.post(
                        "/permission/reply",
                        &json!({ "permission_id": id, "decision": "allow" }),
                    );
                }
            }
        }
    }
    if let Ok(list) = client.get(&format!("/question/list?session_id={session_id}")) {
        if let Some(items) = list.get("questions").and_then(Value::as_array) {
            for item in items {
                if let Some(id) = item.get("id").and_then(Value::as_str) {
                    let _ = client.post(
                        "/question/reply",
                        &json!({ "question_id": id, "decision": "allow" }),
                    );
                }
            }
        }
    }
}

fn turns_terminal(turns: &Value) -> Option<bool> {
    let statuses = turns
        .as_array()
        .map(|rows| {
            rows.iter()
                .filter_map(|t| t.get("status").and_then(Value::as_str).map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if statuses.is_empty() {
        return None;
    }
    Some(
        statuses
            .iter()
            .all(|s| s != "active" && s != "pending" && s != "running"),
    )
}

/// One real task run: fresh daemon + fresh session over a fresh workspace
/// copy. Never touches the checked-in corpus.
fn run_task_real(cfg: &DaemonBenchConfig, bin: &Path, task: &Task) -> Result<TaskResult, String> {
    let workspace = tempfile::tempdir().map_err(|e| e.to_string())?;
    let copy = workspace.path().join("repo");
    fsutil::copy_tree(&task.dir, &copy).map_err(|e| e.to_string())?;

    let data = tempfile::tempdir().map_err(|e| e.to_string())?;
    let config_file = write_config_file(data.path(), cfg)?;

    run_task_real_into(cfg, bin, task, &copy, data.path(), &config_file)
}

/// The inner single-task driver: fresh daemon over `data_dir` +
/// `config_file`, fresh session whose durable workspace root is the
/// caller-provided workspace copy (the corpus path copies first; this
/// seam exists so tests can run the full daemon flow over a workspace of
/// their choosing).
pub fn run_task_real_into(
    cfg: &DaemonBenchConfig,
    bin: &Path,
    task: &Task,
    workspace: &Path,
    data_dir: &Path,
    config_file: &Path,
) -> Result<TaskResult, String> {
    let (mut daemon, client) = DaemonProcess::spawn(bin, data_dir, config_file)?;
    daemon.wait_ready(&client)?;
    let copy = workspace.to_path_buf();

    let t0 = Instant::now();
    let session = client.post(
        "/session/create",
        &json!({
            "provider": pin_id(cfg.provider),
            "model": cfg.model,
            "workspace": copy.display().to_string(),
            "title": format!("bench:{}", task.id),
        }),
    )?;
    let session_id = session
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("session create response without id: {session}"))?
        .to_string();

    // The daemon's own drive entry (SDK form): the session id rides in
    // the body; the prompt path never guesses session identity.
    if let Err(e) = client.post(
        "/session/prompt",
        &json!({
            "session_id": session_id,
            "prompt": build_prompt(task, cfg),
            "files": [],
        }),
    ) {
        native_abort(&client, &session_id);
        return Err(e);
    }

    // Wait for terminal state (bounded); auto-allow permissions along the
    // way so headless runs never stall on the UI permission channel.
    let mut timed_out = false;
    let mut terminal = false;
    let deadline = Instant::now() + cfg.task_timeout;
    while Instant::now() < deadline && !terminal {
        auto_allow(&client, &session_id);
        if let Ok(turns) = client.get(&format!("/native/session/{session_id}/turns")) {
            if turns_terminal(&turns).unwrap_or(false) {
                terminal = true;
                break;
            }
        }
        std::thread::sleep(POLL_INTERVAL);
    }
    if !terminal {
        timed_out = true;
        native_abort(&client, &session_id);
        // Let the abort land so the readings see final durable rows.
        std::thread::sleep(Duration::from_millis(500));
    }
    let wall_ms = t0.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;

    let readings = collect_readings(&client, &session_id);
    let daemon_diag = daemon.stderr_tail();
    drop(daemon);

    let verify = run_verify_sh(&copy, cfg.verify);
    let verified = verify.verified();
    let criteria = corroborate_records(
        score_summary(&task.criteria, &readings.summary),
        &task.criteria,
        &readings.records,
    );
    let mut result = TaskResult {
        task_id: task.id.clone(),
        lang: task.lang.to_string(),
        skipped: None,
        verified,
        wall_ms,
        timed_out,
        attempts: readings.attempts,
        spend_micro: readings.spend_micro,
        tokens_total: readings.tokens_total,
        criteria: criteria.clone(),
        criteria_met: criteria_met(&criteria),
        criteria_total: task.criteria.len(),
        record_status: readings.record_status,
        verify: Some(verify),
        note: None,
    };
    if timed_out {
        result.note = Some(format!(
            "wall budget {}s exhausted; aborted via the native abort endpoint",
            cfg.task_timeout.as_secs()
        ));
    }
    if std::env::var("FAKTOR_BENCH_DEBUG")
        .map(|v| v != "0")
        .unwrap_or(false)
    {
        let note = result.note.get_or_insert_with(String::new);
        if !note.is_empty() {
            note.push_str(" | ");
        }
        note.push_str("daemon stderr tail: ");
        note.push_str(&daemon_diag);
    }
    Ok(result)
}

// ---------------------------------------------------------------- corpus run

/// Run the whole corpus against the real daemon. Corpus-level skips
/// (missing binary / provider env) become skip rows with documented
/// notes; per-task toolchain gaps stay per-task skips.
pub fn run_corpus_real(
    cfg: &DaemonBenchConfig,
    corpus: &Corpus,
    compile_hint: Option<&str>,
) -> CorpusReport {
    let mut notes = Vec::new();
    if !cfg.tag.is_empty() {
        notes.push(format!("tag: {}", cfg.tag));
    }
    let bin = match resolve_binary(compile_hint) {
        Some(b) => b,
        None => {
            notes.push(
                "daemon binary not found: build it (cargo build -p faktor-cli) or point \
                 FAKTOR_BENCH_BIN at one; no task can run in subprocess mode"
                    .into(),
            );
            let results = corpus
                .tasks
                .iter()
                .map(|t| {
                    TaskResult::skipped(&t.id, t.lang, "no daemon binary (see report notes)".into())
                })
                .collect();
            let mut report = CorpusReport::new(results);
            report.notes = notes;
            return report;
        }
    };
    notes.push(format!("daemon binary: {}", bin.display()));

    let only = std::env::var(DaemonBenchConfig::ENV_ONLY).unwrap_or_default();
    let mut results = Vec::new();
    let mut hard_failure: Option<String> = None;
    for task in &corpus.tasks {
        if !only.is_empty() && task.id != only {
            results.push(TaskResult::skipped(
                &task.id,
                task.lang,
                format!("filtered out ({}={only})", DaemonBenchConfig::ENV_ONLY),
            ));
            continue;
        }
        if let Some(reason) = &hard_failure {
            results.push(TaskResult::skipped(
                &task.id,
                task.lang,
                format!("daemon failure on an earlier task ({reason}); run aborted"),
            ));
            continue;
        }
        if let Err(reason) = require_toolchain(task.lang) {
            results.push(TaskResult::skipped(&task.id, task.lang, reason));
            continue;
        }
        match run_task_real(cfg, &bin, task) {
            Ok(row) => results.push(row),
            Err(e) => {
                hard_failure = Some(e.clone());
                results.push(TaskResult::skipped(
                    &task.id,
                    task.lang,
                    format!("daemon run failed: {e}"),
                ));
            }
        }
    }
    let mut report = CorpusReport::new(results);
    report.notes = notes;
    report
}

// ---------------------------------------------------------------- config file

/// The registered id of the daemon's configured provider instance.
/// Non-ollama kinds register wrapped with the configured instance id
/// (`bench`); the daemon's ollama warm-up registers the CONCRETE provider,
/// whose own id is `ollama` — the pin must match the registered name.
fn pin_id(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Ollama => "ollama",
        _ => "bench",
    }
}

/// `http://host:1234/v1/path` → `http://host:1234` (destination rules
/// carry no path). Malformed URLs pass through unchanged so the daemon's
/// strict rule parser produces the real error.
fn origin_of(url: &str) -> String {
    match url.find("://") {
        Some(scheme_end) => {
            let rest = &url[scheme_end + 3..];
            match rest.find('/') {
                Some(slash) => format!("{}{}", &url[..scheme_end + 3], &rest[..slash]),
                None => url.to_string(),
            }
        }
        None => url.to_string(),
    }
}

/// Write the daemon's strict JSON config file into `data_dir` and return
/// its path. Exposed for tests: pure construction, no daemon involved.
pub fn write_config_file(data_dir: &Path, cfg: &DaemonBenchConfig) -> Result<PathBuf, String> {
    let providers = match cfg.provider {
        ProviderKind::Anthropic => json!([{
            "kind": "anthropic",
            "id": "bench",
            "api_key_env": cfg.api_key_env,
        }]),
        ProviderKind::OpenAi => json!([{
            "kind": "open_ai",
            "id": "bench",
            "base_url": cfg.base_url.as_deref().unwrap_or("https://api.openai.com/v1"),
            "api_key_env": cfg.api_key_env,
        }]),
        ProviderKind::Google => json!([{
            "kind": "google",
            "id": "bench",
            "api_key_env": cfg.api_key_env,
        }]),
        ProviderKind::DeepSeek => {
            let mut p = json!({
                "kind": "deepseek",
                "id": "bench",
                "profile": "direct",
                "api_key_env": cfg.api_key_env,
            });
            if let Some(url) = &cfg.base_url {
                p["base_url"] = json!(url);
            }
            p
        }
        ProviderKind::Gateway => json!([{
            "kind": "gateway",
            "id": "bench",
            "base_url": cfg.base_url.as_deref().unwrap_or("https://api.kilo.ai"),
            "api_key_env": cfg.api_key_env,
        }]),
        ProviderKind::Ollama => json!([{
            "kind": "ollama",
            "id": "bench",
            "base_url": cfg.base_url.as_deref().unwrap_or("http://127.0.0.1:11434"),
        }]),
    };
    let mut config = json!({
        "config_version": 1,
        "model": cfg.model,
        "instructions":
            "You are Faktor. Act as a careful senior engineer inside the user's repository. \
             Work only inside the session workspace.",
        "providers": providers,
        "routing_mode": { "pinned": { "provider": pin_id(cfg.provider), "model": cfg.model } },
        "mcp": [],
        "verification": { "quick_max_s": 60, "unit_max_s": 600, "full_as_background": true },
        "tasks": { "shadow_mutation": false },
    });
    if cfg.base_url.is_some() || !cfg.network_rows.is_empty() {
        let mut rows = cfg.network_rows.clone();
        if let Some(url) = &cfg.base_url {
            // Destination rules carry no path components
            // (scheme://host[:port]); the adapter base_url may include a
            // path prefix such as /v1.
            rows.push(origin_of(url));
        }
        config["sandbox"] = json!({ "network": rows, "network_guarantee": "none" });
    }
    let path = data_dir.join("daemon-config.json");
    let text = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    std::fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

// ---------------------------------------------------------------- planning

/// Deterministic skip-vs-run plan for one corpus (used by the normal-mode
/// tests: every selection/skip decision must be reachable without a
/// daemon or provider keys).
pub fn plan_corpus(
    cfg: Option<&DaemonBenchConfig>,
    corpus: &Corpus,
    compile_hint: Option<&str>,
) -> Vec<(String, Result<(), String>)> {
    corpus
        .tasks
        .iter()
        .map(|t| {
            let gate = match cfg {
                None => Err(
                    "no real-model configuration (FAKTOR_BENCH_PROVIDER/MODEL/API_KEY unset); \
                     set the documented env vars or run the fake no-model harness"
                        .into(),
                ),
                Some(cfg) => {
                    if resolve_binary(compile_hint).is_none() {
                        Err("no daemon binary (build `cargo build -p faktor-cli` or set FAKTOR_BENCH_BIN)".into())
                    } else if cfg.provider.needs_key()
                        && std::env::var(&cfg.api_key_env).map_or(true, |v| v.trim().is_empty())
                    {
                        Err(format!(
                            "provider key missing ({} unset); set {} or {}",
                            cfg.api_key_env,
                            DaemonBenchConfig::ENV_API_KEY_ENV,
                            DaemonBenchConfig::ENV_API_KEY
                        ))
                    } else {
                        require_toolchain(t.lang)
                    }
                }
            };
            (t.id.clone(), gate)
        })
        .collect()
}
