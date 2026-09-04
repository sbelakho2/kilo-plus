//! Deterministic lifecycle hooks (audit: Copilot-style hook lifecycle).
//!
//! External process hooks over the operation lifecycle. A hook MAY allow,
//! deny, warn, or return modified metadata — it can never silently mutate
//! core agent state (callers apply verdicts as commands/events).
//!
//! Every hook runs with: a deadline that DOMINATES (on expiry the owned
//! process tree is killed and any partial output is discarded for verdict
//! purposes — the failure policy decides), a cleared environment built from
//! an explicit allowlist (with a documented benign passthrough set when
//! `env_allowlist` is false), bounded streaming output reads, and every run
//! is appended to an audit log that carries the bounded head only.

use std::collections::VecDeque;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookEvent {
    SessionStart,
    SessionResume,
    TaskStart,
    PreModel,
    PostModel,
    PreTool,
    PostTool,
    ToolError,
    PreEdit,
    PostEdit,
    PreCommit,
    SubagentStart,
    SubagentStop,
    AgentError,
    AgentStop,
    TaskComplete,
    SessionEnd,
}

impl HookEvent {
    pub const ALL: [HookEvent; 17] = [
        HookEvent::SessionStart,
        HookEvent::SessionResume,
        HookEvent::TaskStart,
        HookEvent::PreModel,
        HookEvent::PostModel,
        HookEvent::PreTool,
        HookEvent::PostTool,
        HookEvent::ToolError,
        HookEvent::PreEdit,
        HookEvent::PostEdit,
        HookEvent::PreCommit,
        HookEvent::SubagentStart,
        HookEvent::SubagentStop,
        HookEvent::AgentError,
        HookEvent::AgentStop,
        HookEvent::TaskComplete,
        HookEvent::SessionEnd,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailurePolicy {
    /// Hook crash/timeout/deny denies the operation.
    FailClosed,
    /// Hook failure logs a warn and proceeds.
    FailOpen,
    /// Hook failure surfaces a warning only.
    Warn,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HookSpec {
    pub id: String,
    pub events: Vec<HookEvent>,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Explicit (key, value) env entries; an empty value means "inherit
    /// from the daemon env" (the daemon's own value for that key). Explicit
    /// entries pass in BOTH modes below.
    #[serde(default)]
    pub env: Vec<(String, String)>,
    /// true (the DEFAULT): the child env is `env_clear`ed and the hook sees
    /// ONLY the explicit `env` entries plus `FAKTOR_HOOK_INPUT`. false: the
    /// child additionally sees a FIXED benign passthrough set (HOME, PATH,
    /// LANG, LC_ALL, TZ, TERM, USER, SHELL when set in the daemon env).
    /// Either way the base is a cleared env: arbitrary daemon environment
    /// (secrets included) never reaches a hook implicitly.
    pub env_allowlist: bool,
    pub deadline_ms: u64,
    pub stdout_cap: usize,
    pub stderr_cap: usize,
    pub failure_policy: FailurePolicy,
    pub permission_scope: String,
}

impl Default for HookSpec {
    fn default() -> Self {
        Self {
            id: String::new(),
            events: vec![],
            command: String::new(),
            args: vec![],
            env: vec![],
            env_allowlist: true,
            deadline_ms: 5000,
            stdout_cap: 64 * 1024,
            stderr_cap: 64 * 1024,
            failure_policy: FailurePolicy::FailClosed,
            permission_scope: String::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HookInput {
    pub event: HookEvent,
    pub session_id: Option<String>,
    pub task_id: Option<String>,
    pub operation_id: Option<String>,
    pub payload: serde_json::Value,
}

impl Default for HookInput {
    fn default() -> Self {
        Self {
            event: HookEvent::PreTool,
            session_id: None,
            task_id: None,
            operation_id: None,
            payload: serde_json::Value::Null,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum HookVerdict {
    Allow,
    Deny { reason: String },
    Warn { reason: String },
    Modify { metadata: serde_json::Value },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HookAuditRecord {
    pub hook_id: String,
    pub event: HookEvent,
    pub started_ms: i64,
    pub duration_ms: u64,
    pub verdict: String,
    pub exit_code: Option<i32>,
    pub stdout_head: String,
    pub stderr_head: String,
    pub failure_policy: FailurePolicy,
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Benign env passthrough for `env_allowlist: false` hooks (audit):
/// locale/timezone/path/terminal basics only — never credentials or daemon
/// state. With `env_allowlist: true` (the default) NOTHING beyond the
/// explicit `env` entries and `FAKTOR_HOOK_INPUT` reaches the hook.
const BENIGN_ENV_PASSTHROUGH: [&str; 8] = [
    "HOME", "PATH", "LANG", "LC_ALL", "TZ", "TERM", "USER", "SHELL",
];

/// Stream a child pipe into a bounded head: read incrementally, keep at
/// most `cap` bytes, then DRAIN the remainder in fixed chunks so a hostile
/// 10 MB / infinite producer never grows memory past the cap (the child is
/// only ever blocked briefly per kernel pipe buffer, never on us). Returns
/// (lossy head, truncated?) where `truncated` means more bytes arrived than
/// the cap could hold.
fn read_bounded_head(pipe: Box<dyn Read + Send>, cap: usize) -> (String, bool) {
    const CHUNK: usize = 8192;
    let mut head: Vec<u8> = Vec::with_capacity(cap.min(CHUNK));
    let mut scratch = [0u8; CHUNK];
    let mut truncated = false;
    let mut pipe = pipe;
    loop {
        let n = match pipe.read(&mut scratch) {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        if head.len() < cap {
            let take = (cap - head.len()).min(n);
            head.extend_from_slice(&scratch[..take]);
            if take < n {
                truncated = true;
            }
        } else {
            truncated = true;
        }
    }
    (String::from_utf8_lossy(&head).into_owned(), truncated)
}

/// The audit surface for one stream: the bounded head plus an ellipsis when
/// the stream was truncated. Never unbounded — `head` is already capped.
fn audit_head(head: &str, truncated: bool) -> String {
    if truncated {
        let mut s = head.to_string();
        s.push('…');
        s
    } else {
        head.to_string()
    }
}

fn verdict_tag(v: &HookVerdict) -> String {
    match v {
        HookVerdict::Allow => "allow".into(),
        HookVerdict::Deny { .. } => "deny".into(),
        HookVerdict::Warn { .. } => "warn".into(),
        HookVerdict::Modify { .. } => "modify".into(),
    }
}

/// Parse the first `{"verdict":...}` object line (bounded).
fn parse_verdict(out: &str) -> Option<HookVerdict> {
    for line in out.lines().take(64) {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<HookVerdict>(line) {
            return Some(v);
        }
    }
    None
}

#[derive(Clone)]
struct Inner {
    specs: Arc<Mutex<Vec<HookSpec>>>,
    audit: Arc<Mutex<VecDeque<HookAuditRecord>>>,
}

pub struct HookRegistry {
    inner: Inner,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self {
            inner: Inner {
                specs: Arc::new(Mutex::new(Vec::new())),
                audit: Arc::new(Mutex::new(VecDeque::new())),
            },
        }
    }

    pub fn register(&self, spec: HookSpec) -> Result<(), String> {
        if spec.id.is_empty() || spec.id.len() > 128 {
            return Err("hook id must be 1..=128 bytes".into());
        }
        if spec.command.is_empty() || spec.command.len() > 4096 {
            return Err("hook command must be 1..=4096 bytes".into());
        }
        if spec.deadline_ms == 0 || spec.deadline_ms > 300_000 {
            return Err("hook deadline must be in (0, 300000] ms".into());
        }
        let mut specs = self.inner.specs.lock().unwrap();
        if specs.iter().any(|s| s.id == spec.id) {
            return Err(format!("duplicate hook id {}", spec.id));
        }
        specs.push(spec);
        Ok(())
    }

    pub fn matching(&self, event: HookEvent) -> Vec<HookSpec> {
        self.inner
            .specs
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.events.contains(&event))
            .cloned()
            .collect()
    }

    fn audit_push(&self, rec: HookAuditRecord) {
        let mut a = self.inner.audit.lock().unwrap();
        a.push_back(rec);
        while a.len() > 4096 {
            a.pop_front();
        }
    }

    /// Run every hook registered for the event, in registration order.
    /// Any Deny wins; else any Warn produces a Warn; else the last Modify;
    /// else Allow.
    pub fn run(&self, event: HookEvent, input: &HookInput) -> HookVerdict {
        let mut out = HookVerdict::Allow;
        for spec in self.matching(event) {
            let verdict = self.run_one(&spec, input);
            match verdict {
                HookVerdict::Deny { .. } => return verdict,
                HookVerdict::Warn { .. } => {
                    if matches!(out, HookVerdict::Allow) {
                        out = verdict;
                    }
                }
                HookVerdict::Modify { .. } => out = verdict,
                HookVerdict::Allow => {}
            }
        }
        out
    }

    /// Execute ONE hook synchronously with deadline/caps/policy.
    pub fn run_one(&self, spec: &HookSpec, input: &HookInput) -> HookVerdict {
        let started = std::time::Instant::now();
        let input_json = serde_json::json!({
            "event": input.event,
            "session_id": input.session_id,
            "task_id": input.task_id,
            "operation_id": input.operation_id,
            "payload": input.payload,
        });
        let mut cmd = Command::new(&spec.command);
        cmd.args(&spec.args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0); // own group: kill(-pid) reaches the tree
        }
        // Env is ALWAYS built from an env_clear base: hooks never inherit
        // the daemon env implicitly. env_allowlist (default true) passes
        // only the explicit `env` entries; false adds the fixed benign
        // passthrough set. Secrets reach a hook only when a config lists
        // their key explicitly.
        cmd.env_clear();
        for (k, v) in &spec.env {
            if v.is_empty() {
                if let Ok(cur) = std::env::var(k) {
                    cmd.env(k, cur);
                }
            } else {
                cmd.env(k, v);
            }
        }
        if !spec.env_allowlist {
            for k in BENIGN_ENV_PASSTHROUGH {
                if let Ok(cur) = std::env::var(k) {
                    cmd.env(k, cur);
                }
            }
        }
        // Input JSON rides the env (bounded by callers; callers cap it at
        // 64 KiB); stdin stays null so hooks can never feed us input.
        cmd.env("FAKTOR_HOOK_INPUT", input_json.to_string());
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return self.policy_outcome(
                    spec,
                    HookVerdict::Warn {
                        reason: format!("spawn: {e}"),
                    },
                )
            }
        };
        let pid = child.id();
        let deadline_ms = spec.deadline_ms;
        // Dedicated reader threads: reads can never block the caller past
        // the deadline. Each thread reads BOUNDED (head up to the spec cap,
        // remainder drained in chunks) and hands the head back through a
        // channel — the caller never joins unboundedly.
        let out_cap = spec.stdout_cap;
        let err_cap = spec.stderr_cap;
        let out_pipe = child.stdout.take();
        let err_pipe = child.stderr.take();
        let (out_tx, out_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let res = match out_pipe {
                Some(p) => read_bounded_head(Box::new(p), out_cap),
                None => (String::new(), false),
            };
            let _ = out_tx.send(res);
        });
        let (err_tx, err_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let res = match err_pipe {
                Some(p) => read_bounded_head(Box::new(p), err_cap),
                None => (String::new(), false),
            };
            let _ = err_tx.send(res);
        });
        // The waiter owns reaping; the caller enforces the deadline: if no
        // exit arrives within deadline_ms the OWNED tree is killed and the
        // partial output is DISCARDED for verdict purposes (see below). The
        // reaped flag guards the kill: a reaped pid is never signalled (a
        // recycled group must not die for our deadline).
        let reaped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        {
            let reaped = reaped.clone();
            std::thread::spawn(move || {
                let code = child.wait().ok().and_then(|s| s.code());
                reaped.store(true, std::sync::atomic::Ordering::SeqCst);
                let _ = tx.send(code);
            });
        }
        let (exit, timed_out) = match rx.recv_timeout(std::time::Duration::from_millis(deadline_ms))
        {
            Ok(code) => (code, false),
            Err(_) => {
                // Deadline fired: kill the group (only while the child is
                // still ours), then give the killer a moment to reap (never
                // block long past the deadline).
                if !reaped.load(std::sync::atomic::Ordering::SeqCst) {
                    kill_tree(pid);
                }
                let _ = rx.recv_timeout(std::time::Duration::from_millis(500));
                (None, true)
            }
        };
        // Bounded settle: the readers finish at pipe EOF; a grandchild that
        // inherited the pipe may delay them — never the caller. Output that
        // does not arrive within the settle bound is dropped (bounded).
        const SETTLE_MS: u64 = 1000;
        let settle = std::time::Duration::from_millis(SETTLE_MS);
        let (stdout, stdout_truncated) = out_rx
            .recv_timeout(settle)
            .unwrap_or((String::new(), false));
        let (stderr, stderr_truncated) = err_rx
            .recv_timeout(settle)
            .unwrap_or((String::new(), false));
        let duration = started.elapsed().as_millis() as u64;
        // The deadline dominates: a timed-out run's partial output NEVER
        // decides — the verdict is the failure policy's outcome. Only a run
        // that exited on its own within the deadline may have its stdout
        // parsed.
        let outcome = if timed_out {
            self.policy_outcome(
                spec,
                HookVerdict::Warn {
                    reason: format!(
                        "hook {} exceeded the {deadline_ms} ms deadline and was killed",
                        spec.id
                    ),
                },
            )
        } else {
            match parse_verdict(&stdout) {
                Some(v) => v,
                None => {
                    if exit != Some(0) {
                        self.policy_outcome(
                            spec,
                            HookVerdict::Warn {
                                reason: format!(
                                    "hook {} exited {exit:?} without a verdict",
                                    spec.id
                                ),
                            },
                        )
                    } else {
                        HookVerdict::Allow
                    }
                }
            }
        };
        // The audit record carries the bounded head only, plus the outcome
        // actually applied (a timeout's partial stdout is visible forensics
        // but never a verdict).
        self.audit_push(HookAuditRecord {
            hook_id: spec.id.clone(),
            event: input.event,
            started_ms: now_ms() - duration as i64,
            duration_ms: duration,
            verdict: verdict_tag(&outcome),
            exit_code: exit,
            stdout_head: audit_head(&stdout, stdout_truncated),
            stderr_head: audit_head(&stderr, stderr_truncated),
            failure_policy: spec.failure_policy,
        });
        outcome
    }

    fn policy_outcome(&self, spec: &HookSpec, failure: HookVerdict) -> HookVerdict {
        match spec.failure_policy {
            FailurePolicy::FailClosed => HookVerdict::Deny {
                reason: format!("hook {} failed closed", spec.id),
            },
            FailurePolicy::FailOpen | FailurePolicy::Warn => HookVerdict::Warn {
                reason: format!("hook {} failed open: {failure:?}", spec.id),
            },
        }
    }

    pub fn audit(&self) -> Vec<HookAuditRecord> {
        self.inner.audit.lock().unwrap().iter().cloned().collect()
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn kill_tree(pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
        libc::kill(pid as i32, libc::SIGKILL);
    }
    #[cfg(not(unix))]
    {
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_path_via_json_stdout() {
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "ok".into(),
            command: "sh".into(),
            args: vec!["-c".into(), "echo '{\"verdict\":\"allow\"}'".into()],
            events: vec![HookEvent::PreTool],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            r.run(HookEvent::PreTool, &HookInput::default()),
            HookVerdict::Allow
        );
    }

    #[test]
    fn deny_path_blocks() {
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "no".into(),
            command: "sh".into(),
            args: vec![
                "-c".into(),
                "echo '{\"verdict\":\"deny\",\"reason\":\"policy\"}'".into(),
            ],
            events: vec![HookEvent::PreTool],
            ..Default::default()
        })
        .unwrap();
        match r.run(HookEvent::PreTool, &HookInput::default()) {
            HookVerdict::Deny { reason } => assert_eq!(reason, "policy"),
            v => panic!("expected deny, got {v:?}"),
        }
    }

    #[test]
    fn fail_closed_on_crash_without_verdict() {
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "crash".into(),
            command: "sh".into(),
            args: vec!["-c".into(), "exit 3".into()],
            events: vec![HookEvent::PreEdit],
            failure_policy: FailurePolicy::FailClosed,
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(
            r.run(HookEvent::PreEdit, &HookInput::default()),
            HookVerdict::Deny { .. }
        ));
    }

    #[test]
    fn fail_open_proceeds_with_warn_audit() {
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "soft".into(),
            command: "sh".into(),
            args: vec!["-c".into(), "exit 9".into()],
            events: vec![HookEvent::PreTool],
            failure_policy: FailurePolicy::FailOpen,
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(
            r.run(HookEvent::PreTool, &HookInput::default()),
            HookVerdict::Warn { .. }
        ));
        assert!(!r.audit().is_empty());
    }

    #[test]
    fn timeout_kills_and_policy_decides() {
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "slow".into(),
            command: "sh".into(),
            args: vec![
                "-c".into(),
                "sleep 10; echo '{\"verdict\":\"allow\"}'".into(),
            ],
            events: vec![HookEvent::PreModel],
            deadline_ms: 300,
            failure_policy: FailurePolicy::FailClosed,
            ..Default::default()
        })
        .unwrap();
        let t0 = std::time::Instant::now();
        let v = r.run(HookEvent::PreModel, &HookInput::default());
        assert!(
            t0.elapsed().as_millis() < 5000,
            "deadline must bound the hook"
        );
        assert!(
            matches!(v, HookVerdict::Deny { .. }),
            "timeout fails closed: {v:?}"
        );
    }

    #[test]
    fn env_clear_by_default_removes_secrets() {
        // env_allowlist DEFAULTS to true: the child env is env_clear'ed, so
        // a secret in the daemon env never reaches a hook whose spec lists
        // nothing — even when the spec never opted into an allowlist.
        std::env::set_var("FAKTOR_TOKEN", "sekrit");
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "env".into(),
            command: "sh".into(),
            args: vec!["-c".into(),
                "test -n \"$FAKTOR_TOKEN\" && echo '{\"verdict\":\"deny\",\"reason\":\"leak\"}' || echo '{\"verdict\":\"allow\"}'".into()],
            events: vec![HookEvent::PostTool],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            r.run(HookEvent::PostTool, &HookInput::default()),
            HookVerdict::Allow,
            "the unlisted secret must not reach the hook"
        );
        std::env::remove_var("FAKTOR_TOKEN");
    }

    #[test]
    fn explicit_env_entries_pass_under_the_allowlist() {
        // allowlist=true still passes EXPLICIT entries: listed keys are the
        // config's deliberate choice.
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "env".into(),
            command: "sh".into(),
            args: vec!["-c".into(),
                "test \"$ALLOWED\" = 1 && echo '{\"verdict\":\"allow\"}' || echo '{\"verdict\":\"deny\",\"reason\":\"missing\"}'".into()],
            events: vec![HookEvent::PreTool],
            env_allowlist: true,
            env: vec![("ALLOWED".into(), "1".into())],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            r.run(HookEvent::PreTool, &HookInput::default()),
            HookVerdict::Allow
        );
    }

    #[test]
    fn allowlist_off_passes_only_the_benign_set() {
        // env_allowlist=false is NOT a full passthrough: only the fixed
        // benign set (HOME/PATH/...) plus explicit entries reaches the hook
        // — the daemon's secret is stripped in BOTH modes.
        std::env::set_var("FAKTOR_TOKEN", "sekrit");
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "env".into(),
            command: "sh".into(),
            args: vec!["-c".into(),
                "test -z \"$FAKTOR_TOKEN\" && test -n \"$PATH\" && test -n \"$HOME\" && echo '{\"verdict\":\"allow\"}' || echo '{\"verdict\":\"deny\",\"reason\":\"unexpected-env\"}'".into()],
            events: vec![HookEvent::PreModel],
            env_allowlist: false,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            r.run(HookEvent::PreModel, &HookInput::default()),
            HookVerdict::Allow,
            "benign vars pass, the secret never does"
        );
        std::env::remove_var("FAKTOR_TOKEN");
    }

    #[test]
    fn ten_megabyte_stream_stays_bounded_and_the_hook_completes() {
        // A hook that floods 10 MB: reads are incremental into a bounded
        // head (nothing past stdout_cap is retained) and the remainder is
        // drained, so the hook completes instead of deadlocking on a full
        // pipe. Peak memory is not asserted; the audit head must be capped.
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "noisy".into(),
            command: "sh".into(),
            args: vec![
                "-c".into(),
                "dd if=/dev/zero bs=1048576 count=10 2>/dev/null".into(),
            ],
            events: vec![HookEvent::PreTool],
            stdout_cap: 4096,
            stderr_cap: 4096,
            ..Default::default()
        })
        .unwrap();
        let t0 = std::time::Instant::now();
        // exit 0 with no verdict line -> Allow (the run completed within the
        // default deadline; it must not be mistaken for a hang).
        assert_eq!(
            r.run(HookEvent::PreTool, &HookInput::default()),
            HookVerdict::Allow
        );
        assert!(
            t0.elapsed().as_millis() < 10_000,
            "10 MB through a 64 KiB pipe must drain, not stall"
        );
        let audit = r.audit();
        assert_eq!(audit.len(), 1);
        let head = &audit[0].stdout_head;
        assert!(
            head.len() <= 4096 + 3,
            "audit stdout head is bounded (cap + '…' marker), got {} bytes",
            head.len()
        );
    }

    #[test]
    fn deadline_discards_partial_verdict_output() {
        // The hook writes a VALID allow verdict immediately, then runs far
        // past the deadline. The deadline DOMINATES: the tree is killed and
        // that partial stdout is discarded for verdict purposes — the
        // outcome is the failure policy's (FailClosed -> Deny), never the
        // parsed Allow. The partial head still lands in the audit record.
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "liar".into(),
            command: "sh".into(),
            args: vec![
                "-c".into(),
                "echo '{\"verdict\":\"allow\"}'; sleep 10".into(),
            ],
            events: vec![HookEvent::PreModel],
            deadline_ms: 200,
            failure_policy: FailurePolicy::FailClosed,
            ..Default::default()
        })
        .unwrap();
        let t0 = std::time::Instant::now();
        match r.run(HookEvent::PreModel, &HookInput::default()) {
            HookVerdict::Deny { .. } => {}
            v => panic!("timeout must fail closed, partial allow discarded: {v:?}"),
        }
        assert!(
            t0.elapsed().as_millis() < 5000,
            "deadline must bound the run"
        );
        let audit = r.audit();
        assert_eq!(audit.len(), 1);
        assert_eq!(
            audit[0].verdict, "deny",
            "audit carries the applied outcome"
        );
        assert_eq!(audit[0].exit_code, None, "the tree was killed, not exited");
        assert!(
            audit[0].stdout_head.contains("allow"),
            "partial stdout is audited as forensics, never as a verdict: {:?}",
            audit[0].stdout_head
        );

        // Same partial output under FailOpen: the policy outcome is Warn.
        let r2 = HookRegistry::new();
        r2.register(HookSpec {
            id: "liar-open".into(),
            command: "sh".into(),
            args: vec![
                "-c".into(),
                "echo '{\"verdict\":\"allow\"}'; sleep 10".into(),
            ],
            events: vec![HookEvent::PreTool],
            deadline_ms: 200,
            failure_policy: FailurePolicy::FailOpen,
            ..Default::default()
        })
        .unwrap();
        assert!(matches!(
            r2.run(HookEvent::PreTool, &HookInput::default()),
            HookVerdict::Warn { .. }
        ));
    }

    #[test]
    fn verdict_precedence_deny_wins() {
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "a".into(),
            command: "sh".into(),
            args: vec!["-c".into(), "echo '{\"verdict\":\"allow\"}'".into()],
            events: vec![HookEvent::PreTool],
            ..Default::default()
        })
        .unwrap();
        r.register(HookSpec {
            id: "b".into(),
            command: "sh".into(),
            args: vec![
                "-c".into(),
                "echo '{\"verdict\":\"deny\",\"reason\":\"stop\"}'".into(),
            ],
            events: vec![HookEvent::PreTool],
            ..Default::default()
        })
        .unwrap();
        match r.run(HookEvent::PreTool, &HookInput::default()) {
            HookVerdict::Deny { reason } => assert_eq!(reason, "stop"),
            v => panic!("deny must win: {v:?}"),
        }
    }

    #[test]
    fn modify_metadata_passthrough_and_audit_completeness() {
        let r = HookRegistry::new();
        r.register(HookSpec {
            id: "m".into(),
            command: "sh".into(),
            args: vec![
                "-c".into(),
                "echo '{\"verdict\":\"modify\",\"metadata\":{\"note\":\"hi\"}}'".into(),
            ],
            events: vec![HookEvent::PreModel],
            ..Default::default()
        })
        .unwrap();
        match r.run(HookEvent::PreModel, &HookInput::default()) {
            HookVerdict::Modify { metadata } => assert_eq!(metadata["note"], "hi"),
            v => panic!("expected modify: {v:?}"),
        }
        let audit = r.audit();
        assert_eq!(audit.len(), 1);
        assert_eq!(audit[0].hook_id, "m");
        assert!(audit[0].duration_ms > 0);
    }

    #[test]
    fn hostile_registrations_rejected() {
        let r = HookRegistry::new();
        assert!(r
            .register(HookSpec {
                id: String::new(),
                ..Default::default()
            })
            .is_err());
        assert!(r
            .register(HookSpec {
                id: "d".into(),
                command: "x".into(),
                deadline_ms: 0,
                ..Default::default()
            })
            .is_err());
        let ok = HookSpec {
            id: "d".into(),
            command: "true".into(),
            ..Default::default()
        };
        r.register(ok.clone()).unwrap();
        assert!(r.register(ok).is_err(), "duplicate id rejected");
    }
}
