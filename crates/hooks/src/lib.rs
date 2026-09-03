//! Deterministic lifecycle hooks (audit: Copilot-style hook lifecycle).
//!
//! External process hooks over the operation lifecycle. A hook MAY allow,
//! deny, warn, or return modified metadata — it can never silently mutate
//! core agent state (callers apply verdicts as commands/events).
//!
//! Every hook runs with: a deadline, an env allowlist, output caps, a
//! failure policy, and every run is appended to an audit log.

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
    /// (key, value); empty value means "inherit from context env".
    #[serde(default)]
    pub env: Vec<(String, String)>,
    /// true: only keys in `env` pass through.
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
            env_allowlist: false,
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

fn capped(mut s: String, cap: usize) -> String {
    if s.len() > cap {
        s.truncate(cap);
        s.push('…');
    }
    s
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
        if spec.env_allowlist {
            cmd.env_clear();
        }
        for (k, v) in &spec.env {
            if v.is_empty() {
                if let Ok(cur) = std::env::var(k) {
                    cmd.env(k, cur);
                }
            } else {
                cmd.env(k, v);
            }
        }
        // Input JSON via a temp file-free approach: env FAKTOR_HOOK_INPUT is
        // unbounded env; prefer stdin — we set stdin null above; switch to
        // piped and write the payload (bounded by callers; cap 64 KiB).
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
        let deadline = spec.deadline_ms;
        // Dedicated reader threads: reads can never block the caller past
        // the deadline (bounded join below).
        let out_pipe = child.stdout.take();
        let err_pipe = child.stderr.take();
        let out_t = std::thread::spawn(move || {
            let mut s = String::new();
            if let Some(mut o) = out_pipe {
                let _ = o.read_to_string(&mut s);
            }
            s
        });
        let err_t = std::thread::spawn(move || {
            let mut s = String::new();
            if let Some(mut e) = err_pipe {
                let _ = e.read_to_string(&mut s);
            }
            s
        });
        // Watchdog: kill the OWNED tree after the deadline, but only while
        // the child is still ours (a reaped pid is never signalled — that
        // would risk killing an unrelated recycled group).
        let reaped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        {
            let reaped = reaped.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(deadline));
                if !reaped.load(std::sync::atomic::Ordering::SeqCst) {
                    kill_tree(pid);
                }
            });
        }
        // The waiter owns reaping; a channel bounds the caller's wait.
        let (tx, rx) = std::sync::mpsc::channel();
        {
            let reaped = reaped.clone();
            std::thread::spawn(move || {
                let code = child.wait().ok().and_then(|s| s.code());
                reaped.store(true, std::sync::atomic::Ordering::SeqCst);
                let _ = tx.send(code);
            });
        }
        let bound = std::time::Duration::from_millis(deadline + 2000);
        let exit = rx.recv_timeout(bound).ok().flatten().or_else(|| {
            kill_tree(pid);
            // Give the killer a moment, then give up (never block long).
            let _ = rx.recv_timeout(std::time::Duration::from_millis(500));
            None
        });
        let stdout = out_t.join().unwrap_or_default();
        let stderr = err_t.join().unwrap_or_default();
        let duration = started.elapsed().as_millis() as u64;
        self.audit_push(HookAuditRecord {
            hook_id: spec.id.clone(),
            event: input.event,
            started_ms: now_ms() - duration as i64,
            duration_ms: duration,
            verdict: "running".into(),
            exit_code: exit,
            stdout_head: capped(stdout.clone(), spec.stdout_cap),
            stderr_head: capped(stderr.clone(), spec.stderr_cap),
            failure_policy: spec.failure_policy,
        });
        match parse_verdict(&stdout) {
            Some(v) => v,
            None => {
                let code = exit;
                if code != Some(0) {
                    self.policy_outcome(
                        spec,
                        HookVerdict::Warn {
                            reason: format!("hook {} exited {:?} without a verdict", spec.id, code),
                        },
                    )
                } else {
                    HookVerdict::Allow
                }
            }
        }
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
    fn env_allowlist_strips_secrets() {
        let r = HookRegistry::new();
        let mut spec = HookSpec {
            id: "env".into(),
            command: "sh".into(),
            args: vec!["-c".into(), "test -z \"$FAKTOR_TOKEN\" && echo '{\"verdict\":\"allow\"}' || echo '{\"verdict\":\"deny\",\"reason\":\"leak\"}'".into()],
            events: vec![HookEvent::PostTool],
            env_allowlist: true,
            env: vec![("ALLOWED".into(), "1".into())],
            ..Default::default()
        };
        std::env::set_var("FAKTOR_TOKEN", "sekrit");
        r.register(spec.clone()).unwrap();
        spec.env_allowlist = false;
        // Allowlist on: token stripped -> allow.
        let r2 = HookRegistry::new();
        r2.register(spec).unwrap();
        let v = r2.run(HookEvent::PostTool, &HookInput::default());
        // env_allowlist true version was replaced; re-register properly:
        let allow = HookSpec {
            id: "env2".into(),
            command: "sh".into(),
            args: vec![
                "-c".into(),
                "test -z \"$FAKTOR_TOKEN\" && echo '{\"verdict\":\"allow\"}'".into(),
            ],
            events: vec![HookEvent::PreTool],
            env_allowlist: true,
            env: vec![],
            ..Default::default()
        };
        let r3 = HookRegistry::new();
        r3.register(allow.clone()).unwrap();
        assert!(matches!(
            r3.run(HookEvent::PreTool, &HookInput::default()),
            HookVerdict::Allow
        ));
        let _ = r;
        let _ = v;
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
