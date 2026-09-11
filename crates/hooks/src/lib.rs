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
//!
//! Audit P0-40 (unified process supervision): hooks own NO process
//! machinery. Every child is spawned by the workspace's single
//! [`faktor_terminal::ProcessSupervisor`] (the registry holds an `Arc`
//! and passes it into every run); the hook's env-clear/allowlist/bounded
//! output/deadline semantics are expressed as an
//! [`faktor_terminal::EnvSpec`] + caps on [`faktor_terminal::ProcessSupervisor::run_sync`],
//! and `permission_scope` is a typed [`CapabilitySet`] whose lattice subset
//! check against the registry's granted envelope refuses an out-of-scope
//! hook BEFORE any child exists. Audit records (bounded head only) stay
//! here — the caller's durable journal for hook runs.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use faktor_core::id::SessionId;
use faktor_core::CapabilitySet;
use faktor_terminal::{EnvSpec, ProcessOwner, ProcessSupervisor, SpawnConfig};

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
    /// The hook's TYPED capability scope (audit P0-40): a lattice element
    /// over [`CapabilitySet`]. A hook whose scope is not a subset of the
    /// registry's granted envelope is refused before any child is spawned.
    pub permission_scope: CapabilitySet,
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
            permission_scope: CapabilitySet::EMPTY,
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

/// The audit surface for one stream: the bounded head plus an ellipsis when
/// the stream was truncated. Never unbounded — `head` is already capped
/// (the cap enforcement itself lives in the supervisor's reader threads).
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
    /// The ONE process supervisor every hook child runs under (audit
    /// P0-40): hooks own no process machinery of their own.
    supervisor: Arc<ProcessSupervisor>,
    /// The typed capability envelope granted to hooks: a hook whose scope
    /// exceeds it is refused before spawn.
    envelope: CapabilitySet,
}

pub struct HookRegistry {
    inner: Inner,
}

impl HookRegistry {
    /// A registry over the process-wide shared supervisor with an
    /// unrestricted envelope. Kept for constructors that predate the
    /// supervisor wiring (env-var registry, crate tests); production callers
    /// pass their daemon-rooted supervisor through
    /// [`HookRegistry::with_supervisor`].
    pub fn new() -> Self {
        Self::with_supervisor(ProcessSupervisor::shared(), CapabilitySet::ALL)
    }

    /// The production constructor (audit P0-40): every hook run is
    /// submitted to the SAME `Arc<ProcessSupervisor>` that supervises the
    /// rest of the daemon's children (bounded registry, daemon-shutdown
    /// scope, in-memory owner rows). Hooks whose typed `permission_scope`
    /// is not within `envelope` are refused before any child exists.
    pub fn with_supervisor(supervisor: Arc<ProcessSupervisor>, envelope: CapabilitySet) -> Self {
        Self {
            inner: Inner {
                specs: Arc::new(Mutex::new(Vec::new())),
                audit: Arc::new(Mutex::new(VecDeque::new())),
                supervisor,
                envelope,
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
    ///
    /// The child is spawned through the shared supervisor
    /// ([`ProcessSupervisor::run_sync`] — sync by contract: the agent
    /// runtime invokes hooks from synchronous sites); the supervisor owns
    /// env construction ([`EnvSpec`]), the process-group kill on
    /// deadline, the bounded reader threads, and the registry rows. This
    /// method owns the verdict parsing, the failure policy and the
    /// exactly-once audit record.
    pub fn run_one(&self, spec: &HookSpec, input: &HookInput) -> HookVerdict {
        let started = std::time::Instant::now();
        // Typed capability envelope: a hook whose scope exceeds the granted
        // envelope is refused BEFORE any child exists (lattice subset over
        // typed sets — never a string compare). Refusals are audited
        // exactly once like any other run outcome.
        if !spec.permission_scope.is_subset_of(self.inner.envelope) {
            let outcome = self.policy_outcome(
                spec,
                HookVerdict::Warn {
                    reason: format!(
                        "hook {} scope [{scope}] exceeds the granted envelope [{envelope}]",
                        spec.id,
                        scope = spec.permission_scope,
                        envelope = self.inner.envelope,
                    ),
                },
            );
            let duration = started.elapsed().as_millis() as u64;
            self.audit_push(HookAuditRecord {
                hook_id: spec.id.clone(),
                event: input.event,
                started_ms: now_ms() - duration as i64,
                duration_ms: duration,
                verdict: verdict_tag(&outcome),
                exit_code: None,
                stdout_head: String::new(),
                stderr_head: String::new(),
                failure_policy: spec.failure_policy,
            });
            return outcome;
        }
        let input_json = serde_json::json!({
            "event": input.event,
            "session_id": input.session_id,
            "task_id": input.task_id,
            "operation_id": input.operation_id,
            "payload": input.payload,
        });
        // Env is ALWAYS an EnvSpec::Explicit base: hooks never inherit the
        // daemon env implicitly. env_allowlist (default true) passes only
        // the explicit `env` entries; false adds the fixed benign
        // passthrough set (empty values mean "the daemon's value for that
        // key"). The deny-set still removes secret-shaped names. The input
        // JSON rides the env (bounded by callers at 64 KiB) and stdin stays
        // null.
        let mut entries: Vec<(std::ffi::OsString, std::ffi::OsString)> = Vec::new();
        if !spec.env_allowlist {
            for key in BENIGN_ENV_PASSTHROUGH {
                entries.push((key.into(), std::ffi::OsString::new()));
            }
        }
        entries.extend(spec.env.iter().map(|(k, v)| (k.into(), v.into())));
        entries.push(("FAKTOR_HOOK_INPUT".into(), input_json.to_string().into()));
        let cfg = SpawnConfig {
            cmd: spec.command.clone(),
            args: spec.args.clone(),
            cwd: std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/")),
            env: EnvSpec::Explicit(entries),
            owner: process_owner(input),
            ..Default::default()
        };
        let out = match self.inner.supervisor.run_sync(
            cfg,
            std::time::Duration::from_millis(spec.deadline_ms),
            spec.stdout_cap,
            spec.stderr_cap,
        ) {
            Ok(out) => out,
            Err(e) => {
                // Spawn refused (not found, or the supervisor's bounded
                // live-child ceiling): policy outcome, exactly as a spawn
                // failure always resolved.
                return self.policy_outcome(
                    spec,
                    HookVerdict::Warn {
                        reason: format!("spawn: {e}"),
                    },
                );
            }
        };
        let duration = started.elapsed().as_millis() as u64;
        // The deadline dominates: a timed-out run's partial output NEVER
        // decides — the verdict is the failure policy's outcome. Only a run
        // that exited on its own within the deadline may have its stdout
        // parsed.
        let outcome = if out.timed_out {
            self.policy_outcome(
                spec,
                HookVerdict::Warn {
                    reason: format!(
                        "hook {} exceeded the {} ms deadline and was killed",
                        spec.id, spec.deadline_ms
                    ),
                },
            )
        } else {
            match parse_verdict(&out.stdout_head) {
                Some(v) => v,
                None => {
                    if out.exit_code != Some(0) {
                        self.policy_outcome(
                            spec,
                            HookVerdict::Warn {
                                reason: format!(
                                    "hook {} exited {:?} without a verdict",
                                    spec.id, out.exit_code
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
            exit_code: out.exit_code,
            stdout_head: audit_head(&out.stdout_head, out.stdout_truncated),
            stderr_head: audit_head(&out.stderr_head, out.stderr_truncated),
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

/// Map the hook input's session id to the supervisor's owner row (audit
/// P0-40 / zero-orphans): hook children of a session die with the session
/// via the supervisor's `kill_all_for`. An unparsable id falls back to the
/// daemon owner (hook children are deadline-bounded in all cases).
fn process_owner(input: &HookInput) -> ProcessOwner {
    match &input.session_id {
        Some(s) => match s.parse::<u64>() {
            Ok(n) => ProcessOwner::Session(SessionId::new(n)),
            Err(_) => ProcessOwner::Daemon,
        },
        None => ProcessOwner::Daemon,
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
        // A sub-millisecond hook legitimately rounds duration_ms to 0 on
        // fast hosts; the audit contract is ordering + presence, not a
        // nonzero wall duration (that was a host-speed assumption).
        assert!(audit[0].started_ms > 0 && audit[0].duration_ms < 60_000);
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

    // ------------------------------------------- audit P0-40 adversarial

    /// (a) 100 concurrent hook executions through ONE supervisor: the
    /// bounded registry admits exactly the configured live ceiling, the
    /// 101st child is refused with a typed Oversized error BEFORE any
    /// process exists, every admitted run completes, the registry drains,
    /// and the audit log carries exactly one record per run.
    #[test]
    fn hundred_concurrent_hooks_obey_the_bounded_registry_and_audit_once() {
        use faktor_core::error::ErrorKind;
        let dir = tempfile::tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::with_limit(cas, 100);
        let reg = Arc::new(HookRegistry::with_supervisor(
            sup.clone(),
            CapabilitySet::ALL,
        ));
        let specs: Vec<HookSpec> = (0..100)
            .map(|i| HookSpec {
                id: format!("conc-{i}"),
                command: "sh".into(),
                args: vec![
                    "-c".into(),
                    "sleep 1; echo '{\"verdict\":\"allow\"}'".into(),
                ],
                events: vec![HookEvent::PreTool],
                ..Default::default()
            })
            .collect();
        for spec in &specs {
            reg.register(spec.clone()).unwrap();
        }
        let specs = Arc::new(specs);
        let barrier = Arc::new(std::sync::Barrier::new(101));
        let mut handles = Vec::new();
        for i in 0..100 {
            let reg = reg.clone();
            let specs = specs.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                reg.run_one(&specs[i], &HookInput::default())
            }));
        }
        barrier.wait();
        // Wait until the registry is exactly full, then prove the 101st
        // concurrent child is refused BEFORE it exists.
        for _ in 0..200 {
            if sup.alive().len() == 100 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(sup.alive().len(), 100, "the registry is exactly full");
        let err = sup
            .spawn(faktor_terminal::SpawnConfig {
                cmd: "sh".into(),
                args: vec!["-c".into(), "true".into()],
                ..Default::default()
            })
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized, "{err:?}");
        assert_eq!(sup.alive().len(), 100, "the refused child never existed");
        let mut verdicts_ok = 0;
        for h in handles {
            assert!(matches!(h.join().unwrap(), HookVerdict::Allow));
            verdicts_ok += 1;
        }
        assert_eq!(verdicts_ok, 100);
        // No orphans: every admitted child exited; the registry drains.
        for _ in 0..200 {
            sup.reap();
            if sup.registered() == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(sup.registered(), 0, "registry fully drained");
        assert!(sup.alive().is_empty(), "no live children remain");
        // Exactly-once audit records: one per run, none lost or duplicated.
        let audit = reg.audit();
        assert_eq!(audit.len(), 100, "exactly one audit record per run");
        for rec in &audit {
            assert_eq!(rec.verdict, "allow");
            assert_eq!(rec.exit_code, Some(0));
        }
    }

    /// (a) deadline kill reaches the GRANDCHILD: the hook's child spawns a
    /// sleep-30 grandchild and waits; the deadline fires, the OWNED tree is
    /// group-killed, the grandchild dies, no orphan survives, the verdict
    /// is the failure policy's and the audit records the kill exactly once.
    #[test]
    fn deadline_kill_takes_the_hook_grandchild() {
        let dir = tempfile::tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let r = HookRegistry::with_supervisor(sup.clone(), CapabilitySet::ALL);
        let pidfile = dir.path().join("grandchild.pid");
        let cmd = format!("sleep 30 & echo $! > '{}'; wait", pidfile.display());
        r.register(HookSpec {
            id: "grandchild".into(),
            command: "sh".into(),
            args: vec!["-c".into(), cmd],
            events: vec![HookEvent::PreModel],
            deadline_ms: 500,
            failure_policy: FailurePolicy::FailClosed,
            ..Default::default()
        })
        .unwrap();
        let t0 = std::time::Instant::now();
        let v = r.run(HookEvent::PreModel, &HookInput::default());
        assert!(matches!(v, HookVerdict::Deny { .. }), "{v:?}");
        assert!(
            t0.elapsed().as_millis() < 8000,
            "the deadline must bound the run"
        );
        let gc: u32 = std::fs::read_to_string(&pidfile)
            .unwrap()
            .trim()
            .parse()
            .expect("grandchild pid file");
        let mut gone = false;
        for _ in 0..100 {
            let alive = unsafe { libc::kill(gc as i32, 0) == 0 };
            if !alive {
                gone = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        assert!(gone, "the hook grandchild must die with the group kill");
        let audit = r.audit();
        assert_eq!(audit.len(), 1, "exactly one audit record");
        assert_eq!(audit[0].verdict, "deny", "timeout fails closed");
        assert_eq!(audit[0].exit_code, None, "the tree was killed");
        assert!(
            audit[0].duration_ms >= 500,
            "the run lasted through the deadline"
        );
    }

    /// (d) typed capability envelope: a hook whose typed scope exceeds the
    /// registry's granted envelope is refused BEFORE any child is spawned
    /// (marker file never appears), under the failure policy; an in-scope
    /// hook with the same command runs.
    #[test]
    fn scope_exceeding_the_envelope_is_refused_before_spawn() {
        use faktor_core::{CapabilityKind, CapabilitySet};
        let dir = tempfile::tempdir().unwrap();
        let cas = Arc::new(faktor_cas::Cas::open(dir.path().join("cas")).unwrap());
        let sup = ProcessSupervisor::new(cas);
        let envelope = CapabilitySet::of(CapabilityKind::Execute)
            .union(CapabilitySet::of(CapabilityKind::Read));
        let r = HookRegistry::with_supervisor(sup.clone(), envelope);
        let marker = dir.path().join("ran");
        let cmd = format!(
            "touch '{}'; echo '{{\"verdict\":\"allow\"}}'",
            marker.display()
        );
        r.register(HookSpec {
            id: "over".into(),
            command: "sh".into(),
            args: vec!["-c".into(), cmd.clone()],
            events: vec![HookEvent::PreTool],
            permission_scope: CapabilitySet::of(CapabilityKind::Network),
            ..Default::default()
        })
        .unwrap();
        let over = r
            .matching(HookEvent::PreTool)
            .into_iter()
            .find(|s| s.id == "over")
            .unwrap();
        let v = r.run_one(&over, &HookInput::default());
        assert!(
            matches!(v, HookVerdict::Deny { .. }),
            "an out-of-envelope scope fails closed: {v:?}"
        );
        assert!(
            !marker.exists(),
            "the refused hook must never spawn a child"
        );
        let audit = r.audit();
        assert_eq!(audit.len(), 1, "the refusal is audited exactly once");
        assert_eq!(audit[0].verdict, "deny");
        assert_eq!(audit[0].exit_code, None);
        assert!(audit[0].stdout_head.is_empty());
        // The SAME command with an in-envelope scope runs.
        r.register(HookSpec {
            id: "within".into(),
            command: "sh".into(),
            args: vec!["-c".into(), cmd],
            events: vec![HookEvent::PreTool],
            permission_scope: CapabilitySet::of(CapabilityKind::Execute),
            ..Default::default()
        })
        .unwrap();
        let within = r
            .matching(HookEvent::PreTool)
            .into_iter()
            .find(|s| s.id == "within")
            .unwrap();
        assert_eq!(
            r.run_one(&within, &HookInput::default()),
            HookVerdict::Allow
        );
        assert!(marker.exists(), "an in-envelope hook runs");
        assert_eq!(r.audit().len(), 2);
    }

    /// The source-level invariant (audit P0-40): crates/hooks and
    /// crates/mcp contain NO `std::process::Command` spawn/construction
    /// outside the supervisor crate. Bounded scan of the two crates'
    /// production sections (cut at the first `#[cfg(test)]`), modeled on
    /// the wave-17 egress scan.
    #[test]
    fn no_child_spawn_outside_the_supervisor_crate() {
        const MARKERS: [&str; 4] = [
            "process::Command",
            "Command::new",
            "Command::spawn",
            ".spawn()",
        ];
        let crates_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/hooks sits directly under crates/")
            .to_path_buf();
        let files = ["hooks/src/lib.rs", "mcp/src/lib.rs", "terminal/src/lib.rs"];
        let mut scanned = 0usize;
        let mut offenders: Vec<String> = Vec::new();
        for rel in files {
            let path = crates_root.join(rel);
            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            // Production section only: cut at the first #[cfg(test)].
            let production = source.split("#[cfg(test)]").next().unwrap_or(&source);
            scanned += 1;
            if rel == "terminal/src/lib.rs" {
                continue; // the ONE crate that owns process spawning
            }
            for (idx, line) in production.lines().enumerate() {
                if MARKERS.iter().any(|m| line.contains(m)) {
                    offenders.push(format!("{rel}:{}: {}", idx + 1, line.trim()));
                }
            }
        }
        assert!(scanned >= 3, "the scan walked nothing: {scanned}");
        assert!(
            offenders.is_empty(),
            "child spawn machinery outside the supervisor crate:\n  {}\n\
             hooks and mcp must route every child through the shared \
             faktor-terminal::ProcessSupervisor.",
            offenders.join("\n  ")
        );
    }
}
