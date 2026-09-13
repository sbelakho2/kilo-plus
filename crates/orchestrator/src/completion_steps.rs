//! PR/CI-fix completion step EXECUTION (the P2 follow-up to the durable
//! completion-contract gate in `faktor-core`/`faktor-session`).
//!
//! [`CompletionStepRunner`] executes the conditional steps a task run's
//! ACCEPTED durable completion contract requests — in gate order
//! (commit, push, pr) — against the run's integration/candidate root, and
//! records every outcome through the SAME durable setter the gate reads
//! ([`SessionHandle::set_completion_step_status`]). Nothing here is a second
//! completion authority: after the runner, the existing
//! `complete_verified_task` gate is asked to complete and refuses exactly
//! as before (missing -> `CompletionStepMissing`, non-succeeded ->
//! `CompletionStepNotSucceeded`, any `Failed` row -> `CompletionStepFailed`,
//! terminal).
//!
//! Ordering (normative): deterministic verification FIRST, then these steps,
//! then the completion gate. The orchestrator invokes the runner only AFTER
//! a run's drive has ended (verification ran) and records outcomes durably
//! BEFORE any later completion attempt. A run with no contract — or the
//! all-false default — NEVER constructs or invokes the runner: the default
//! path stays byte-identical.
//!
//! Semantics per step:
//!
//! - **commit**: stage every change in the task's integration/candidate root
//!   and `git commit` on the current branch, with a bounded message derived
//!   from the goal. A clean tree is honestly `Skipped` ("nothing to
//!   commit"); an unborn HEAD (empty repository) is `Skipped` too — this
//!   path never mints an empty or initial commit. A clean tree whose HEAD
//!   already carries the SAME deterministic message (a crash between the
//!   commit and its status row) records `Succeeded` ("already committed").
//! - **push**: push the current branch to the configured remote through the
//!   workspace's git manager, consulting the injected egress policy for any
//!   real network destination (local path / `file://` remotes are not
//!   egress). No remote configured => `Skipped` recorded; policy denial =>
//!   `Failed` with the typed reason; an already-pushed HEAD (local
//!   remote-tracking ref equal to HEAD, or git's "Everything up-to-date")
//!   => `Succeeded` with a note, so retries are idempotent.
//! - **pr**: create the PR ONLY when `pr_command` is configured (strict
//!   config; templated with `{branch}`/`{base}`/`{remote}`; executed through
//!   the process supervisor with a sanitized baseline environment and
//!   bounded captured output — never a direct network call). Unconfigured =>
//!   `Skipped`; output naming a PR URL => `Succeeded` with the parsed URL; a
//!   command that reports the PR "already exists" => `Succeeded` (idempotent)
//!   with the URL when present.
//!
//! Stop rule: only a `Failed` step stops the ordered execution (terminal for
//! the contract revision). Remaining requested steps are recorded `Skipped`
//! ("not attempted: <step> failed") and are never executed. A `Skipped` step
//! (e.g. nothing to commit, no remote, unconfigured PR) does NOT stop the
//! sequence but is still an unmet step for the gate: it stays retryable.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use faktor_core::cancellation::CancellationToken;
use faktor_core::completion::{CompletionStep, CompletionStepOutcome};
use faktor_core::id::TaskId;
use faktor_git::{CommitOutcome, PushOutcome, WorktreeManager};
use faktor_session::{SessionHandle, MAX_COMPLETION_STEP_DETAIL};
use faktor_terminal::{EnvSpec, ProcessOwner, ProcessSupervisor, SpawnConfig};

/// Hard bound on the goal-derived commit message (UTF-8 bytes).
pub const MAX_COMMIT_MESSAGE_BYTES: usize = 200;
/// Hard bound on one configured PR command template (and on ONE typed argv
/// element: the program or one argument).
pub const MAX_PR_COMMAND_BYTES: usize = 4096;
/// Hard bound on the number of typed `pr_args` elements.
pub const MAX_PR_ARGS: usize = 128;
/// Bound on the captured PR command excerpt put into the durable detail.
pub const MAX_PR_OUTPUT_BYTES: usize = 2000;
/// Deliberate bound on one PR command (a wedged helper must not hang the
/// daemon; the supervisor kills it and the step fails typed).
pub const PR_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// The strict execution configuration of the completion-step runner. The
/// daemon maps its `[completion]` config section onto this type; the
/// defaults are intentionally inert (no PR command), so an unconfigured
/// daemon records the documented `Skipped` outcomes.
///
/// Two PR shapes are accepted:
/// - `pr_program` + `pr_args` (P3, preferred): a typed argv executed
///   directly through the supervisor — no shell, no whitespace splitting, so
///   quoted arguments and paths with spaces are representable. Every element
///   substitutes `{branch}`/`{base}`/`{remote}` independently.
/// - `pr_command` (DEPRECATED): the historic whitespace-split string
///   template, kept working with the exact same security checks. It is
///   mutually exclusive with the typed shape.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionStepsConfig {
    /// The git remote the push step targets.
    pub remote: String,
    /// The base branch rendered into the PR template.
    pub base_branch: String,
    /// DEPRECATED: the whitespace-split PR command template. `None` = the
    /// documented "not configured" `Skipped` outcome for a requested PR
    /// step (unless `pr_program` is configured).
    #[serde(default)]
    pub pr_command: Option<String>,
    /// The typed argv program (executed directly; spaces are preserved).
    /// `None` with `pr_command: None` = the documented `Skipped` outcome.
    #[serde(default)]
    pub pr_program: Option<String>,
    /// The typed argv arguments; each element is placeholder-substituted.
    #[serde(default)]
    pub pr_args: Vec<String>,
}

impl Default for CompletionStepsConfig {
    fn default() -> Self {
        Self {
            remote: "origin".into(),
            base_branch: "main".into(),
            pr_command: None,
            pr_program: None,
            pr_args: Vec::new(),
        }
    }
}

impl CompletionStepsConfig {
    /// Strict validation: bounded, option-shaped remote and base branch; a
    /// PR command that is bounded, single-line and templated only with the
    /// documented placeholders and at least `{branch}` (a command that
    /// cannot name the PR head is refused). The legacy string template
    /// additionally refuses shell metacharacters (it is split on
    /// whitespace); the typed argv shape allows every non-control character
    /// because no shell and no splitting is ever involved. Configuring BOTH
    /// shapes is ambiguous and refused.
    pub fn validate(&self) -> Result<(), String> {
        validate_remote_name(&self.remote)?;
        validate_branch_name(&self.base_branch)?;
        match (&self.pr_command, &self.pr_program) {
            (Some(_), Some(_)) => Err(
                "pr_command (deprecated) and pr_program are mutually exclusive; configure exactly one"
                    .into(),
            ),
            (Some(command), None) => validate_pr_command(command),
            (None, Some(program)) => validate_pr_argv(program, &self.pr_args),
            (None, None) if self.pr_args.is_empty() => Ok(()),
            (None, None) => Err("pr_args requires pr_program".into()),
        }
    }
}

/// The egress decision seam the push step consults for network remotes. The
/// daemon wires its sandbox `PermissionEngine::check_egress` here; tests
/// inject allow/deny policies. Local-path and `file://` remotes are NOT
/// egress and never consult this policy.
pub trait EgressPolicy: Send + Sync {
    /// `Ok(())` allows the destination; `Err(reason)` is the typed denial.
    fn check(&self, destination: &str) -> Result<(), String>;
}

impl<F> EgressPolicy for F
where
    F: Fn(&str) -> Result<(), String> + Send + Sync,
{
    fn check(&self, destination: &str) -> Result<(), String> {
        self(destination)
    }
}

/// One task-run execution context: where the steps run and what the commit
/// message / PR template derive from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionStepContext {
    /// The task's integration/candidate root (the live shadow root for a
    /// shadowed run, else the session's durable owner worktree root).
    pub root: PathBuf,
    /// The task goal (the commit message derives from it, bounded).
    pub goal: String,
}

/// One recorded step outcome (mirrors the durable row plus its seq).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionStepRecord {
    pub step: CompletionStep,
    pub status: CompletionStepOutcome,
    pub detail: String,
    /// The durable row seq when THIS invocation appended one; `None` for an
    /// idempotent replay of an existing row or a terminal pre-existing
    /// failure (nothing new is written).
    pub seq: Option<i64>,
}

/// The outcome of one runner invocation.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompletionStepReport {
    /// Every requested step, in gate order, with its durable outcome.
    pub records: Vec<CompletionStepRecord>,
    /// The PR URL parsed from the configured command's output, when any.
    pub pr_url: Option<String>,
}

impl CompletionStepReport {
    /// The requested steps in execution order.
    pub fn requested(&self) -> Vec<CompletionStep> {
        self.records.iter().map(|r| r.step).collect()
    }

    /// The latest outcome of one step in this report.
    pub fn outcome_of(&self, step: CompletionStep) -> Option<CompletionStepOutcome> {
        self.records
            .iter()
            .rev()
            .find(|r| r.step == step)
            .map(|r| r.status)
    }

    /// True when every requested step recorded `Succeeded`.
    pub fn all_succeeded(&self) -> bool {
        self.records
            .iter()
            .all(|r| r.status == CompletionStepOutcome::Succeeded)
    }

    /// The first failed step, if any.
    pub fn failed_step(&self) -> Option<CompletionStep> {
        self.records
            .iter()
            .find(|r| r.status == CompletionStepOutcome::Failed)
            .map(|r| r.step)
    }
}

/// Typed failure of the runner's OWN plumbing (contract/ledger reads, config
/// render). Per-step outcomes are never errors: they are recorded durably as
/// `Failed` rows (the gate's terminal refusal), not returned as `Err`.
#[derive(Debug, thiserror::Error)]
pub enum CompletionStepError {
    #[error("completion step session read: {0}")]
    Session(#[from] faktor_session::TaskError),
    #[error("completion step ledger read: {0}")]
    Ledger(#[from] faktor_core::Error),
    #[error("completion step config: {0}")]
    Config(String),
}

/// What one step execution produced.
enum StepExecution {
    Succeeded {
        detail: String,
        pr_url: Option<String>,
    },
    Skipped {
        detail: String,
    },
    Failed {
        detail: String,
    },
}

impl StepExecution {
    fn status(&self) -> CompletionStepOutcome {
        match self {
            StepExecution::Succeeded { .. } => CompletionStepOutcome::Succeeded,
            StepExecution::Skipped { .. } => CompletionStepOutcome::Skipped,
            StepExecution::Failed { .. } => CompletionStepOutcome::Failed,
        }
    }

    fn detail(&self) -> &str {
        match self {
            StepExecution::Succeeded { detail, .. }
            | StepExecution::Skipped { detail }
            | StepExecution::Failed { detail } => detail,
        }
    }
}

/// The completion-step executor: git (through the workspace's
/// [`WorktreeManager`], never a shell string), egress policy and the PR
/// command supervisor.
pub struct CompletionStepRunner {
    git: WorktreeManager,
    supervisor: Arc<ProcessSupervisor>,
    egress: Arc<dyn EgressPolicy>,
    config: CompletionStepsConfig,
}

impl std::fmt::Debug for CompletionStepRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionStepRunner")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl CompletionStepRunner {
    /// Build a runner over the daemon's process supervisor. The config is
    /// validated strictly BEFORE any step can run.
    pub fn new(
        supervisor: Arc<ProcessSupervisor>,
        egress: Arc<dyn EgressPolicy>,
        config: CompletionStepsConfig,
    ) -> Result<Self, CompletionStepError> {
        config.validate().map_err(CompletionStepError::Config)?;
        Ok(Self {
            git: WorktreeManager::new(supervisor.clone()),
            supervisor,
            egress,
            config,
        })
    }

    pub fn config(&self) -> &CompletionStepsConfig {
        &self.config
    }

    /// Execute and durably record every requested step of the task's
    /// ACCEPTED contract. With no contract (or the all-false default) this
    /// is a pure no-op: no runner step runs and no durable row is written,
    /// so the default completion path stays byte-identical.
    ///
    /// Idempotent under retry: a step whose latest durable row is already
    /// `Succeeded` is not re-executed and not re-recorded; a terminal
    /// `Failed` row stops everything (nothing new is written); a `Skipped`
    /// row is retried (it is the retryable unmet state).
    pub async fn run(
        &self,
        handle: &SessionHandle,
        task_id: TaskId,
        ctx: &CompletionStepContext,
    ) -> Result<CompletionStepReport, CompletionStepError> {
        let Some((revision, contract)) = handle.completion_contract(task_id)? else {
            return Ok(CompletionStepReport::default());
        };
        let requested = contract.requested_steps();
        if requested.is_empty() {
            return Ok(CompletionStepReport::default());
        }
        let rows = handle.ledger_completion_step_statuses(task_id.raw(), revision.raw())?;
        let mut latest: BTreeMap<CompletionStep, (CompletionStepOutcome, String)> = BTreeMap::new();
        for row in rows {
            latest.insert(row.step, (row.status, row.detail));
        }
        // A terminal failure anywhere in the contract revision is final: the
        // gate can never certify under this revision, so nothing is
        // re-executed and nothing new is written (a later Succeeded row
        // could not resurrect it either).
        if latest
            .values()
            .any(|(status, _)| *status == CompletionStepOutcome::Failed)
        {
            let failed_tag = latest
                .iter()
                .find(|(_, (status, _))| *status == CompletionStepOutcome::Failed)
                .map(|(step, (_, _))| step.tag())
                .unwrap_or("a prior step");
            let report = CompletionStepReport {
                records: requested
                    .into_iter()
                    .map(|step| {
                        let (status, detail) = match latest.get(&step) {
                            Some((status, detail)) => (*status, detail.clone()),
                            None => (
                                CompletionStepOutcome::Skipped,
                                format!(
                                    "not attempted: {failed_tag} failed terminally under this contract revision"
                                ),
                            ),
                        };
                        CompletionStepRecord {
                            step,
                            status,
                            detail,
                            seq: None,
                        }
                    })
                    .collect(),
                pr_url: None,
            };
            return Ok(report);
        }
        let mut report = CompletionStepReport::default();
        for step in requested {
            match latest.get(&step) {
                Some((CompletionStepOutcome::Succeeded, detail)) => {
                    report.records.push(CompletionStepRecord {
                        step,
                        status: CompletionStepOutcome::Succeeded,
                        detail: bounded_detail(&format!(
                            "already succeeded (idempotent replay): {detail}"
                        )),
                        seq: None,
                    });
                    continue;
                }
                Some((CompletionStepOutcome::Skipped, _)) => {
                    // Retryable unmet: fall through and execute again.
                }
                Some((CompletionStepOutcome::Failed, _)) => unreachable!("pre-scanned above"),
                None => {}
            }
            let execution = self.execute(step, ctx).await;
            let detail = bounded_detail(execution.detail());
            let seq = handle
                .set_completion_step_status(task_id, step, execution.status(), &detail)
                .map_err(CompletionStepError::Session)?;
            if let StepExecution::Succeeded {
                pr_url: Some(url), ..
            } = &execution
            {
                report.pr_url = Some(url.clone());
            }
            report.records.push(CompletionStepRecord {
                step,
                status: execution.status(),
                detail,
                seq: Some(seq),
            });
            if execution.status() == CompletionStepOutcome::Failed {
                break;
            }
        }
        // A Failed step stops the ordered execution; every remaining
        // requested step is recorded Skipped ("not attempted") so the
        // durable picture names WHY it is missing.
        if let Some(failed) = report.failed_step() {
            let failed = failed.tag();
            let remaining: Vec<CompletionStep> = contract
                .requested_steps()
                .into_iter()
                .filter(|s| !report.records.iter().any(|r| r.step == *s))
                .collect();
            for step in remaining {
                let detail = bounded_detail(&format!(
                    "not attempted: {failed} failed (terminal for this contract revision)"
                ));
                let seq = handle
                    .set_completion_step_status(
                        task_id,
                        step,
                        CompletionStepOutcome::Skipped,
                        &detail,
                    )
                    .map_err(CompletionStepError::Session)?;
                report.records.push(CompletionStepRecord {
                    step,
                    status: CompletionStepOutcome::Skipped,
                    detail,
                    seq: Some(seq),
                });
            }
        }
        Ok(report)
    }

    async fn execute(&self, step: CompletionStep, ctx: &CompletionStepContext) -> StepExecution {
        match step {
            CompletionStep::Commit => self.execute_commit(ctx).await,
            CompletionStep::Push => self.execute_push(ctx).await,
            CompletionStep::Pr => self.execute_pr(ctx).await,
        }
    }

    async fn execute_commit(&self, ctx: &CompletionStepContext) -> StepExecution {
        let message = commit_message(&ctx.goal);
        match self
            .git
            .commit_all(&ctx.root, &message, ProcessOwner::Daemon)
            .await
        {
            Ok(CommitOutcome::Committed { sha, subject }) => StepExecution::Succeeded {
                detail: format!(
                    "committed {} on the current branch: {subject}",
                    short_sha(&sha)
                ),
                pr_url: None,
            },
            Ok(CommitOutcome::NothingToCommit) => {
                // Crash between the commit and its status row: HEAD already
                // carries the SAME deterministic message, so the step IS
                // done — an idempotent success, never a poisoning Skipped.
                match self.git.head_message(&ctx.root, ProcessOwner::Daemon).await {
                    Ok(Some(previous)) if previous.trim() == message => StepExecution::Succeeded {
                        detail: "already committed (idempotent replay: HEAD carries the \
                                 deterministic completion message)"
                            .into(),
                        pr_url: None,
                    },
                    _ => StepExecution::Skipped {
                        detail: "nothing to commit (working tree clean)".into(),
                    },
                }
            }
            Ok(CommitOutcome::EmptyRepository) => StepExecution::Skipped {
                detail: "empty repository state (unborn HEAD); refusing an initial commit".into(),
            },
            Err(e) => StepExecution::Failed {
                detail: format!("git commit failed: {}", e.message),
            },
        }
    }

    async fn execute_push(&self, ctx: &CompletionStepContext) -> StepExecution {
        let owner = ProcessOwner::Daemon;
        let branch = match self.git.current_branch(&ctx.root, owner.clone()).await {
            Ok(branch) if !branch.is_empty() => branch,
            Ok(_) => {
                return StepExecution::Failed {
                    detail: "cannot push: the repository is on a detached HEAD".into(),
                }
            }
            Err(e) => {
                return StepExecution::Failed {
                    detail: format!("git branch read failed: {}", e.message),
                }
            }
        };
        let remote = self.config.remote.clone();
        let url = match self.git.remote_url(&ctx.root, &remote, owner.clone()).await {
            Ok(None) => {
                return StepExecution::Skipped {
                    detail: "no git remote configured (the repository has no remotes)".into(),
                }
            }
            Ok(Some(url)) => url,
            Err(e) => {
                return StepExecution::Failed {
                    detail: format!("git remote read failed: {}", e.message),
                }
            }
        };
        // Push is an external effect: consult the egress policy for real
        // network destinations (local path / file:// remotes are not egress).
        if is_egress_destination(&url) {
            if let Err(denied) = self.egress.check(&url) {
                return StepExecution::Failed {
                    detail: format!("egress policy denied push to {url}: {denied}"),
                };
            }
        }
        let head = match self.git.head_sha(&ctx.root, owner.clone()).await {
            Ok(Some(head)) => head,
            Ok(None) => {
                return StepExecution::Failed {
                    detail: "cannot push: HEAD is unborn".into(),
                }
            }
            Err(e) => {
                return StepExecution::Failed {
                    detail: format!("git head read failed: {}", e.message),
                }
            }
        };
        // Idempotency WITHOUT a network round-trip: a local remote-tracking
        // ref already naming HEAD means an earlier run pushed this commit.
        if let Ok(Some(pushed)) = self
            .git
            .pushed_ref_sha(&ctx.root, &remote, &branch, owner.clone())
            .await
        {
            if pushed == head {
                return StepExecution::Succeeded {
                    detail: format!(
                        "already pushed {branch} to {remote} ({url}); idempotent replay"
                    ),
                    pr_url: None,
                };
            }
        }
        match self
            .git
            .push_branch(&ctx.root, &remote, &branch, owner)
            .await
        {
            Ok(PushOutcome::Pushed { note }) => StepExecution::Succeeded {
                detail: format!("pushed {branch} to {remote} ({url}){}", note_suffix(&note)),
                pr_url: None,
            },
            Ok(PushOutcome::AlreadyCurrent { note }) => StepExecution::Succeeded {
                detail: format!(
                    "already pushed {branch} to {remote} ({url}); up-to-date{}",
                    note_suffix(&note)
                ),
                pr_url: None,
            },
            Err(e) => StepExecution::Failed {
                detail: format!("git push failed: {}", e.message),
            },
        }
    }

    async fn execute_pr(&self, ctx: &CompletionStepContext) -> StepExecution {
        if self.config.pr_command.is_none() && self.config.pr_program.is_none() {
            return StepExecution::Skipped {
                detail: "[completion] pr_command is not configured; no PR was created".into(),
            };
        }
        let owner = ProcessOwner::Daemon;
        let branch = match self.git.current_branch(&ctx.root, owner).await {
            Ok(branch) if !branch.is_empty() => branch,
            Ok(_) => {
                return StepExecution::Failed {
                    detail: "cannot create a PR from a detached HEAD (no current branch)".into(),
                }
            }
            Err(e) => {
                return StepExecution::Failed {
                    detail: format!("git branch read failed: {}", e.message),
                }
            }
        };
        let (program, args) = if let Some(program) = self.config.pr_program.as_deref() {
            match render_pr_argv(
                program,
                &self.config.pr_args,
                &branch,
                &self.config.base_branch,
                &self.config.remote,
            ) {
                Ok(rendered) => rendered,
                Err(e) => {
                    return StepExecution::Failed {
                        detail: format!("pr_program render failed: {e}"),
                    }
                }
            }
        } else {
            let template = self.config.pr_command.as_deref().unwrap_or_default();
            match render_pr_command(
                template,
                &branch,
                &self.config.base_branch,
                &self.config.remote,
            ) {
                Ok(rendered) => rendered,
                Err(e) => {
                    return StepExecution::Failed {
                        detail: format!("pr_command render failed: {e}"),
                    }
                }
            }
        };
        let cfg = SpawnConfig {
            cmd: program,
            args,
            cwd: ctx.root.clone(),
            // The supervisor ALWAYS env_clears and layers this safe baseline
            // (PATH/HOME/platform bits; secret-shaped names are denied) —
            // never the daemon's full environment.
            env: EnvSpec::default_baseline(),
            owner: ProcessOwner::Daemon,
            capture: true,
            ..Default::default()
        };
        let out = match self
            .supervisor
            .run(cfg, PR_COMMAND_TIMEOUT, CancellationToken::new())
            .await
        {
            Ok(out) => out,
            Err(e) => {
                return StepExecution::Failed {
                    detail: format!("pr command could not run: {}", e.message),
                }
            }
        };
        let excerpt = truncate_bytes(&out.excerpt, MAX_PR_OUTPUT_BYTES);
        let url = parse_pr_url(&out.excerpt);
        let already_exists = out.excerpt.to_ascii_lowercase().contains("already exists");
        match (out.exit_code, already_exists, url) {
            (Some(0), _, Some(url)) => StepExecution::Succeeded {
                detail: format!("pr created: {url}"),
                pr_url: Some(url),
            },
            (Some(0), _, None) => StepExecution::Succeeded {
                detail: format!("pr command exited 0 (no URL in output): {excerpt}"),
                pr_url: None,
            },
            (_, true, url) => StepExecution::Succeeded {
                detail: match &url {
                    Some(url) => format!("pr already exists: {url}"),
                    None => "pr already exists (idempotent replay)".into(),
                },
                pr_url: url,
            },
            (code, _, _) => StepExecution::Failed {
                detail: format!(
                    "pr command exited {}: {excerpt}",
                    code.map(|c| c.to_string())
                        .unwrap_or_else(|| "signal".into())
                ),
            },
        }
    }
}

// ------------------------------------------------------------------ helpers

/// The bounded, deterministic commit message derived from the goal:
/// `faktor: <goal>` truncated on a UTF-8 boundary; an empty goal gets the
/// documented fallback so a commit is never messageless.
pub fn commit_message(goal: &str) -> String {
    let goal = goal.trim();
    let base = if goal.is_empty() {
        "complete the task".to_string()
    } else {
        goal.to_string()
    };
    truncate_bytes(&format!("faktor: {base}"), MAX_COMMIT_MESSAGE_BYTES)
}

/// TRUE when the remote URL names a real network destination that must be
/// consulted through the egress policy. Local paths (absolute, relative,
/// Windows drive) and `file://` URLs are NOT egress; scp-like
/// `user@host:path` and every network scheme are.
pub fn is_egress_destination(url: &str) -> bool {
    let url = url.trim();
    if url.is_empty() {
        return false;
    }
    if let Some((scheme, _)) = url.split_once("://") {
        return matches!(
            scheme.to_ascii_lowercase().as_str(),
            "http" | "https" | "ssh" | "git" | "git+ssh" | "ftp" | "ftps"
        );
    }
    if url.starts_with("file:")
        || url.starts_with('/')
        || url.starts_with("./")
        || url.starts_with("../")
    {
        return false;
    }
    // Windows drive path (`C:\...`) or a UNC path stays local.
    if url.len() >= 2 && url.as_bytes()[1] == b':' && url.as_bytes()[0].is_ascii_alphabetic() {
        return false;
    }
    // scp-like shorthand (`user@host:path`) is a network destination.
    url.contains('@') && url.contains(':')
}

/// Render one strict PR template. The template was validated at config time
/// (known placeholders, no control bytes, `{branch}`
/// present); the rendered command is split on whitespace into program+args
/// and is NEVER passed through a shell.
pub fn render_pr_command(
    template: &str,
    branch: &str,
    base: &str,
    remote: &str,
) -> Result<(String, Vec<String>), String> {
    validate_pr_command(template)?;
    let rendered = template
        .replace("{branch}", branch)
        .replace("{base}", base)
        .replace("{remote}", remote);
    if rendered.contains('{') || rendered.contains('}') {
        return Err("rendered pr_command still carries an unresolved placeholder".into());
    }
    let mut parts = rendered.split_whitespace();
    let program = parts
        .next()
        .ok_or_else(|| "pr_command renders to an empty program".to_string())?
        .to_string();
    Ok((program, parts.map(str::to_string).collect()))
}

/// Strict PR-template validation (DEPRECATED shape): bounded, single-line,
/// no shell metacharacters (no shell is ever involved, and the template is
/// split on whitespace), known placeholders only and at least `{branch}`
/// (the head the PR is created from).
pub fn validate_pr_command(template: &str) -> Result<(), String> {
    if template.trim().is_empty() {
        return Err("pr_command must not be empty".into());
    }
    if template.len() > MAX_PR_COMMAND_BYTES {
        return Err(format!(
            "pr_command of {} bytes exceeds MAX_PR_COMMAND_BYTES ({MAX_PR_COMMAND_BYTES})",
            template.len()
        ));
    }
    for c in template.chars() {
        if c.is_control() {
            return Err("pr_command must not contain control characters or newlines".into());
        }
        if matches!(
            c,
            '|' | '&'
                | ';'
                | '<'
                | '>'
                | '`'
                | '$'
                | '"'
                | '\''
                | '*'
                | '?'
                | '!'
                | '#'
                | '('
                | ')'
        ) {
            return Err(format!(
                "pr_command contains the shell metacharacter {c:?}; the command runs without a shell, so metacharacters are refused (escape it as an argv word instead)"
            ));
        }
    }
    scan_pr_placeholders(template, "pr_command")?;
    if !template.contains("{branch}") {
        return Err("pr_command must template {branch} (the PR head is never implicit)".into());
    }
    Ok(())
}

/// Strict typed-argv validation (P3): `pr_program` plus each `pr_args`
/// element are bounded, single-line (no control characters) argv values with
/// known placeholders only (per element) and `{branch}` somewhere. Spaces
/// are LEGAL — argv elements are executed directly, never split or shelled.
pub fn validate_pr_argv(program: &str, args: &[String]) -> Result<(), String> {
    if program.is_empty() {
        return Err("pr_program must not be empty".into());
    }
    if program.len() > MAX_PR_COMMAND_BYTES {
        return Err(format!(
            "pr_program of {} bytes exceeds MAX_PR_COMMAND_BYTES ({MAX_PR_COMMAND_BYTES})",
            program.len()
        ));
    }
    if args.len() > MAX_PR_ARGS {
        return Err(format!(
            "pr_args of {} elements exceeds MAX_PR_ARGS ({MAX_PR_ARGS})",
            args.len()
        ));
    }
    let mut branch_seen = false;
    for element in std::iter::once(program).chain(args.iter().map(String::as_str)) {
        for c in element.chars() {
            if c.is_control() {
                return Err(
                    "pr_program/pr_args must not contain control characters or newlines".into(),
                );
            }
        }
        branch_seen |= scan_pr_placeholders(element, "pr_program/pr_args")?;
    }
    if !branch_seen {
        return Err(
            "pr_program/pr_args must template {branch} (the PR head is never implicit)".into(),
        );
    }
    Ok(())
}

/// Scan one template element for the documented placeholders; returns TRUE
/// when `{branch}` occurs. Unknown names, a stray `}` and an unclosed `{`
/// are typed refusals.
fn scan_pr_placeholders(template: &str, label: &str) -> Result<bool, String> {
    let bytes = template.as_bytes();
    let mut index = 0;
    let mut branch_seen = false;
    while index < bytes.len() {
        match bytes[index] {
            b'{' => {
                let end = bytes[index + 1..]
                    .iter()
                    .position(|b| *b == b'}')
                    .map(|p| index + 1 + p)
                    .ok_or_else(|| format!("{label} carries an unclosed '{{'"))?;
                let name = &template[index + 1..end];
                if !matches!(name, "branch" | "base" | "remote") {
                    return Err(format!(
                        "{label} placeholder {{{name}}} is unknown; supported: {{branch}}, {{base}}, {{remote}}"
                    ));
                }
                branch_seen |= name == "branch";
                index = end + 1;
            }
            b'}' => return Err(format!("{label} carries a stray '}}'")),
            _ => index += 1,
        }
    }
    Ok(branch_seen)
}

/// Render one typed argv (P3): validate, substitute each element's
/// `{branch}`/`{base}`/`{remote}` independently, and return the program plus
/// the argument vector exactly as the supervisor must spawn it (spaces
/// inside an element are preserved verbatim).
pub fn render_pr_argv(
    program: &str,
    args: &[String],
    branch: &str,
    base: &str,
    remote: &str,
) -> Result<(String, Vec<String>), String> {
    validate_pr_argv(program, args)?;
    let render = |element: &str| {
        element
            .replace("{branch}", branch)
            .replace("{base}", base)
            .replace("{remote}", remote)
    };
    let rendered_program = render(program);
    if rendered_program.contains('{') || rendered_program.contains('}') {
        return Err("rendered pr_program still carries an unresolved placeholder".into());
    }
    if rendered_program.is_empty() {
        return Err("pr_program renders to an empty program".into());
    }
    let rendered_args: Vec<String> = args.iter().map(|arg| render(arg)).collect();
    Ok((rendered_program, rendered_args))
}

fn validate_remote_name(remote: &str) -> Result<(), String> {
    if remote.is_empty() || remote.len() > 128 {
        return Err("remote name must be 1..=128 bytes".into());
    }
    if remote.starts_with('-')
        || remote.contains("..")
        || remote.contains('/')
        || remote.contains('\\')
    {
        return Err(format!("remote name {remote:?} rejected"));
    }
    for c in remote.chars() {
        if c.is_control() || c.is_whitespace() || matches!(c, '~' | '^' | ':' | '?' | '*' | '[') {
            return Err(format!("remote name {remote:?} rejected"));
        }
    }
    Ok(())
}

fn validate_branch_name(branch: &str) -> Result<(), String> {
    if branch.is_empty() || branch.len() > 128 {
        return Err("base branch must be 1..=128 bytes".into());
    }
    if branch.starts_with('-') || branch.contains("..") {
        return Err(format!("base branch {branch:?} rejected"));
    }
    for c in branch.chars() {
        if c.is_control()
            || c.is_whitespace()
            || matches!(c, '~' | '^' | ':' | '?' | '*' | '[' | '\\')
        {
            return Err(format!("base branch {branch:?} rejected"));
        }
    }
    Ok(())
}

/// The FIRST `http(s)://…` URL in the output, trailing punctuation trimmed.
fn parse_pr_url(output: &str) -> Option<String> {
    for token in output.split_whitespace() {
        let Some(start) = token.find("https://").or_else(|| token.find("http://")) else {
            continue;
        };
        let candidate = token[start..].trim_end_matches(|c: char| {
            matches!(
                c,
                '.' | ',' | ';' | ':' | ')' | ']' | '}' | '>' | '"' | '\''
            )
        });
        if candidate.len() > "https://".len() {
            return Some(candidate.to_string());
        }
    }
    None
}

fn note_suffix(note: &str) -> String {
    if note.is_empty() {
        String::new()
    } else {
        format!(": {note}")
    }
}

fn short_sha(sha: &str) -> &str {
    &sha[..sha.len().min(12)]
}

fn bounded_detail(detail: &str) -> String {
    truncate_bytes(detail, MAX_COMPLETION_STEP_DETAIL)
}

fn truncate_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

#[cfg(test)]
mod tests {
    //! Adversarial covers of the completion-step runner: no-contract parity,
    //! truthful commit outcomes with a bounded message, Skipped/retryable
    //! gate interplay, egress-denied and remote-less pushes, unconfigured /
    //! configured PR commands (URL parse + already-exists idempotency),
    //! ordering + terminal stop, and reopen idempotency across a simulated
    //! crash between steps.
    use super::*;
    use faktor_core::completion::CompletionContract;
    use faktor_core::id::VerificationRecordId;
    use faktor_core::state::{TaskState, TaskTransition, VerificationStatus};
    use faktor_session::{CompletionContractGate, SessionManager, Task, TaskBudget, TaskError};
    use std::path::Path;

    fn allow_all() -> Arc<dyn EgressPolicy> {
        Arc::new(|_url: &str| Ok(()))
    }

    fn deny_all() -> Arc<dyn EgressPolicy> {
        Arc::new(|url: &str| Err(format!("network denied by test policy for {url}")))
    }

    fn manager() -> (tempfile::TempDir, Arc<SessionManager>) {
        let dir = tempfile::tempdir().unwrap();
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        (dir, m)
    }

    fn session_with_task(
        m: &Arc<SessionManager>,
        goal: &str,
        contract: Option<CompletionContract>,
    ) -> (SessionHandle, TaskId) {
        let ws = m.create_workspace("/w").unwrap();
        let h = m
            .create_session(ws, "completion-steps", "fake", "fake")
            .unwrap();
        let task_id = h.task_id().unwrap();
        let now = h.now_ms();
        h.create_task(Task {
            task_id,
            session_id: h.id(),
            goal: goal.into(),
            acceptance_criteria: vec![],
            plan: vec![],
            attachments: Vec::new(),
            budget: TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: now,
            updated_ms: now,
        })
        .unwrap();
        if let Some(contract) = contract {
            let rev = h.task_revision(task_id).unwrap();
            h.set_completion_contract(task_id, rev, contract).unwrap();
        }
        (h, task_id)
    }

    fn runner(
        dir: &Path,
        config: CompletionStepsConfig,
        egress: Arc<dyn EgressPolicy>,
    ) -> CompletionStepRunner {
        let cas = Arc::new(faktor_cas::Cas::open(dir.join("exec-cas")).unwrap());
        let supervisor = ProcessSupervisor::new(cas);
        CompletionStepRunner::new(supervisor, egress, config).unwrap()
    }

    fn git(cwd: &Path, args: &[&str]) {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?} in {}: {}",
            cwd.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    fn git_output(cwd: &Path, args: &[&str]) -> String {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(cwd)
            .output()
            .unwrap();
        assert!(out.status.success(), "git {args:?}");
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn init_repo(root: &Path) -> std::path::PathBuf {
        let repo = root.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        git(&repo, &["init", "-q", "-b", "main"]);
        git(&repo, &["config", "user.email", "test@kilo.local"]);
        git(&repo, &["config", "user.name", "Kilo Test"]);
        std::fs::write(repo.join("README.md"), "base\n").unwrap();
        git(&repo, &["add", "-A"]);
        git(&repo, &["commit", "-q", "-m", "init"]);
        repo
    }

    fn add_bare_remote(root: &Path, repo: &Path) -> std::path::PathBuf {
        let bare = root.join("remote.git");
        git(root, &["init", "--bare", "-q", bare.to_str().unwrap()]);
        git(repo, &["remote", "add", "origin", bare.to_str().unwrap()]);
        bare
    }

    fn contract(commit: bool, push: bool, pr: bool) -> CompletionContract {
        CompletionContract {
            include_commit: commit,
            include_push: push,
            include_pr: pr,
        }
    }

    fn ctx(root: &Path, goal: &str) -> CompletionStepContext {
        CompletionStepContext {
            root: root.to_path_buf(),
            goal: goal.to_string(),
        }
    }

    fn step_rows(
        h: &SessionHandle,
        task_id: TaskId,
    ) -> Vec<faktor_session::CompletionStepStatusRow> {
        // Statuses are keyed by the CONTRACT revision, not the task row's
        // current revision (machine transitions bump the latter).
        let (revision, _) = h.completion_contract(task_id).unwrap().expect("contract");
        h.ledger_completion_step_statuses(task_id.raw(), revision.raw())
            .unwrap()
    }

    fn drive_to_verifying(h: &SessionHandle, task_id: TaskId) {
        for transition in [
            TaskTransition::StartRunning,
            TaskTransition::RequestVerification,
            TaskTransition::StartVerification,
        ] {
            let rev = h.task_revision(task_id).unwrap();
            h.transition_task(task_id, rev, transition, None).unwrap();
        }
    }

    fn passing_record(h: &SessionHandle, task_id: TaskId) -> VerificationRecordId {
        h.create_verification_record(
            task_id,
            None,
            vec![],
            vec![],
            vec![],
            vec![],
            None,
            VerificationStatus::Passed,
            h.now_ms(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn no_contract_is_a_pure_no_op_parity_path() {
        let (dir, m) = manager();
        let (h, task_id) = session_with_task(&m, "no contract", None);
        let runner = runner(dir.path(), CompletionStepsConfig::default(), allow_all());
        // The root does not even exist: if the runner touched git the test
        // would fail — parity means the runner is never invoked.
        let report = runner
            .run(&h, task_id, &ctx(&dir.path().join("nope"), "no contract"))
            .await
            .unwrap();
        assert!(report.records.is_empty());
        assert!(report.pr_url.is_none());
        assert!(h
            .ledger_completion_contract(task_id.raw())
            .unwrap()
            .is_none());
        let rev = h.task_revision(task_id).unwrap();
        assert!(h
            .ledger_completion_step_statuses(task_id.raw(), rev.raw())
            .unwrap()
            .is_empty());
        assert_eq!(
            h.completion_contract_gate(task_id).unwrap(),
            CompletionContractGate::Satisfied
        );
    }

    #[tokio::test]
    async fn commit_step_commits_real_changes_with_a_bounded_goal_message() {
        let (dir, m) = manager();
        let repo = init_repo(dir.path());
        std::fs::write(repo.join("feature.txt"), "new content\n").unwrap();
        let long_goal = format!("implement the feature {}", "x".repeat(500));
        let (h, task_id) = session_with_task(&m, &long_goal, Some(contract(true, false, false)));
        let runner = runner(dir.path(), CompletionStepsConfig::default(), allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, &long_goal))
            .await
            .unwrap();
        assert!(report.all_succeeded(), "{report:?}");
        assert!(dir_clean(&repo));
        let message = git_output(&repo, &["log", "-1", "--pretty=%B"]);
        assert!(
            message.starts_with("faktor: implement the feature"),
            "{message}"
        );
        assert!(
            message.len() <= MAX_COMMIT_MESSAGE_BYTES,
            "{}",
            message.len()
        );
        let rows = step_rows(&h, task_id);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].step, CompletionStep::Commit);
        assert_eq!(rows[0].status, CompletionStepOutcome::Succeeded);
        // Idempotent replay: the status row is not duplicated.
        let replay = runner
            .run(&h, task_id, &ctx(&repo, &long_goal))
            .await
            .unwrap();
        assert_eq!(
            replay.outcome_of(CompletionStep::Commit),
            Some(CompletionStepOutcome::Succeeded)
        );
        assert_eq!(
            step_rows(&h, task_id).len(),
            1,
            "no duplicate row on replay"
        );
        assert!(dir_clean(&repo));
    }

    fn dir_clean(repo: &Path) -> bool {
        git_output(repo, &["status", "--porcelain"]).is_empty()
    }

    #[tokio::test]
    async fn nothing_to_commit_is_skipped_and_the_gate_treats_it_as_unmet() {
        let (dir, m) = manager();
        let repo = init_repo(dir.path());
        let (h, task_id) = session_with_task(&m, "clean tree", Some(contract(true, false, false)));
        let runner = runner(dir.path(), CompletionStepsConfig::default(), allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "clean tree"))
            .await
            .unwrap();
        assert_eq!(
            report.outcome_of(CompletionStep::Commit),
            Some(CompletionStepOutcome::Skipped),
            "{report:?}"
        );
        let rows = step_rows(&h, task_id);
        assert_eq!(rows[0].status, CompletionStepOutcome::Skipped);
        // The documented gate rule: Skipped is NOT succeeded, so a passing
        // proof still refuses — retryably — and the task stays Verifying.
        drive_to_verifying(&h, task_id);
        let record = passing_record(&h, task_id);
        let rev = h.task_revision(task_id).unwrap();
        let err = h.complete_verified_task(task_id, rev, record).unwrap_err();
        match err {
            TaskError::CompletionStepNotSucceeded { status, step, .. } => {
                assert_eq!(step, CompletionStep::Commit);
                assert_eq!(status, CompletionStepOutcome::Skipped);
            }
            other => panic!("expected a retryable Skipped refusal, got {other}"),
        }
        assert_eq!(
            h.get_task(task_id).unwrap().unwrap().state,
            TaskState::Verifying
        );
        // A later real change + re-run flips the SAME contract revision to
        // Succeeded (Skipped stays retryable) and the gate then certifies.
        std::fs::write(repo.join("later.txt"), "later\n").unwrap();
        let report = runner
            .run(&h, task_id, &ctx(&repo, "clean tree"))
            .await
            .unwrap();
        assert!(report.all_succeeded(), "{report:?}");
        let rev = h.task_revision(task_id).unwrap();
        let record = passing_record(&h, task_id);
        let done = h.complete_verified_task(task_id, rev, record).unwrap();
        assert_eq!(done.state, TaskState::VerifiedComplete);
    }

    #[tokio::test]
    async fn push_to_a_local_bare_remote_succeeds_and_bypasses_egress() {
        let (dir, m) = manager();
        let repo = init_repo(dir.path());
        let bare = add_bare_remote(dir.path(), &repo);
        let (h, task_id) = session_with_task(&m, "push it", Some(contract(false, true, false)));
        // A DENY-ALL policy: local path remotes are not egress, so the push
        // still succeeds (the policy is never consulted).
        let runner = runner(dir.path(), CompletionStepsConfig::default(), deny_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "push it"))
            .await
            .unwrap();
        assert_eq!(
            report.outcome_of(CompletionStep::Push),
            Some(CompletionStepOutcome::Succeeded),
            "{report:?}"
        );
        let head = git_output(&repo, &["rev-parse", "HEAD"]);
        assert_eq!(
            git_output(&bare, &["rev-parse", "main"]),
            head,
            "the bare remote must hold the pushed commit"
        );
        let rows = step_rows(&h, task_id);
        assert_eq!(rows[0].status, CompletionStepOutcome::Succeeded);
        // Replay: latest row is Succeeded, so no second row and no push.
        let replay = runner
            .run(&h, task_id, &ctx(&repo, "push it"))
            .await
            .unwrap();
        assert!(replay.all_succeeded());
        assert_eq!(step_rows(&h, task_id).len(), 1);
    }

    #[tokio::test]
    async fn push_denied_by_egress_policy_is_a_typed_failure() {
        let (dir, m) = manager();
        let repo = init_repo(dir.path());
        git(
            &repo,
            &[
                "remote",
                "add",
                "origin",
                "https://git.example.invalid/team/repo.git",
            ],
        );
        let (h, task_id) = session_with_task(&m, "push denied", Some(contract(false, true, false)));
        let runner = runner(dir.path(), CompletionStepsConfig::default(), deny_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "push denied"))
            .await
            .unwrap();
        let failed = report.failed_step();
        assert_eq!(failed, Some(CompletionStep::Push), "{report:?}");
        let rows = step_rows(&h, task_id);
        assert_eq!(rows[0].status, CompletionStepOutcome::Failed);
        assert!(
            rows[0].detail.contains("egress policy denied"),
            "{}",
            rows[0].detail
        );
        // The gate refuses TERMINALLY: even a passing proof cannot certify.
        drive_to_verifying(&h, task_id);
        let record = passing_record(&h, task_id);
        let rev = h.task_revision(task_id).unwrap();
        let err = h.complete_verified_task(task_id, rev, record).unwrap_err();
        assert!(
            matches!(err, TaskError::CompletionStepFailed { .. }),
            "{err}"
        );
        // A re-run never retries a terminally failed step and writes nothing.
        let replay = runner
            .run(&h, task_id, &ctx(&repo, "push denied"))
            .await
            .unwrap();
        assert_eq!(replay.failed_step(), Some(CompletionStep::Push));
        assert_eq!(step_rows(&h, task_id).len(), 1, "terminal rows are frozen");
    }

    #[tokio::test]
    async fn push_without_a_remote_is_skipped() {
        let (dir, m) = manager();
        let repo = init_repo(dir.path());
        let (h, task_id) = session_with_task(&m, "no remote", Some(contract(false, true, false)));
        let runner = runner(dir.path(), CompletionStepsConfig::default(), allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "no remote"))
            .await
            .unwrap();
        assert_eq!(
            report.outcome_of(CompletionStep::Push),
            Some(CompletionStepOutcome::Skipped),
            "{report:?}"
        );
        let rows = step_rows(&h, task_id);
        assert!(rows[0].detail.contains("no git remote configured"));
    }

    #[tokio::test]
    async fn detached_head_push_is_failed_typed() {
        let (dir, m) = manager();
        let repo = init_repo(dir.path());
        add_bare_remote(dir.path(), &repo);
        git(&repo, &["checkout", "-q", "--detach"]);
        let (h, task_id) = session_with_task(&m, "detached", Some(contract(false, true, false)));
        let runner = runner(dir.path(), CompletionStepsConfig::default(), allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "detached"))
            .await
            .unwrap();
        assert_eq!(
            report.failed_step(),
            Some(CompletionStep::Push),
            "{report:?}"
        );
        assert!(step_rows(&h, task_id)[0].detail.contains("detached HEAD"));
    }

    #[tokio::test]
    async fn unconfigured_pr_command_is_skipped() {
        let (dir, m) = manager();
        let (h, task_id) =
            session_with_task(&m, "no pr command", Some(contract(false, false, true)));
        let runner = runner(dir.path(), CompletionStepsConfig::default(), allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&dir.path().join("nope"), "no pr command"))
            .await
            .unwrap();
        assert_eq!(
            report.outcome_of(CompletionStep::Pr),
            Some(CompletionStepOutcome::Skipped),
            "{report:?}"
        );
        assert!(step_rows(&h, task_id)[0]
            .detail
            .contains("pr_command is not configured"));
    }

    #[cfg(unix)]
    fn fake_pr_script(dir: &Path, body: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let script = dir.join("fake-pr.sh");
        std::fs::write(&script, format!("#!/bin/sh\n{body}\n")).unwrap();
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();
        script
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn configured_pr_command_succeeds_and_parses_the_url() {
        let (dir, m) = manager();
        let repo = init_repo(dir.path());
        let script = fake_pr_script(dir.path(), "echo \"https://example.test/pr/42\"");
        let config = CompletionStepsConfig {
            pr_command: Some(format!("{} {{branch}} {{base}}", script.display())),
            ..Default::default()
        };
        let (h, task_id) = session_with_task(&m, "open a pr", Some(contract(false, false, true)));
        let runner = runner(dir.path(), config, allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "open a pr"))
            .await
            .unwrap();
        assert!(report.all_succeeded(), "{report:?}");
        assert_eq!(report.pr_url.as_deref(), Some("https://example.test/pr/42"));
        assert!(step_rows(&h, task_id)[0]
            .detail
            .contains("https://example.test/pr/42"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn typed_argv_pr_program_executes_with_spaces_preserved() {
        let (dir, m) = manager();
        let repo = init_repo(dir.path());
        // A program path WITH SPACES and an argument WITH SPACES: the typed
        // argv is executed directly (no shell, no whitespace splitting), so
        // each element must arrive verbatim. The spy writes its argv to
        // `args.txt` in the step's cwd (the repo root).
        use std::os::unix::fs::PermissionsExt;
        let tools = dir.path().join("my tools");
        std::fs::create_dir_all(&tools).unwrap();
        let script = tools.join("fake pr.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > argv-spy.txt\necho \"https://example.test/pr/11\"\n",
        )
        .unwrap();
        let mut perms = std::fs::metadata(&script).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script, perms).unwrap();

        let config = CompletionStepsConfig {
            pr_program: Some(script.display().to_string()),
            pr_args: vec![
                "pr".into(),
                "create".into(),
                "--head".into(),
                "{branch}".into(),
                "--title".into(),
                "spaces stay one arg".into(),
                "--remote".into(),
                "{remote}".into(),
            ],
            base_branch: "main".into(),
            ..Default::default()
        };
        let (h, task_id) =
            session_with_task(&m, "typed argv pr", Some(contract(false, false, true)));
        let runner = runner(dir.path(), config, allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "typed argv pr"))
            .await
            .unwrap();
        assert!(report.all_succeeded(), "{report:?}");
        assert_eq!(report.pr_url.as_deref(), Some("https://example.test/pr/11"));
        let observed = std::fs::read_to_string(repo.join("argv-spy.txt")).unwrap();
        assert_eq!(
            observed, "pr\ncreate\n--head\nmain\n--title\nspaces stay one arg\n--remote\norigin\n",
            "every argv element must arrive verbatim, spaces included"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn existing_pr_is_an_idempotent_success_with_the_url() {
        let (dir, m) = manager();
        let repo = init_repo(dir.path());
        let script = fake_pr_script(
            dir.path(),
            "echo \"PR already exists: https://example.test/pr/7\"\nexit 1",
        );
        let config = CompletionStepsConfig {
            pr_command: Some(format!("{} {{branch}}", script.display())),
            ..Default::default()
        };
        let (h, task_id) = session_with_task(&m, "existing pr", Some(contract(false, false, true)));
        let runner = runner(dir.path(), config, allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "existing pr"))
            .await
            .unwrap();
        assert_eq!(
            report.outcome_of(CompletionStep::Pr),
            Some(CompletionStepOutcome::Succeeded),
            "{report:?}"
        );
        assert_eq!(report.pr_url.as_deref(), Some("https://example.test/pr/7"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn failed_commit_stops_push_and_pr_and_records_them_skipped() {
        let (dir, m) = manager();
        // A repo whose pre-commit hook REJECTS: the commit fails
        // deterministically on every machine (never reliant on a missing
        // global git identity).
        let repo = init_repo(dir.path());
        std::fs::write(repo.join("b.txt"), "will-not-commit").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let hook = repo.join(".git/hooks/pre-commit");
            std::fs::write(&hook, "#!/bin/sh\nexit 1\n").unwrap();
            let mut perms = std::fs::metadata(&hook).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&hook, perms).unwrap();
        }
        let bare = add_bare_remote(dir.path(), &repo);
        let marker = dir.path().join("pr-ran.marker");
        let script = fake_pr_script(
            dir.path(),
            &format!(
                "echo ran > {}\necho https://example.test/pr/1",
                marker.display()
            ),
        );
        let config = CompletionStepsConfig {
            pr_command: Some(format!("{} {{branch}}", script.display())),
            ..Default::default()
        };
        let (h, task_id) = session_with_task(&m, "ordering", Some(contract(true, true, true)));
        let runner = runner(dir.path(), config, allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "ordering"))
            .await
            .unwrap();
        // Gate order: commit failed; push and pr are recorded Skipped
        // ("not attempted") and never executed.
        assert_eq!(
            report.failed_step(),
            Some(CompletionStep::Commit),
            "{report:?}"
        );
        assert_eq!(
            report
                .records
                .iter()
                .map(|r| (r.step, r.status))
                .collect::<Vec<_>>(),
            vec![
                (CompletionStep::Commit, CompletionStepOutcome::Failed),
                (CompletionStep::Push, CompletionStepOutcome::Skipped),
                (CompletionStep::Pr, CompletionStepOutcome::Skipped),
            ]
        );
        assert!(
            !marker.exists(),
            "the PR command must not run after a failed commit"
        );
        assert!(
            std::process::Command::new("git")
                .args(["rev-parse", "--verify", "main"])
                .current_dir(&bare)
                .output()
                .unwrap()
                .status
                .code()
                != Some(0),
            "the push must not run after a failed commit"
        );
        assert!(step_rows(&h, task_id).iter().all(|r| !r.detail.is_empty()));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn crash_between_steps_reopens_and_completes_idempotently() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        let cas = dir.path().join("cas");
        let repo = init_repo(dir.path());
        std::fs::write(repo.join("work.txt"), "committed before the crash\n").unwrap();
        let contract = contract(true, true, false);
        let task_id;
        {
            // Phase 1: commit succeeds, push is skipped (no remote yet) —
            // the "crash" happens between the two steps.
            let m = SessionManager::open(&store, &cas, true).unwrap();
            let (h, tid) = session_with_task(&m, "crash between steps", Some(contract));
            task_id = tid;
            let runner = runner(dir.path(), CompletionStepsConfig::default(), allow_all());
            let report = runner
                .run(&h, tid, &ctx(&repo, "crash between steps"))
                .await
                .unwrap();
            assert_eq!(
                report.outcome_of(CompletionStep::Commit),
                Some(CompletionStepOutcome::Succeeded)
            );
            assert_eq!(
                report.outcome_of(CompletionStep::Push),
                Some(CompletionStepOutcome::Skipped)
            );
        }
        // The remote comes up while the "daemon is down".
        add_bare_remote(dir.path(), &repo);
        // Phase 2: reopen the SAME store and re-run: the commit is NOT
        // re-executed (its durable row is already Succeeded) and the push
        // completes; the gate is then satisfied.
        let m = SessionManager::open(&store, &cas, true).unwrap();
        let h = m
            .get_session(faktor_core::id::SessionId::new(1))
            .unwrap()
            .unwrap();
        {
            let rev = h.task_revision(task_id).unwrap();
            let rows = h
                .ledger_completion_step_statuses(task_id.raw(), rev.raw())
                .unwrap();
            assert_eq!(rows.len(), 2, "durable rows survived the reopen: {rows:?}");
        }
        let runner = runner(dir.path(), CompletionStepsConfig::default(), allow_all());
        let report = runner
            .run(&h, task_id, &ctx(&repo, "crash between steps"))
            .await
            .unwrap();
        assert!(report.all_succeeded(), "{report:?}");
        assert_eq!(
            report.outcome_of(CompletionStep::Commit),
            Some(CompletionStepOutcome::Succeeded)
        );
        let rev = h.task_revision(task_id).unwrap();
        let rows = h
            .ledger_completion_step_statuses(task_id.raw(), rev.raw())
            .unwrap();
        assert_eq!(
            rows.iter()
                .filter(|r| r.step == CompletionStep::Commit)
                .count(),
            1,
            "the committed step is never re-recorded on replay: {rows:?}"
        );
        assert_eq!(
            rows.iter()
                .filter(|r| r.step == CompletionStep::Push)
                .count(),
            2,
            "the retried Skipped push appends the Succeeded row: {rows:?}"
        );
        assert_eq!(
            h.completion_contract_gate(task_id).unwrap(),
            CompletionContractGate::Satisfied
        );
        assert!(dir_clean(&repo));
        let head = git_output(&repo, &["rev-parse", "HEAD"]);
        let bare = dir.path().join("remote.git");
        assert_eq!(git_output(&bare, &["rev-parse", "main"]), head);
    }

    #[test]
    fn strict_config_validation_and_template_rendering() {
        assert!(CompletionStepsConfig::default().validate().is_ok());
        let bad = |command: &str| {
            CompletionStepsConfig {
                pr_command: Some(command.into()),
                ..Default::default()
            }
            .validate()
            .is_err()
        };
        assert!(bad(""));
        assert!(bad("gh pr create --head {branch}; rm -rf /"));
        assert!(bad("gh pr create --head {branch} && true"));
        assert!(bad("gh pr create --head {branch} $(whoami)"));
        assert!(bad("gh pr create --head {nope}"));
        assert!(bad("gh pr create --head main"));
        assert!(bad("gh pr create\n--head {branch}"));
        assert!(bad(&format!(
            "gh pr create --head {{branch}} {}",
            "x".repeat(MAX_PR_COMMAND_BYTES)
        )));
        assert!(CompletionStepsConfig {
            remote: "a b".into(),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(CompletionStepsConfig {
            base_branch: "a..b".into(),
            ..Default::default()
        }
        .validate()
        .is_err());
        let (program, args) = render_pr_command(
            "gh pr create --head {branch} --base {base}",
            "feat/x",
            "main",
            "origin",
        )
        .unwrap();
        assert_eq!(program, "gh");
        assert_eq!(
            args,
            vec!["pr", "create", "--head", "feat/x", "--base", "main"]
        );
        assert!(!is_egress_destination("/tmp/remote.git"));
        assert!(!is_egress_destination("file:///tmp/remote.git"));
        assert!(!is_egress_destination(r"C:\repos\remote.git"));
        assert!(is_egress_destination("https://github.com/o/r.git"));
        assert!(is_egress_destination("git@github.com:o/r.git"));
        assert_eq!(
            parse_pr_url("see https://example.test/pr/1)."),
            Some("https://example.test/pr/1".to_string())
        );
        assert!(parse_pr_url("no url here").is_none());
    }

    /// P3 typed argv: per-element placeholder substitution (spaces are legal
    /// argv characters), strict refusals, and legacy-template parity with
    /// the exact historic security checks.
    #[test]
    fn typed_argv_pr_config_substitutes_per_element_and_keeps_legacy_parity() {
        let program = "/opt/My Tools/gh";
        let args: Vec<String> = [
            "pr",
            "create",
            "--head",
            "{branch}",
            "--base",
            "{base}",
            "--title",
            "my PR title",
            "{remote}",
        ]
        .into_iter()
        .map(str::to_string)
        .collect();
        let config = CompletionStepsConfig {
            pr_program: Some(program.into()),
            pr_args: args.clone(),
            ..Default::default()
        };
        config.validate().expect("spaces are legal argv characters");
        let (rendered_program, rendered_args) =
            render_pr_argv(program, &args, "feat/x", "main", "origin").unwrap();
        assert_eq!(rendered_program, program, "the program path stays exact");
        assert_eq!(
            rendered_args,
            vec![
                "pr",
                "create",
                "--head",
                "feat/x",
                "--base",
                "main",
                "--title",
                "my PR title",
                "origin"
            ],
            "every element substitutes independently; spaces are preserved"
        );

        // Strict refusals: empty program, unknown/unclosed/stray placeholder,
        // missing {branch}, control chars, over-bound args, args without a
        // program, and both shapes configured at once.
        let bad = |program: Option<&str>, args: Vec<&str>| {
            CompletionStepsConfig {
                pr_program: program.map(str::to_string),
                pr_args: args.into_iter().map(str::to_string).collect(),
                ..Default::default()
            }
            .validate()
            .is_err()
        };
        assert!(bad(Some(""), vec!["{branch}"]));
        assert!(bad(Some("/bin/gh"), vec!["--head", "{nope}"]));
        assert!(bad(Some("/bin/gh"), vec!["--head", "{branch"]));
        assert!(bad(Some("/bin/gh"), vec!["--head", "{branch}}"]));
        assert!(bad(Some("/bin/gh"), vec!["--head", "main"]));
        assert!(bad(Some("/bin/gh"), vec!["--head", "{branch}\u{7}"]));
        let too_many: Vec<&str> = std::iter::repeat_n("{branch}", MAX_PR_ARGS + 1).collect();
        assert!(bad(Some("/bin/gh"), too_many));
        assert!(bad(None, vec!["{branch}"]));
        assert!(CompletionStepsConfig {
            pr_command: Some("gh pr create --head {branch}".into()),
            pr_program: Some("/bin/gh".into()),
            ..Default::default()
        }
        .validate()
        .is_err());

        // Legacy parity: the historic render is byte-identical and every
        // historic refusal still fires.
        let (legacy_program, legacy_args) = render_pr_command(
            "gh pr create --head {branch} --base {base}",
            "feat/x",
            "main",
            "origin",
        )
        .unwrap();
        assert_eq!(legacy_program, "gh");
        assert_eq!(
            legacy_args,
            vec!["pr", "create", "--head", "feat/x", "--base", "main"]
        );
        for invalid in [
            "",
            "gh pr create --head {branch}; rm -rf /",
            "gh pr create --head {branch} && true",
            "gh pr create --head {branch} $(whoami)",
            "gh pr create --head {nope}",
            "gh pr create --head main",
            "gh pr create\n--head {branch}",
        ] {
            assert!(
                validate_pr_command(invalid).is_err(),
                "{invalid:?} must stay refused"
            );
        }
        assert!(
            validate_pr_argv("gh pr create --head {branch}; rm -rf /", &[]).is_ok(),
            "typed argv never splits or shells: spaces/metacharacters are plain characters"
        );
    }
}
