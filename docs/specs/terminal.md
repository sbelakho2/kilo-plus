# faktor-terminal spec — ProcessSupervisor (spec §22, §23)

Crate: crates/terminal (exists, Cargo.toml configured; faktor-core, faktor-cas deps).

## Public API

```rust
pub struct SpawnConfig {
    pub cmd: String, pub args: Vec<String>, pub cwd: PathBuf,
    pub env: Vec<(String, String)>, pub owner: ProcessOwner,
    pub capture: bool,          // ring buffer + artifact capture
    pub artifact_max: usize,    // durable artifact cap (e.g. 300MB)
}
pub enum ProcessOwner { Session(SessionId), Workspace(WorkspaceId), Daemon }
pub struct ChildHandle { pub id: u64, pub pid: u32, pub owner: ProcessOwner, pub started_ms: i64 }
pub struct Reaped { pub id: u64, pub pid: u32, pub exit_code: Option<i32>, pub owner: ProcessOwner }
pub struct CommandOutput {
    pub excerpt: String,        // last 200 lines, error lines, exit code
    pub exit_code: Option<i32>,
    pub artifact: Option<String>,   // artifact://<hash> when beyond excerpt
    pub slice_hint: Option<String>, // artifact://<hash>?slice=...&len=...
    pub ring_lines: usize,
}
pub struct ProcessSupervisor { /* registry: id → ChildState(pid, owner, started, exited) */ }
impl ProcessSupervisor {
    pub fn new(cas: Arc<Cas>) -> Arc<Self>;
    /// Run to completion (bounded by deadline + cancellation): capture
    /// ring buffer (200 lines) live; when capture exceeds `capture_max`
    /// bytes it spills to a CAS artifact; output is bounded at all times.
    pub async fn run(&self, cfg: SpawnConfig, deadline: Duration,
        token: CancellationToken) -> Result<CommandOutput, Error>;
    /// Spawn detached; caller polls/reaps. Every child has a runtime owner.
    pub fn spawn(&self, cfg: SpawnConfig) -> Result<ChildHandle>;
    pub fn kill(&self, id: u64, grace_ms: u64) -> Result<()>; // SIGTERM→grace→SIGKILL, group-kill on Unix
    pub fn reap(&self) -> Vec<Reaped>;       // no zombies: collect exited children
    pub fn alive(&self) -> Vec<ChildHandle>;
    pub fn transfer(&self, id: u64, new_owner: ProcessOwner) -> Result<()>; // deliberate transfer
    pub fn kill_all_for(&self, owner: ProcessOwner) -> Vec<u64>; // session death ⇒ children die
    pub fn registered(&self) -> usize;
}
```

## Rules
- Unix: `std::process::Command` with `process_group(0)` (setsid) so kill kills the whole tree; SIGTERM, grace period (default 2s), then SIGKILL to the group. Windows: `#[cfg(windows)]` uses Job Objects via `windows-sys` (add dep, windows only); kill-on-close. Keep the crate compiling on macOS/Linux without windows-sys.
- `run()`: stdout+stderr piped; a live ring buffer keeps the last 200 lines; total captured bytes bounded by `artifact_max` (default 100MB); beyond that → spill to CAS artifact, ring keeps the tail. Never store unbounded output in RAM. `CommandOutput.excerpt` = last 200 lines + lines matching /error|panic|failed/i (bounded) + exit code.
- Deadline exceeded or cancelled: kill the process group, return Error::timeout/cancelled, record Reaped with the kill.
- No orphans: when the supervisor drops a session's processes are NOT auto-killed (daemon owns them) — but `kill_all_for(Session)` must kill the group. Zombie reaping is explicit (`reap()`).
- 300MB log adversarial test: stream a synthetic 300MB of output; RAM stays bounded (assert excerpt small, artifact present, ring ≤ 200 lines).
- For tests use real `sleep`/`sh -c` processes (fast). Long tests: `#[ignore]` `[soak]`.

## Adversarial tests (name every one)
1. ring_buffer_caps_at_200_lines (write 10k lines)
2. huge_output_spills_to_cas_and_ram_stays_bounded (300MB synthesized; assert artifact + excerpt < 64KB)
3. kill_terminates_process_group (spawn `sh -c "sleep 30 & sleep 30"`, kill, verify no survivor via /bin/ps -p)
4. deadline_kills_group_and_returns_timeout
5. cancellation_kills_group_and_returns_cancelled
6. reap_collects_exit_codes_and_no_zombies (spawn 10 quick `true`/`false`, reap all, no leftover)
7. kill_all_for_session_kills_children
8. transfer_changes_owner_and_survives
9. unknown_id_operations_are_not_found (kill/reap/transfer on bogus id)
10. exit_code_propagation (true→0, false→1, `exit 42`→42)
11. malicious_command_vector (args with spaces/shell metachars run as exec args, not shell: `sh -c 'echo "$1"' x '; rm -rf'` — arg must be passed literally)
12. missing_command_is_not_found (spawn /nonexistent)
13. spawn_before_run_races (spawn 50 concurrent, all get unique ids, all reapable)
14. stderr_and_stdout_both_captured
15. artifact_roundtrip_readable_via_cas (excerpt + slice_hint point at a CAS blob that roundtrips)

Build/test: `cargo build -p faktor-terminal`, `cargo test -p faktor-terminal` zero warnings; `cargo clippy -p faktor-terminal --all-targets` no errors. Do NOT modify other crates. Do NOT commit.
