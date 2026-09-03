//! SQLite persistence done correctly: WAL, single logical writer + genuinely
//! bounded reader pool, busy timeout, explicit transactional migrations,
//! integrity checks, automatic backups.
//!
//! Large blobs never live here — they go to the CAS; SQLite stores hashes.
//! Message/part rows store JSON payloads so the store stays protocol-agnostic.
//!
//! # Reader pool bound
//!
//! `read()` acquires a semaphore permit before touching a connection, so at
//! most `READER_POOL` (4) connections exist concurrently; a 20-reader storm
//! therefore uses at most 4 connections and the remaining callers block on
//! the permit, bounded by the busy timeout (`StoreError::Busy`). The pool is
//! a concurrency limit, not merely a retention limit.
//!
//! # Async boundary
//!
//! This crate is intentionally synchronous; do not introduce tokio here.
//! Async callers must wrap store calls in `spawn_blocking` (the daemon
//! already does this via the session layer) so the runtime is never blocked.
//!
//! # Stability rule
//!
//! Every value read back from the database is parsed fallibly: corrupt or
//! version-skewed rows surface as `StoreError::Corrupt` (or `Sqlite`) —
//! never a panic. `unwrap`/`expect` appear only where the input is provably
//! constructed in-process this session (each site is commented).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};

use kilop_core::event::{Event, EventKind, JournalInvariants};
use kilop_core::id::{EventSeq, OpId, SessionId, TaskId, WorkspaceId, WorktreeId};
use kilop_core::state::{AgentState, SessionLifecycle};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("store is corrupted: integrity check failed with {0:?}")]
    Corrupt(Vec<String>),
    #[error("event sequence gap or duplicate detected at {0}")]
    SeqViolation(u64),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("reader pool busy: {0}")]
    Busy(String),
    #[error("migration failed: {0}")]
    Migration(String),
}

pub type StoreResult<T> = Result<T, StoreError>;

/// Max concurrent read connections. This is a concurrency limit (semaphore
/// permits), not just a retention limit.
const READER_POOL: usize = 4;

/// How long `read()` waits for a permit before failing with `Busy`. Matches
/// the SQLite `busy_timeout` pragma (5s), so pool-level and engine-level
/// waits behave consistently.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// A counting semaphore on stable std (there is no `std::sync::Semaphore` on
/// stable Rust as of 1.98): `acquire_timeout` blocks on a condvar until a
/// permit frees or the deadline passes. Readers hold a `Permit` for the
/// whole borrow, which is what caps live read connections at `READER_POOL`.
#[derive(Debug)]
struct Semaphore {
    permits: Mutex<usize>,
    available: Condvar,
}

/// RAII permit; releases one permit and wakes one waiter on drop.
#[derive(Debug)]
struct Permit(Arc<Semaphore>);

impl Drop for Permit {
    fn drop(&mut self) {
        let mut p = self.0.permits.lock().unwrap_or_else(|e| e.into_inner());
        *p += 1;
        self.0.available.notify_one();
    }
}

impl Semaphore {
    fn new(permits: usize) -> Self {
        Self {
            permits: Mutex::new(permits),
            available: Condvar::new(),
        }
    }

    /// Block until a permit is free or `deadline` passes (`Busy`).
    fn acquire_timeout(self: &Arc<Self>, deadline: Instant) -> StoreResult<Permit> {
        let mut p = self
            .permits
            .lock()
            .map_err(|_| StoreError::Migration("reader pool semaphore poisoned".into()))?;
        loop {
            if *p > 0 {
                *p -= 1;
                return Ok(Permit(Arc::clone(self)));
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(StoreError::Busy(format!(
                    "no reader permit within {}ms ({} readers already active)",
                    BUSY_TIMEOUT.as_millis(),
                    READER_POOL
                )));
            }
            let (guard, _) = self
                .available
                .wait_timeout(p, deadline - now)
                .map_err(|_| StoreError::Migration("reader pool semaphore poisoned".into()))?;
            p = guard;
        }
    }
}

/// Bounded pool of idle read connections + the semaphore that caps how many
/// readers may borrow one at a time.
#[derive(Debug)]
struct ReaderPool {
    conns: Mutex<Vec<Connection>>,
    sem: Arc<Semaphore>,
    /// Connections ever opened (doctor/test probe: proves the cap held).
    created: AtomicU64,
}

impl ReaderPool {
    fn new() -> Self {
        Self {
            conns: Mutex::new(Vec::with_capacity(READER_POOL)),
            sem: Arc::new(Semaphore::new(READER_POOL)),
            created: AtomicU64::new(0),
        }
    }
}

/// A borrowed read connection; returned to the pool on drop. The semaphore
/// permit is held for the borrow's lifetime, which is what bounds concurrent
/// readers at `READER_POOL`.
pub struct ReadConn {
    conn: Option<Connection>,
    pool: Arc<ReaderPool>,
    _permit: Permit,
}

impl ReadConn {
    pub fn get(&self) -> &Connection {
        // In-process invariant: a ReadConn is handed out exactly once and its
        // connection is only `take`n by Drop, so a live ReadConn always has
        // its connection. Never reachable from DB state.
        self.conn.as_ref().expect("read conn already returned")
    }
}

impl std::ops::Deref for ReadConn {
    type Target = Connection;

    fn deref(&self) -> &Connection {
        self.get()
    }
}

impl Drop for ReadConn {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            if let Ok(mut conns) = self.pool.conns.lock() {
                if conns.len() < READER_POOL {
                    conns.push(conn);
                    // The semaphore permit is released right after this
                    // method, via the `_permit` field's drop.
                    return;
                }
            }
            drop(conn);
        }
    }
}

/// The daemon's durable store. `write` takes a single writer lock; `read`
/// borrows a connection from a small pool (SQLite WAL allows concurrent
/// readers). All mutations happen inside explicit transactions.
#[derive(Debug)]
pub struct Store {
    root: PathBuf,
    writer: Mutex<Connection>,
    pool: Arc<ReaderPool>,
}

#[derive(Debug, Clone)]
pub struct SessionRow {
    pub id: SessionId,
    pub workspace_id: WorkspaceId,
    /// Durable worktree identity of the session (v8+). The standalone
    /// default is 1/1 (the session's worktree/task ids are adopted
    /// deliberately when a WorktreeManager-created worktree takes it over).
    pub worktree_id: WorktreeId,
    /// Durable task identity of the session (v8+); standalone default 1.
    pub task_id: TaskId,
    pub title: String,
    pub provider: String,
    pub model: String,
    pub state: AgentState,
    pub lifecycle: kilop_core::state::SessionLifecycle,
    pub created_ms: i64,
    pub updated_ms: i64,
}

/// One atomic session transition: verify expected lifecycle/state, move
/// lifecycle+state, and append the journal event in a SINGLE SQLite
/// transaction. A crash can never leave the lifecycle and the journal
/// contradictory (the old two-step update-then-append had exactly that
/// window). `expected_* = None` skips the corresponding check.
#[derive(Debug, Clone)]
pub struct SessionTransition {
    /// When `Some`, the session row must have exactly this lifecycle or the
    /// transition fails with `StoreError::Conflict` and writes nothing.
    pub expected_lifecycle: Option<SessionLifecycle>,
    /// When `Some`, the lifecycle is updated to this value.
    pub new_lifecycle: Option<SessionLifecycle>,
    /// When `Some`, the session row must have exactly this state.
    pub expected_state: Option<AgentState>,
    /// The state the session row AND the journal event land on.
    pub new_state: AgentState,
    /// Journal event kind appended in the same transaction.
    pub event_kind: EventKind,
    pub event_payload: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct MessageRow {
    pub id: i64,
    pub session_id: SessionId,
    pub seq: i64,
    pub role: String,
    pub data: serde_json::Value,
    pub created_ms: i64,
}

#[derive(Debug, Clone)]
pub struct PartRow {
    pub id: i64,
    pub message_id: i64,
    pub kind: String,
    pub data: serde_json::Value,
    pub created_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolRunRow {
    pub id: i64,
    pub session_id: SessionId,
    pub op_id: OpId,
    pub tool: String,
    pub args: serde_json::Value,
    pub status: String,
    pub started_ms: i64,
    pub ended_ms: Option<i64>,
    pub effect_status: String,
    pub recovery: serde_json::Value,
    pub expected_hash: Option<String>,
    /// Durable replay descriptor (v7+): the stored invocation crash recovery
    /// may re-execute ONCE for idempotent tools. NULL on legacy rows.
    pub replay_descriptor: Option<serde_json::Value>,
    /// Physical attempt counter of the SAME logical operation (v7+): the
    /// original run is attempt 0; each crash recovery replay bumps it.
    pub attempt: i64,
    /// Durable workspace-write postcondition (v7+): `{workspace_id,
    /// worktree_id, relative_path, expected_hash}` — the hash of the ACTUAL
    /// bytes as written, recorded by the tool at execution end. NULL until
    /// the tool reports it (or for non-write tools).
    pub postcondition: Option<serde_json::Value>,
}

/// One durable logical-turn record (v7). Created transactionally when a
/// prompt is admitted as the ACTIVE logical turn (submit_prompt / queue
/// admission); it fixes the turn's exact operation identity and effective
/// model/provider envelope so crash recovery resumes the SAME turn with the
/// SAME identity instead of synthesizing a fresh operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnRecordRow {
    pub id: i64,
    pub session_id: SessionId,
    pub turn_op_id: OpId,
    /// Durable queue seq when the turn was admitted from the prompt queue.
    pub queue_seq: Option<i64>,
    /// Durable message seq of the materialized user prompt.
    pub prompt_message_id: Option<i64>,
    pub effective_provider: String,
    pub effective_model: String,
    /// Reasoning mode / variant of the logical turn (NULL when unset).
    pub variant: Option<String>,
    /// Tool-call parsing mode of the logical turn (NULL until driven).
    pub tool_mode: Option<String>,
    pub started_at: i64,
    /// active | completed | cancelled | failed
    pub status: String,
    pub updated_ms: i64,
}

pub const TURN_RECORD_ACTIVE: &str = "active";
pub const TURN_RECORD_COMPLETED: &str = "completed";
pub const TURN_RECORD_CANCELLED: &str = "cancelled";
pub const TURN_RECORD_FAILED: &str = "failed";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointRow {
    pub id: i64,
    pub session_id: SessionId,
    pub sequence: i64,
    pub path: String,
    pub before_hash: String,
    pub after_hash: String,
    /// CAS hash of the AFTER-content blob (v3+). NULL on rows recorded
    /// before the column existed: redo/diff refuse those honestly.
    pub after_cas_hash: Option<String>,
    /// Per-side EXISTENCE flags (v6+). A hash alone cannot distinguish a
    /// missing file from an empty one (both sides of a missing→empty write
    /// hash to blake3("")). When a side `exists` is false its hash column is
    /// the empty string (no content exists to address).
    ///
    /// Backward compatibility: pre-v6 rows carry NO marker, so these read as
    /// true — old rows were only recorded for real files (the caller had
    /// read/hashed the content on both sides).
    pub before_exists: bool,
    pub after_exists: bool,
    pub created_ms: i64,
    pub restored_ms: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorktreeRow {
    pub id: i64,
    pub workspace_id: WorkspaceId,
    pub path: String,
    pub branch: String,
    pub active: bool,
}

pub struct QueuedPrompt {
    pub queue_seq: i64,
    pub op_id: OpId,
    pub prompt: String,
    pub files: Vec<String>,
    pub model: Option<String>,
    pub variant: Option<String>,
    pub agent: Option<String>,
    pub status: String,
    pub requested_at: i64,
}

/// Result of the atomic claim/admission of the queue head.
#[derive(Debug, Clone)]
pub struct AdmittedPrompt {
    pub queue_seq: i64,
    pub op_id: OpId,
    pub prompt: String,
    pub files: Vec<String>,
    pub model: Option<String>,
    pub variant: Option<String>,
    pub agent: Option<String>,
    /// Message seq of the materialized user message (== the admission
    /// journal event seq).
    pub message_seq: i64,
}

impl Store {
    /// Open (creating if needed) and migrate. `integrity_check: true` runs a
    /// full integrity check before use and refuses to open a corrupt store.
    pub fn open(root: impl Into<PathBuf>, integrity_check: bool) -> StoreResult<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        let db_path = root.join("kilo-plus.db");

        let mut conn = Connection::open(&db_path)?;
        configure(&conn)?;
        migrate(&mut conn)?;

        if integrity_check {
            let issues = check_integrity(&conn)?;
            if !issues.is_empty() {
                return Err(StoreError::Corrupt(issues));
            }
        }

        let writer = Mutex::new(conn);
        let pool = Arc::new(ReaderPool::new());
        Ok(Self { root, writer, pool })
    }

    pub fn path(&self) -> PathBuf {
        self.root.join("kilo-plus.db")
    }

    fn write(&self) -> MutexGuard<'_, Connection> {
        // In-process invariant: the writer mutex is only poisoned by a panic
        // in a query (a bug, not corrupt data), so unwinding is correct.
        self.writer.lock().expect("store writer poisoned")
    }

    /// Borrow a read connection. A semaphore permit is acquired first, so at
    /// most `READER_POOL` connections exist concurrently: 20 simultaneous
    /// readers use at most 4 connections and the rest wait on the permit,
    /// bounded by the busy timeout (`StoreError::Busy`).
    fn read(&self) -> StoreResult<ReadConn> {
        let permit = self
            .pool
            .sem
            .acquire_timeout(Instant::now() + BUSY_TIMEOUT)?;
        let mut conns = self
            .pool
            .conns
            .lock()
            .map_err(|_| StoreError::Migration("reader pool poisoned".into()))?;
        let conn = match conns.pop() {
            Some(c) => c,
            None => {
                let c = Connection::open_with_flags(
                    self.path(),
                    rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                        | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
                )?;
                configure(&c)?;
                self.pool.created.fetch_add(1, Ordering::Relaxed);
                c
            }
        };
        Ok(ReadConn {
            conn: Some(conn),
            pool: self.pool.clone(),
            _permit: permit,
        })
    }

    /// Idle connections currently in the pool; at most `READER_POOL`.
    /// Test probe.
    #[cfg(test)]
    pub(crate) fn reader_pool_len(&self) -> usize {
        self.pool.conns.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// Read connections ever opened since `open`; with the semaphore this
    /// never exceeds `READER_POOL` even under heavy contention.
    /// Test probe.
    #[cfg(test)]
    pub(crate) fn connections_created(&self) -> u64 {
        self.pool.created.load(Ordering::Relaxed)
    }

    // ---------------------------------------------------------------- workspaces

    pub fn create_workspace(&self, root: &str) -> StoreResult<WorkspaceId> {
        let conn = self.write();
        conn.execute(
            "INSERT OR IGNORE INTO workspace(root, created_ms) VALUES (?1, ?2)",
            params![root, now_ms()],
        )?;
        let id: i64 = conn.query_row(
            "SELECT id FROM workspace WHERE root = ?1",
            params![root],
            |r| r.get(0),
        )?;
        Ok(WorkspaceId::new(id as u64))
    }

    /// The recorded root path of a workspace; `None` when the workspace id is
    /// unknown (the revert/diff wire surface needs the on-disk root to open
    /// the file service handle).
    pub fn workspace_root(&self, id: WorkspaceId) -> StoreResult<Option<String>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT root FROM workspace WHERE id = ?1",
                params![id.raw() as i64],
                |r| r.get(0),
            )
            .optional()?;
        Ok(out)
    }

    // ---------------------------------------------------------------- sessions

    pub fn create_session(
        &self,
        workspace_id: WorkspaceId,
        title: &str,
        provider: &str,
        model: &str,
    ) -> StoreResult<SessionRow> {
        let conn = self.write();
        let now = now_ms();
        conn.execute(
            "INSERT INTO session(workspace_id, title, provider, model, state, lifecycle, created_ms, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, 'open', ?6, ?6)",
            params![
                workspace_id.raw() as i64,
                title,
                provider,
                model,
                // In-process constructed enum: serialization of a unit
                // variant can never fail.
                serde_json::to_string(&AgentState::Idle).unwrap(),
                now
            ],
        )?;
        let id: i64 = conn.last_insert_rowid();
        // Seed the journal with SessionCreated so every session starts at seq 1.
        // The session row and its seed event are one transaction.
        let tx = conn.unchecked_transaction()?;
        self.insert_event_locked(
            &tx,
            SessionId::new(id as u64),
            None,
            EventKind::SessionCreated,
            AgentState::Idle,
            now,
            Some(serde_json::json!({ "title": title, "provider": provider, "model": model })),
        )?;
        tx.commit()?;
        Ok(
            match self.get_session_locked(&conn, SessionId::new(id as u64))? {
                Some(row) => row,
                None => {
                    return Err(StoreError::Corrupt(vec![
                        "just-created session not readable back".into(),
                    ]))
                }
            },
        )
    }

    pub fn get_session(&self, id: SessionId) -> StoreResult<Option<SessionRow>> {
        let conn = self.read()?;
        let row = self.get_session_locked(&conn, id)?;
        Ok(row)
    }

    fn get_session_locked(
        &self,
        conn: &Connection,
        id: SessionId,
    ) -> StoreResult<Option<SessionRow>> {
        let mut stmt = conn.prepare(
            "SELECT id, workspace_id, worktree_id, task_id, title, provider, model, state, lifecycle, created_ms, updated_ms
             FROM session WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id.raw() as i64])?;
        match rows.next()? {
            Some(row) => Ok(Some(session_row_map(row)?)),
            None => Ok(None),
        }
    }

    pub fn list_sessions(&self, workspace_id: Option<WorkspaceId>) -> StoreResult<Vec<SessionRow>> {
        let conn = self.read()?;
        let mut stmt = match workspace_id {
            Some(_) => conn.prepare(
                "SELECT id, workspace_id, worktree_id, task_id, title, provider, model, state, lifecycle, created_ms, updated_ms
                 FROM session WHERE workspace_id = ?1 ORDER BY updated_ms DESC",
            )?,
            None => conn.prepare(
                "SELECT id, workspace_id, worktree_id, task_id, title, provider, model, state, lifecycle, created_ms, updated_ms
                 FROM session ORDER BY updated_ms DESC",
            )?,
        };
        let mut rows = match workspace_id {
            Some(w) => stmt.query(params![w.raw() as i64])?,
            None => stmt.query([])?,
        };
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(session_row_map(row)?);
        }
        Ok(out)
    }

    /// Durably adopt a worktree/task identity (v8). The standalone session
    /// default is 1/1; WorktreeManager-created worktrees call this to make
    /// the identity durable so every later tool call (and crash recovery)
    /// carries the REAL worktree/task ids. The journal is intentionally
    /// untouched: adoption is identity bookkeeping, not a turn transition.
    pub fn adopt_session_identity(
        &self,
        id: SessionId,
        worktree_id: WorktreeId,
        task_id: TaskId,
    ) -> StoreResult<()> {
        if worktree_id.raw() == 0 || task_id.raw() == 0 {
            return Err(StoreError::Migration(
                "worktree/task ids must be non-zero".into(),
            ));
        }
        let conn = self.write();
        let n = conn.execute(
            "UPDATE session SET worktree_id = ?2, task_id = ?3, updated_ms = ?4 WHERE id = ?1",
            params![
                id.raw() as i64,
                worktree_id.raw() as i64,
                task_id.raw() as i64,
                now_ms()
            ],
        )?;
        if n == 0 {
            return Err(StoreError::Migration(format!(
                "adopt_session_identity: session {id} does not exist"
            )));
        }
        Ok(())
    }

    pub fn set_session_lifecycle(
        &self,
        id: SessionId,
        lifecycle: kilop_core::state::SessionLifecycle,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE session SET lifecycle = ?2, updated_ms = ?3 WHERE id = ?1",
            params![
                id.raw() as i64,
                // In-process constructed enum (see create_session).
                serde_json::to_string(&lifecycle).unwrap(),
                now_ms()
            ],
        )?;
        Ok(())
    }

    pub fn set_session_state(&self, id: SessionId, state: AgentState) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE session SET state = ?2, updated_ms = ?3 WHERE id = ?1",
            params![
                id.raw() as i64,
                // In-process constructed enum (see create_session).
                serde_json::to_string(&state).unwrap(),
                now_ms()
            ],
        )?;
        Ok(())
    }

    /// Single conditional lifecycle UPDATE (`WHERE lifecycle = expected`).
    /// Returns whether a row was updated. Used by prompt auto-resume
    /// (`Suspended -> Open`); the journal is intentionally untouched there —
    /// resuming on prompt is not a new event.
    pub fn set_lifecycle_if(
        &self,
        id: SessionId,
        expected: SessionLifecycle,
        new: SessionLifecycle,
    ) -> StoreResult<bool> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE session SET lifecycle = ?3, updated_ms = ?4
             WHERE id = ?1 AND lifecycle = ?2",
            params![
                id.raw() as i64,
                // In-process constructed enums (see create_session).
                serde_json::to_string(&expected).unwrap(),
                serde_json::to_string(&new).unwrap(),
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// Durable session-title update (session.update, P1). Bumps
    /// `updated_ms` so list ordering reflects the rename. Returns whether a
    /// row was updated (callers check existence first for a clean
    /// NotFound). The journal is intentionally untouched: the title is
    /// session metadata, not a state-machine transition.
    pub fn update_session_title(&self, id: SessionId, title: &str) -> StoreResult<bool> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE session SET title = ?2, updated_ms = ?3 WHERE id = ?1",
            params![id.raw() as i64, title, now_ms()],
        )?;
        Ok(n > 0)
    }

    /// ONE SQLite transaction: read the session row, verify
    /// `expected_lifecycle`/`expected_state` (mismatch -> `Conflict`, nothing
    /// written), update lifecycle+state+updated_ms, append the event with the
    /// next gapless per-session seq, commit. Returns the event seq.
    ///
    /// This is the atomic guard for lifecycle+event transitions: the session
    /// layer's `end_session`/`suspend`/`resume` call it so a crash between
    /// "update lifecycle" and "append event" can never be observed.
    pub fn transition_session(
        &self,
        session_id: SessionId,
        op_id: Option<OpId>,
        t: SessionTransition,
    ) -> StoreResult<EventSeq> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        // (a) read the session row inside the transaction.
        let Some(row) = self.get_session_locked(&tx, session_id)? else {
            return Err(StoreError::Conflict(format!(
                "session {session_id} does not exist; cannot transition"
            )));
        };
        // (b) verify the expected values; mismatch aborts with nothing written.
        if let Some(expected) = t.expected_lifecycle {
            if row.lifecycle != expected {
                return Err(StoreError::Conflict(format!(
                    "session {session_id} lifecycle is {:?}, expected {:?}",
                    row.lifecycle, expected
                )));
            }
        }
        if let Some(expected) = t.expected_state {
            if row.state != expected {
                return Err(StoreError::Conflict(format!(
                    "session {session_id} state is {:?}, expected {:?}",
                    row.state, expected
                )));
            }
        }
        // (c) update lifecycle+state+updated_ms.
        let now = now_ms();
        // In-process constructed enums (see create_session).
        let state_json = serde_json::to_string(&t.new_state).unwrap();
        match t.new_lifecycle {
            Some(lifecycle) => {
                tx.execute(
                    "UPDATE session SET lifecycle = ?2, state = ?3, updated_ms = ?4 WHERE id = ?1",
                    params![
                        session_id.raw() as i64,
                        // In-process constructed enum (see create_session).
                        serde_json::to_string(&lifecycle).unwrap(),
                        state_json,
                        now
                    ],
                )?;
            }
            None => {
                tx.execute(
                    "UPDATE session SET state = ?2, updated_ms = ?3 WHERE id = ?1",
                    params![session_id.raw() as i64, state_json, now],
                )?;
            }
        }
        // (d) append the event with the next gapless seq (shared insert path).
        let seq = self.insert_event_locked(
            &tx,
            session_id,
            op_id,
            t.event_kind,
            t.new_state,
            now,
            t.event_payload,
        )?;
        // (e) commit: lifecycle change and event are durable together.
        tx.commit()?;
        Ok(seq)
    }

    // ---------------------------------------------------------------- event journal

    /// Append an event with the next per-session sequence number, atomically.
    /// Duplicate/gap sequences are impossible under the transaction; the
    /// primary key enforces it structurally.
    pub fn append_event(
        &self,
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
    ) -> StoreResult<EventSeq> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let seq = self.insert_event_locked(&tx, session_id, op_id, kind, state, ts_ms, payload)?;
        tx.commit()?;
        Ok(seq)
    }

    /// The shared gapless-seq event insert path. Runs inside the CALLER'S
    /// transaction so `append_event` and `transition_session` are one atomic
    /// unit (a nested transaction here would silently demote to a savepoint).
    // Fixed-arity journal insert; the parameter list is a stable call
    // contract used across the workspace.
    #[allow(clippy::too_many_arguments)]
    fn insert_event_locked(
        &self,
        tx: &rusqlite::Transaction<'_>,
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
    ) -> StoreResult<EventSeq> {
        // Serialize appends per session so seq computation is race-free.
        // (The store writer lock already serializes; the per-session query is
        // a second belt for future multi-writer refactors.)
        let prev: Option<i64> = tx.query_row(
            "SELECT MAX(seq) FROM event WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        let seq = JournalInvariants::next_seq(prev.map(|p| EventSeq::new(p as u64)));
        let ts = JournalInvariants::monotonic_ts(
            prev.map(|_| {
                // Use the previous event's ts for monotonicity.
                tx.query_row(
                    "SELECT ts_ms FROM event WHERE session_id = ?1 AND seq = (SELECT MAX(seq) FROM event WHERE session_id = ?1)",
                    params![session_id.raw() as i64],
                    |r| r.get::<_, i64>(0),
                ).unwrap_or(0)
            }),
            ts_ms,
        );
        tx.execute(
            "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                seq.raw() as i64,
                session_id.raw() as i64,
                op_id.map(|o| o.raw() as i64),
                kind_name(kind),
                // In-process constructed enum (see create_session).
                serde_json::to_string(&state).unwrap(),
                ts,
                payload.map(|p| p.to_string()),
            ],
        )?;
        tx.execute(
            "UPDATE session SET state = ?2, updated_ms = ?3 WHERE id = ?1",
            params![
                session_id.raw() as i64,
                // In-process constructed enum (see create_session).
                serde_json::to_string(&state).unwrap(),
                ts
            ],
        )?;
        Ok(seq)
    }

    /// Events strictly after `after_seq` (SSE resume cursor).
    pub fn events_after(
        &self,
        session_id: SessionId,
        after_seq: EventSeq,
    ) -> StoreResult<Vec<Event>> {
        self.events_range(session_id, after_seq.raw() + 1, None)
    }

    pub fn events_range(
        &self,
        session_id: SessionId,
        from_seq: u64,
        limit: Option<u64>,
    ) -> StoreResult<Vec<Event>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT seq, session_id, op_id, kind, state, ts_ms, payload FROM event
             WHERE session_id = ?1 AND seq >= ?2 ORDER BY seq ASC LIMIT ?3",
        )?;
        let limit = limit.unwrap_or(u64::MAX);
        let mut rows = stmt.query(params![
            session_id.raw() as i64,
            from_seq as i64,
            limit as i64
        ])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(event_map(row, session_id)?);
        }
        Ok(out)
    }

    pub fn last_event_seq(&self, session_id: SessionId) -> StoreResult<Option<EventSeq>> {
        let conn = self.read()?;
        let out = conn.query_row(
            "SELECT MAX(seq) FROM event WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get::<_, Option<i64>>(0),
        )?;
        Ok(out.map(|o| EventSeq::new(o as u64)))
    }

    // ---------------------------------------------------------------- messages

    /// Paging is fundamental: the webview sees the latest page immediately and
    /// earlier pages stream on demand. Returns newest-first page.
    pub fn messages_before(
        &self,
        session_id: SessionId,
        before_seq: Option<i64>,
        limit: u64,
    ) -> StoreResult<Vec<MessageRow>> {
        let conn = self.read()?;
        let (sql, params) = match before_seq {
            Some(b) => (
                "SELECT id, session_id, seq, role, data, created_ms FROM message
                 WHERE session_id = ?1 AND seq < ?2 ORDER BY seq DESC LIMIT ?3",
                vec![session_id.raw() as i64, b, limit as i64],
            ),
            None => (
                "SELECT id, session_id, seq, role, data, created_ms FROM message
                 WHERE session_id = ?1 ORDER BY seq DESC LIMIT ?2",
                vec![session_id.raw() as i64, limit as i64],
            ),
        };
        let mut stmt = conn.prepare(sql)?;
        let mut rows = stmt.query(rusqlite::params_from_iter(params))?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(message_map(row)?);
        }
        Ok(out)
    }

    pub fn put_message(
        &self,
        session_id: SessionId,
        seq: i64,
        role: &str,
        data: serde_json::Value,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO message(session_id, seq, role, data, created_ms) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![session_id.raw() as i64, seq, role, data.to_string(), now_ms()],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn put_part(
        &self,
        message_id: i64,
        kind: &str,
        data: serde_json::Value,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO part(message_id, kind, data, created_ms) VALUES (?1, ?2, ?3, ?4)",
            params![message_id, kind, data.to_string(), now_ms()],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn parts_of(&self, message_id: i64) -> StoreResult<Vec<PartRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, message_id, kind, data, created_ms FROM part WHERE message_id = ?1 ORDER BY id ASC",
        )?;
        let mut rows = stmt.query(params![message_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(part_map(row)?);
        }
        Ok(out)
    }

    pub fn message_count(&self, session_id: SessionId) -> StoreResult<i64> {
        let conn = self.read()?;
        let out = conn.query_row(
            "SELECT COUNT(*) FROM message WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        Ok(out)
    }

    /// `created_ms` of one message by (session, seq); `None` when the message
    /// does not exist. The revert wire surface uses it as the checkpoint
    /// cutoff: only checkpoints recorded at or before the message may roll
    /// back to it.
    pub fn message_created_ms(&self, session_id: SessionId, seq: i64) -> StoreResult<Option<i64>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT created_ms FROM message WHERE session_id = ?1 AND seq = ?2",
                params![session_id.raw() as i64, seq],
                |r| r.get(0),
            )
            .optional()?;
        Ok(out)
    }

    /// Durable single-message removal (deleteMessage, P1): the message row
    /// AND its part rows are deleted in ONE transaction — a crash can never
    /// leave orphan parts (part rows reference the message row by foreign
    /// key, so the order is structural, not incidental). Message sequences
    /// are STABLE: rows are removed, nothing is renumbered, and the paging
    /// projection simply skips the hole. Returns whether a message row
    /// existed (false = nothing deleted). The journal is intentionally
    /// untouched: it is the durable log of what happened; deleting a
    /// conversation row is not a state-machine event.
    pub fn delete_message(&self, session_id: SessionId, seq: i64) -> StoreResult<bool> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let id: Option<i64> = tx
            .query_row(
                "SELECT id FROM message WHERE session_id = ?1 AND seq = ?2",
                params![session_id.raw() as i64, seq],
                |r| r.get(0),
            )
            .optional()?;
        let Some(id) = id else {
            return Ok(false);
        };
        tx.execute("DELETE FROM part WHERE message_id = ?1", params![id])?;
        tx.execute(
            "DELETE FROM message WHERE session_id = ?1 AND seq = ?2",
            params![session_id.raw() as i64, seq],
        )?;
        tx.commit()?;
        Ok(true)
    }

    // ---------------------------------------------------------------- task ledger

    pub fn get_task_ledger(&self, session_id: SessionId) -> StoreResult<Option<serde_json::Value>> {
        let conn = self.read()?;
        let raw: Option<String> = conn
            .query_row(
                "SELECT ledger FROM task WHERE session_id = ?1 ORDER BY updated_ms DESC LIMIT 1",
                params![session_id.raw() as i64],
                |r| r.get(0),
            )
            .optional()?;
        match raw {
            Some(s) => Ok(Some(parse_json(
                &format!("task ledger for session {session_id}"),
                &s,
            )?)),
            None => Ok(None),
        }
    }

    pub fn put_task_ledger(
        &self,
        session_id: SessionId,
        ledger: serde_json::Value,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "DELETE FROM task WHERE session_id = ?1",
            params![session_id.raw() as i64],
        )?;
        conn.execute(
            "INSERT INTO task(session_id, ledger, updated_ms) VALUES (?1, ?2, ?3)",
            params![session_id.raw() as i64, ledger.to_string(), now_ms()],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------------- tool runs

    #[allow(clippy::too_many_arguments)]
    pub fn start_tool_run(
        &self,
        session_id: SessionId,
        op_id: OpId,
        tool: &str,
        args: serde_json::Value,
        recovery: serde_json::Value,
        expected_hash: Option<String>,
        replay_descriptor: Option<serde_json::Value>,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO tool_run(session_id, op_id, tool, args, status, started_ms, effect_status, recovery, expected_hash, replay_descriptor)
             VALUES (?1, ?2, ?3, ?4, 'running', ?5, 'unknown', ?6, ?7, ?8)",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                tool,
                args.to_string(),
                now_ms(),
                recovery.to_string(),
                expected_hash,
                replay_descriptor.map(|d| d.to_string()),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Record the workspace-write postcondition a tool reported at execution
    /// end (v7): recovery verifies the CURRENT file bytes against it through
    /// the workspace file service — never a hash inferred from args JSON.
    /// Only a still-running row may be annotated (loud otherwise).
    #[allow(clippy::too_many_arguments)]
    pub fn record_tool_postcondition(
        &self,
        session_id: SessionId,
        op_id: OpId,
        postcondition: &serde_json::Value,
    ) -> StoreResult<()> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE tool_run SET postcondition = ?3
             WHERE session_id = ?1 AND op_id = ?2 AND status = 'running'",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                postcondition.to_string()
            ],
        )?;
        if n == 0 {
            return Err(StoreError::Migration(
                "record_tool_postcondition: no running row".into(),
            ));
        }
        Ok(())
    }

    /// Bump the physical-attempt counter of one still-running tool run (v7:
    /// a crash-recovery replay is a NEW PHYSICAL attempt of the SAME logical
    /// operation). Loud when the row is not running.
    pub fn bump_tool_run_attempt(&self, session_id: SessionId, op_id: OpId) -> StoreResult<i64> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let n = tx.execute(
            "UPDATE tool_run SET attempt = attempt + 1
             WHERE session_id = ?1 AND op_id = ?2 AND status = 'running'",
            params![session_id.raw() as i64, op_id.raw() as i64],
        )?;
        if n == 0 {
            return Err(StoreError::Migration(
                "bump_tool_run_attempt: no running row".into(),
            ));
        }
        let attempt: i64 = tx.query_row(
            "SELECT attempt FROM tool_run WHERE session_id = ?1 AND op_id = ?2",
            params![session_id.raw() as i64, op_id.raw() as i64],
            |r| r.get(0),
        )?;
        tx.commit()?;
        Ok(attempt)
    }

    pub fn finish_tool_run(
        &self,
        session_id: SessionId,
        op_id: OpId,
        status: &str,
        effect_status: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE tool_run SET status = ?3, effect_status = ?4, ended_ms = ?5
             WHERE session_id = ?1 AND op_id = ?2",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                status,
                effect_status,
                now_ms()
            ],
        )?;
        if n == 0 {
            return Err(StoreError::Migration(
                "finish_tool_run: no matching row".into(),
            ));
        }
        Ok(())
    }

    pub fn set_tool_run_effect(
        &self,
        session_id: SessionId,
        op_id: OpId,
        effect_status: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE tool_run SET effect_status = ?3 WHERE session_id = ?1 AND op_id = ?2",
            params![session_id.raw() as i64, op_id.raw() as i64, effect_status],
        )?;
        Ok(())
    }

    /// Unfinished tool runs (ToolStarted without ToolCompleted): the crash
    /// recovery scanner's input.
    pub fn pending_tool_runs(&self, session_id: SessionId) -> StoreResult<Vec<ToolRunRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, op_id, tool, args, status, started_ms, ended_ms, effect_status, recovery, expected_hash, replay_descriptor, attempt, postcondition
             FROM tool_run WHERE session_id = ?1 AND status = 'running'",
        )?;
        let mut rows = stmt.query(params![session_id.raw() as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(tool_run_map(row)?);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- turn records

    /// Durably open a logical-turn record. Called transactionally when a
    /// prompt is admitted as the ACTIVE logical turn (immediate admission in
    /// `submit_prompt`, or queue admission). Re-admission of the SAME turn op
    /// (a crash between admission and the first drive; the queue row is
    /// re-admitted after recovery) UPSERTS the same record — the turn's
    /// identity is never duplicated. Any OTHER still-active record of the
    /// session is finalized as failed in the same transaction (at most one
    /// active logical turn may exist per session).
    #[allow(clippy::too_many_arguments)]
    pub fn start_turn_record(
        &self,
        session_id: SessionId,
        turn_op_id: OpId,
        queue_seq: Option<i64>,
        prompt_message_id: Option<i64>,
        provider: &str,
        model: &str,
        variant: Option<&str>,
    ) -> StoreResult<i64> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE turn_record SET status = ?3, updated_ms = ?4
             WHERE session_id = ?1 AND status = 'active' AND turn_op_id != ?2",
            params![
                session_id.raw() as i64,
                turn_op_id.raw() as i64,
                TURN_RECORD_FAILED,
                now_ms()
            ],
        )?;
        let now = now_ms();
        tx.execute(
            "INSERT INTO turn_record(session_id, turn_op_id, queue_seq, prompt_message_id, effective_provider, effective_model, variant, started_at, status, updated_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'active', ?8)
             ON CONFLICT(session_id, turn_op_id) DO UPDATE SET
                queue_seq = excluded.queue_seq,
                prompt_message_id = excluded.prompt_message_id,
                effective_provider = excluded.effective_provider,
                effective_model = excluded.effective_model,
                variant = excluded.variant,
                started_at = excluded.started_at,
                status = 'active',
                updated_ms = excluded.updated_ms",
            params![
                session_id.raw() as i64,
                turn_op_id.raw() as i64,
                queue_seq,
                prompt_message_id,
                provider,
                model,
                variant,
                now
            ],
        )?;
        let id = tx.last_insert_rowid();
        tx.commit()?;
        Ok(id)
    }

    /// Finalize the record's effective envelope at logical-turn start (the
    /// per-message model override and the tool mode are only known once the
    /// runtime drives the turn). Only an active record is updated.
    pub fn set_turn_record_envelope(
        &self,
        session_id: SessionId,
        turn_op_id: OpId,
        provider: &str,
        model: &str,
        variant: Option<&str>,
        tool_mode: Option<&str>,
    ) -> StoreResult<bool> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE turn_record SET effective_provider = ?3, effective_model = ?4, variant = ?5, tool_mode = ?6, updated_ms = ?7
             WHERE session_id = ?1 AND turn_op_id = ?2 AND status = 'active'",
            params![
                session_id.raw() as i64,
                turn_op_id.raw() as i64,
                provider,
                model,
                variant,
                tool_mode,
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// Close an active turn record (completed | cancelled | failed).
    /// No-op when the record is absent or already closed (idempotent).
    pub fn finish_turn_record(
        &self,
        session_id: SessionId,
        turn_op_id: OpId,
        status: &str,
    ) -> StoreResult<bool> {
        if !matches!(
            status,
            TURN_RECORD_COMPLETED | TURN_RECORD_CANCELLED | TURN_RECORD_FAILED
        ) {
            return Err(StoreError::Migration(format!(
                "finish_turn_record: invalid status {status:?}"
            )));
        }
        let conn = self.write();
        let n = conn.execute(
            "UPDATE turn_record SET status = ?3, updated_ms = ?4
             WHERE session_id = ?1 AND turn_op_id = ?2 AND status = 'active'",
            params![
                session_id.raw() as i64,
                turn_op_id.raw() as i64,
                status,
                now_ms()
            ],
        )?;
        Ok(n > 0)
    }

    /// The session's single active logical-turn record (at most one exists).
    pub fn active_turn_record(&self, session_id: SessionId) -> StoreResult<Option<TurnRecordRow>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT id, session_id, turn_op_id, queue_seq, prompt_message_id, effective_provider, effective_model, variant, tool_mode, started_at, status, updated_ms
                 FROM turn_record WHERE session_id = ?1 AND status = 'active'
                 ORDER BY started_at DESC, id DESC LIMIT 1",
                params![session_id.raw() as i64],
                turn_record_map,
            )
            .optional()?;
        Ok(out)
    }

    pub fn turn_record_of(
        &self,
        session_id: SessionId,
        turn_op_id: OpId,
    ) -> StoreResult<Option<TurnRecordRow>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT id, session_id, turn_op_id, queue_seq, prompt_message_id, effective_provider, effective_model, variant, tool_mode, started_at, status, updated_ms
                 FROM turn_record WHERE session_id = ?1 AND turn_op_id = ?2",
                params![session_id.raw() as i64, turn_op_id.raw() as i64],
                turn_record_map,
            )
            .optional()?;
        Ok(out)
    }

    /// Every turn record of a session (oldest first; diagnostics/tests).
    pub fn turn_records_of(&self, session_id: SessionId) -> StoreResult<Vec<TurnRecordRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, turn_op_id, queue_seq, prompt_message_id, effective_provider, effective_model, variant, tool_mode, started_at, status, updated_ms
             FROM turn_record WHERE session_id = ?1 ORDER BY started_at ASC, id ASC",
        )?;
        let mut rows = stmt.query(params![session_id.raw() as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(turn_record_map(row)?);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- provider calls

    // Fixed-arity provider telemetry; the parameter list is a stable call
    // contract used across the workspace.
    #[allow(clippy::too_many_arguments)]
    pub fn record_provider_call(
        &self,
        session_id: SessionId,
        op_id: OpId,
        provider: &str,
        model: &str,
        status: &str,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        error: Option<&str>,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO provider_call(session_id, op_id, provider, model, started_ms, ended_ms, status, tokens_in, tokens_out, error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                provider,
                model,
                now_ms(),
                now_ms(),
                status,
                tokens_in.map(|t| t as i64),
                tokens_out.map(|t| t as i64),
                error
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    // ---------------------------------------------------------------- checkpoints

    /// ALLOCATE the next per-session checkpoint sequence and insert the row
    /// in ONE transaction (P1 "checkpoint numbering race"): two concurrent
    /// writers must never both receive the same sequence. The sequence is
    /// `MAX(sequence)+1` over the session's rows, computed and inserted under
    /// the single writer lock, so allocation is atomic and gapless regardless
    /// of what any caller guessed outside the store.
    ///
    /// `before_hash`/`after_hash` carry the side's content hash, or the empty
    /// string when that side does not exist (`before_exists=false`). Returns
    /// the row id and the allocated sequence.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_checkpoint(
        &self,
        session_id: SessionId,
        path: &str,
        before_exists: bool,
        before_hash: &str,
        after_exists: bool,
        after_hash: &str,
        after_cas_hash: Option<&str>,
    ) -> StoreResult<(i64, i64)> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let prev: i64 = tx.query_row(
            "SELECT COALESCE(MAX(sequence), 0) FROM checkpoint WHERE session_id = ?1",
            params![session_id.raw() as i64],
            |r| r.get(0),
        )?;
        let sequence = prev + 1;
        tx.execute(
            "INSERT INTO checkpoint(session_id, sequence, path, before_hash, after_hash, after_cas_hash, before_exists, after_exists, created_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                session_id.raw() as i64,
                sequence,
                path,
                before_hash,
                after_hash,
                after_cas_hash,
                before_exists as i64,
                after_exists as i64,
                now_ms()
            ],
        )?;
        let id = tx.last_insert_rowid();
        tx.commit()?;
        Ok((id, sequence))
    }

    /// Raw checkpoint insert at an explicit caller-chosen sequence (the
    /// session layer's journal-backed path, which validates duplicates
    /// itself). Both sides exist. Prefer [`Store::insert_checkpoint`] for
    /// content-aware checkpoints: it allocates the sequence atomically.
    pub fn put_checkpoint(
        &self,
        session_id: SessionId,
        sequence: i64,
        path: &str,
        before_hash: &str,
        after_hash: &str,
        after_cas_hash: Option<&str>,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO checkpoint(session_id, sequence, path, before_hash, after_hash, after_cas_hash, created_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id.raw() as i64,
                sequence,
                path,
                before_hash,
                after_hash,
                after_cas_hash,
                now_ms()
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn checkpoints_of(&self, session_id: SessionId) -> StoreResult<Vec<CheckpointRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, sequence, path, before_hash, after_hash, after_cas_hash, before_exists, after_exists, created_ms, restored_ms
             FROM checkpoint WHERE session_id = ?1 ORDER BY sequence ASC, id ASC",
        )?;
        let rows = stmt.query_map(params![session_id.raw() as i64], |r| {
            Ok(CheckpointRow {
                id: r.get(0)?,
                session_id,
                sequence: r.get(2)?,
                path: r.get(3)?,
                before_hash: r.get(4)?,
                after_hash: r.get(5)?,
                after_cas_hash: r.get(6)?,
                before_exists: r.get::<_, i64>(7)? != 0,
                after_exists: r.get::<_, i64>(8)? != 0,
                created_ms: r.get(9)?,
                restored_ms: r.get(10)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Redo/undo marker: rollback sets restored_ms, redo clears it (a row
    /// must not read as "restored" after an unrevert; audit round 5).
    pub fn clear_checkpoint_restored(&self, id: i64) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE checkpoint SET restored_ms = NULL WHERE id = ?1",
            params![id],
        )?;
        Ok(())
    }

    pub fn mark_checkpoint_restored(&self, id: i64) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE checkpoint SET restored_ms = ?2 WHERE id = ?1",
            params![id, now_ms()],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------------- artifacts

    /// Artifact rows reference CAS hashes; the blob itself lives in the CAS.
    pub fn put_artifact(
        &self,
        session_id: SessionId,
        kind: &str,
        cas_hash: &str,
        summary: &str,
        size: i64,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT OR IGNORE INTO artifact(session_id, kind, cas_hash, summary, created_ms, size)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                session_id.raw() as i64,
                kind,
                cas_hash,
                summary,
                now_ms(),
                size
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn artifact(&self, cas_hash: &str) -> StoreResult<Option<(String, String)>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT summary, kind FROM artifact WHERE cas_hash = ?1",
                params![cas_hash],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)),
            )
            .ok();
        Ok(out)
    }

    // ---------------------------------------------------------------- worktrees

    pub fn put_worktree(
        &self,
        workspace_id: WorkspaceId,
        path: &str,
        branch: &str,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT OR IGNORE INTO worktree(workspace_id, path, branch, active)
             VALUES (?1, ?2, ?3, 1)",
            params![workspace_id.raw() as i64, path, branch],
        )?;
        conn.query_row(
            "SELECT id FROM worktree WHERE path = ?1",
            params![path],
            |r| r.get(0),
        )
        .map_err(Into::into)
    }

    pub fn worktrees_of(&self, workspace_id: WorkspaceId) -> StoreResult<Vec<WorktreeRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, workspace_id, path, branch, active FROM worktree WHERE workspace_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![workspace_id.raw() as i64], |r| {
            Ok(WorktreeRow {
                id: r.get(0)?,
                workspace_id,
                path: r.get(2)?,
                branch: r.get(3)?,
                active: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    pub fn remove_worktree(&self, path: &str) -> StoreResult<()> {
        let conn = self.write();
        conn.execute("DELETE FROM worktree WHERE path = ?1", params![path])?;
        Ok(())
    }

    // ---------------------------------------------------------------- memory facts

    pub fn upsert_memory_fact(
        &self,
        session_id: SessionId,
        kind: &str,
        key: &str,
        value: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO memory_fact(session_id, kind, key, value, updated_ms) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(session_id, kind, key) DO UPDATE SET value = ?4, updated_ms = ?5",
            params![session_id.raw() as i64, kind, key, value, now_ms()],
        )?;
        Ok(())
    }

    pub fn memory_facts(
        &self,
        session_id: SessionId,
    ) -> StoreResult<Vec<(String, String, String)>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT kind, key, value FROM memory_fact WHERE session_id = ?1 ORDER BY updated_ms DESC",
        )?;
        let rows = stmt.query_map(params![session_id.raw() as i64], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- compactions

    pub fn record_compaction(
        &self,
        session_id: SessionId,
        before_tokens: i64,
        after_tokens: i64,
        target_tokens: i64,
        accepted: bool,
        strategy: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO compaction(session_id, before_tokens, after_tokens, target_tokens, accepted, strategy, created_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                session_id.raw() as i64,
                before_tokens,
                after_tokens,
                target_tokens,
                accepted as i64,
                strategy,
                now_ms()
            ],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------------- permissions

    pub fn insert_permission(
        &self,
        session_id: SessionId,
        op_id: OpId,
        capability: &str,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO permission(session_id, op_id, capability, decision, expires_ms)
             VALUES (?1, ?2, ?3, 'pending', ?4)",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                capability,
                now_ms() + 60_000
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn resolve_permission(&self, id: i64, decision: &str) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE permission SET decision = ?2, resolved_ms = ?3 WHERE id = ?1 AND decision = 'pending'",
            params![id, decision, now_ms()],
        )?;
        Ok(())
    }

    pub fn pending_permission(&self, id: i64) -> StoreResult<Option<(SessionId, OpId, String)>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT session_id, op_id, capability FROM permission WHERE id = ?1 AND decision = 'pending'",
                params![id],
                |r| {
                    Ok((
                        SessionId::new(r.get::<_, i64>(0)? as u64),
                        OpId::new(r.get::<_, i64>(1)? as u64),
                        r.get::<_, String>(2)?,
                    ))
                },
            )
            .ok();
        Ok(out)
    }

    // ---------------------------------------------------------------- prompt queue
    /// Durably queue a prompt that arrived while another turn was active.
    /// The full execution envelope is stored; the user conversation message
    /// is NOT materialized yet (deferred materialization — audit round 7:
    /// conversation chronology is insertion order, so the message is
    /// appended at admission, after the preceding turn's output).
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_prompt(
        &self,
        session: SessionId,
        op_id: OpId,
        prompt: &str,
        files: &[String],
        model: Option<&str>,
        variant: Option<&str>,
        agent: Option<&str>,
        requested_at: i64,
    ) -> StoreResult<i64> {
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let prev: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM prompt_queue WHERE session_id = ?1",
            params![session.raw() as i64],
            |r| r.get(0),
        )?;
        let seq = prev + 1;
        tx.execute(
            "INSERT INTO prompt_queue(session_id, seq, op_id, prompt, files, model, variant, agent, status, requested_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'pending', ?9)",
            params![
                session.raw() as i64,
                seq,
                op_id.raw() as i64,
                prompt,
                serde_json::to_string(files).unwrap_or_else(|_| "[]".into()),
                model,
                variant,
                agent,
                requested_at
            ],
        )?;
        tx.commit()?;
        Ok(seq)
    }

    fn queue_row_from(r: &rusqlite::Row<'_>) -> rusqlite::Result<QueuedPrompt> {
        Ok(QueuedPrompt {
            queue_seq: r.get(0)?,
            op_id: OpId::new(r.get::<_, i64>(1)? as u64),
            prompt: r.get(2)?,
            files: serde_json::from_str(&r.get::<_, String>(3)?).unwrap_or_default(),
            model: r.get(4)?,
            variant: r.get(5)?,
            agent: r.get(6)?,
            status: r.get(7)?,
            requested_at: r.get(8)?,
        })
    }

    /// Oldest row that is not terminal (pending/claimed/running) — FIFO head.
    pub fn queue_head(&self, session: SessionId) -> StoreResult<Option<QueuedPrompt>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT seq, op_id, prompt, files, model, variant, agent, status, requested_at
                 FROM prompt_queue
                 WHERE session_id = ?1 AND status IN ('pending','claimed','running')
                 ORDER BY seq ASC LIMIT 1",
                params![session.raw() as i64],
                Self::queue_row_from,
            )
            .ok();
        Ok(out)
    }

    /// Count of rows in each durable status (diagnostics).
    pub fn queue_status_counts(&self, session: SessionId) -> StoreResult<serde_json::Value> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT status, COUNT(*) FROM prompt_queue WHERE session_id = ?1 GROUP BY status",
        )?;
        let rows = stmt.query_map(params![session.raw() as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?;
        let mut counts = serde_json::Map::new();
        for row in rows {
            let (k, v) = row?;
            counts.insert(k, serde_json::json!(v));
        }
        Ok(serde_json::Value::Object(counts))
    }

    /// Atomic claim + admission of the queue head (audit round 7): ONE
    /// transaction establishes (a) the head is pending and the session is
    /// eligible, (b) pending -> claimed, (c) the user message is materialized
    /// at the true conversation tail, (d) the session row moves to the
    /// target state, (e) the admission journal event seq is computed so the
    /// session layer can journal the turn-open with a gapless sequence. No
    /// other submission can cut between those operations (single writer).
    ///
    /// Returns Ok(None) when the head is absent or the session state is not
    /// in `eligible_states` (nothing is touched in either case).
    pub fn admit_queue_head(
        &self,
        session: SessionId,
        eligible_states: &[&str],
        target_state: &str,
    ) -> StoreResult<Option<(AdmittedPrompt, i64)>> {
        type QueueHeadRow = (
            i64,
            OpId,
            String,
            String,
            Option<String>,
            Option<String>,
            Option<String>,
        );
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let head: Option<QueueHeadRow> = tx
            .query_row(
                "SELECT seq, op_id, prompt, files, model, variant, agent FROM prompt_queue
                 WHERE session_id = ?1 AND status = 'pending' ORDER BY seq ASC LIMIT 1",
                params![session.raw() as i64],
                |r| {
                    Ok((
                        r.get(0)?,
                        OpId::new(r.get::<_, i64>(1)? as u64),
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, Option<String>>(4)?,
                        r.get::<_, Option<String>>(5)?,
                        r.get::<_, Option<String>>(6)?,
                    ))
                },
            )
            .ok();
        let Some((queue_seq, op_id, prompt, files_json, model, variant, agent)) = head else {
            return Ok(None);
        };
        let state: String = tx
            .query_row(
                "SELECT state FROM session WHERE id = ?1",
                params![session.raw() as i64],
                |r| r.get(0),
            )
            .map_err(|e| StoreError::Migration(format!("session missing: {e}")))?;
        let state_label: String = serde_json::from_str(&state).unwrap_or_default();
        if !eligible_states.contains(&state_label.as_str()) {
            return Ok(None);
        }
        tx.execute(
            "UPDATE prompt_queue SET status = 'claimed', claimed_at = ?2
             WHERE session_id = ?1 AND seq = ?3",
            params![session.raw() as i64, now_ms(), queue_seq],
        )?;
        let prev_event: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM event WHERE session_id = ?1",
            params![session.raw() as i64],
            |r| r.get(0),
        )?;
        let event_seq = prev_event + 1;
        tx.execute(
            "INSERT INTO message(session_id, seq, role, data, created_ms)
             VALUES (?1, ?2, 'user', ?3, ?4)",
            params![
                session.raw() as i64,
                event_seq,
                serde_json::json!({ "text": prompt }).to_string(),
                now_ms()
            ],
        )?;
        tx.execute(
            "UPDATE session SET state = ?2, updated_ms = ?3 WHERE id = ?1",
            params![
                session.raw() as i64,
                serde_json::to_string(target_state).unwrap(),
                now_ms()
            ],
        )?;
        tx.commit()?;
        Ok(Some((
            AdmittedPrompt {
                queue_seq,
                op_id,
                prompt,
                files: serde_json::from_str(&files_json).unwrap_or_default(),
                model,
                variant,
                agent,
                message_seq: event_seq,
            },
            event_seq,
        )))
    }

    pub fn mark_queue_status(
        &self,
        session: SessionId,
        queue_seq: i64,
        status: &str,
    ) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "UPDATE prompt_queue SET status = ?3, completed_at = ?4
             WHERE session_id = ?1 AND seq = ?2",
            params![session.raw() as i64, queue_seq, status, now_ms()],
        )?;
        Ok(())
    }

    /// Op ids of all non-terminal queue rows for a session (abort(None)
    /// must durably cancel queued prompts too).
    pub fn queue_op_ids(&self, session: SessionId) -> StoreResult<Vec<OpId>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT op_id FROM prompt_queue
             WHERE session_id = ?1 AND status IN ('pending','claimed')",
        )?;
        let rows = stmt.query_map(params![session.raw() as i64], |r| r.get::<_, i64>(0))?;
        let mut out = Vec::new();
        for v in rows {
            out.push(OpId::new(v? as u64));
        }
        Ok(out)
    }

    /// Durable cancellation of queued rows (abort semantics): pending and
    /// claimed rows for the given ops become cancelled and are never
    /// admitted. Returns how many rows were cancelled.
    pub fn cancel_queued_ops(&self, session: SessionId, ops: &[OpId]) -> StoreResult<i64> {
        let conn = self.write();
        let mut n = 0i64;
        for op in ops {
            n += conn.execute(
                "UPDATE prompt_queue SET status = 'cancelled', completed_at = ?3
                 WHERE session_id = ?1 AND op_id = ?2 AND status IN ('pending','claimed')",
                params![session.raw() as i64, op.raw() as i64, now_ms()],
            )? as i64;
        }
        Ok(n)
    }

    /// Recovery pass: claimed rows that were never executed (crash between
    /// claim and execution) return to pending so they are re-admitted;
    /// running rows are left for turn-level recovery. Returns the re-admitted
    /// count.
    pub fn recover_claimed_queue_rows(&self, session: SessionId) -> StoreResult<i64> {
        let conn = self.write();
        let n = conn.execute(
            "UPDATE prompt_queue SET status = 'pending', claimed_at = NULL
             WHERE session_id = ?1 AND status = 'claimed'",
            params![session.raw() as i64],
        )? as i64;
        Ok(n)
    }

    /// All session ids with non-terminal queue rows (startup kick list).
    pub fn sessions_with_pending_queues(&self) -> StoreResult<Vec<SessionId>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT DISTINCT session_id FROM prompt_queue
             WHERE status IN ('pending','claimed','running')",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, i64>(0))?;
        let mut out = Vec::new();
        for v in rows {
            out.push(SessionId::new(v? as u64));
        }
        Ok(out)
    }

    /// Durable bump of one loop signal (spec §28): count the same key
    /// across turns and daemon restarts. Returns true when `threshold` is
    /// reached; the count then resets (a trip closes the window).
    pub fn bump_loop_signal(
        &self,
        session: SessionId,
        key: &str,
        threshold: u32,
        ts_ms: i64,
    ) -> StoreResult<bool> {
        if key.is_empty() || key.len() > 1024 || threshold < 2 {
            return Err(StoreError::Migration(
                "loop signal key must be 1..=1024 bytes; threshold >= 2".into(),
            ));
        }
        let conn = self.write();
        let tx = conn.unchecked_transaction()?;
        let prev: Option<i64> = tx
            .query_row(
                "SELECT count FROM loop_signal WHERE session_id = ?1 AND key = ?2",
                params![session.raw() as i64, key],
                |r| r.get(0),
            )
            .optional()?;
        let count = prev.unwrap_or(0) + 1;
        tx.execute(
            "INSERT OR REPLACE INTO loop_signal(session_id, key, count, updated_ms)
             VALUES (?1, ?2, ?3, ?4)",
            params![session.raw() as i64, key, count, ts_ms],
        )?;
        if count >= i64::from(threshold) {
            tx.execute(
                "DELETE FROM loop_signal WHERE session_id = ?1 AND key = ?2",
                params![session.raw() as i64, key],
            )?;
            tx.commit()?;
            return Ok(true);
        }
        tx.commit()?;
        Ok(false)
    }

    /// Clear every loop signal of the session (the task made progress).
    pub fn reset_loop_signals(&self, session: SessionId) -> StoreResult<()> {
        let conn = self.write();
        conn.execute(
            "DELETE FROM loop_signal WHERE session_id = ?1",
            params![session.raw() as i64],
        )?;
        Ok(())
    }

    pub fn loop_signal_counts(&self, session: SessionId) -> StoreResult<Vec<(String, i64)>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT key, count FROM loop_signal WHERE session_id = ?1 ORDER BY updated_ms DESC LIMIT 100",
        )?;
        let rows = stmt.query_map(params![session.raw() as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    pub fn integrity_check(&self) -> StoreResult<Vec<String>> {
        let conn = self.read()?;
        let out = check_integrity(&conn)?;
        Ok(out)
    }

    /// Online backup via the SQLite backup API (safe while the daemon runs).
    pub fn backup_to(&self, dest: &Path) -> StoreResult<()> {
        let src = self.write();
        let mut dst = Connection::open(dest)?;
        let backup = rusqlite::backup::Backup::new(&src, &mut dst)?;
        backup.run_to_completion(50, std::time::Duration::from_millis(100), None)?;
        Ok(())
    }

    /// `doctor`-style diagnostic.
    pub fn diagnostics(&self) -> StoreResult<serde_json::Value> {
        let conn = self.read()?;
        let sessions: i64 = conn.query_row("SELECT COUNT(*) FROM session", [], |r| r.get(0))?;
        let events: i64 = conn.query_row("SELECT COUNT(*) FROM event", [], |r| r.get(0))?;
        let messages: i64 = conn.query_row("SELECT COUNT(*) FROM message", [], |r| r.get(0))?;
        let journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |r| r.get(0))?;
        let integrity = check_integrity(&conn)?;
        Ok(serde_json::json!({
            "journal_mode": journal_mode,
            "sessions": sessions,
            "events": events,
            "messages": messages,
            "integrity": integrity,
        }))
    }
}

fn configure(conn: &Connection) -> StoreResult<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous = NORMAL;
         PRAGMA busy_timeout = 5000;
         PRAGMA foreign_keys = ON;",
    )?;
    Ok(())
}

fn check_integrity(conn: &Connection) -> StoreResult<Vec<String>> {
    let mut stmt = conn.prepare("PRAGMA integrity_check")?;
    let mut rows = stmt.query([])?;
    let mut issues = Vec::new();
    while let Some(row) = rows.next()? {
        let line: String = row.get(0)?;
        if line != "ok" {
            issues.push(line);
        }
    }
    Ok(issues)
}

const MIGRATIONS: &[&str] = &[
    // v1 — initial schema
    "CREATE TABLE IF NOT EXISTS workspace (
        id INTEGER PRIMARY KEY,
        root TEXT NOT NULL UNIQUE,
        created_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS session (
        id INTEGER PRIMARY KEY,
        workspace_id INTEGER NOT NULL REFERENCES workspace(id),
        title TEXT NOT NULL DEFAULT '',
        provider TEXT NOT NULL DEFAULT '',
        model TEXT NOT NULL DEFAULT '',
        state TEXT NOT NULL DEFAULT '\"idle\"',
        created_ms INTEGER NOT NULL,
        updated_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS event (
        seq INTEGER NOT NULL,
        session_id INTEGER NOT NULL REFERENCES session(id),
        op_id INTEGER,
        kind TEXT NOT NULL,
        state TEXT NOT NULL,
        ts_ms INTEGER NOT NULL,
        payload TEXT,
        PRIMARY KEY (session_id, seq)
     );
     CREATE TABLE IF NOT EXISTS message (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        seq INTEGER NOT NULL,
        role TEXT NOT NULL,
        data TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        UNIQUE (session_id, seq)
     );
     CREATE TABLE IF NOT EXISTS part (
        id INTEGER PRIMARY KEY,
        message_id INTEGER NOT NULL REFERENCES message(id),
        kind TEXT NOT NULL,
        data TEXT NOT NULL,
        created_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS task (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        ledger TEXT NOT NULL,
        updated_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS tool_run (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        op_id INTEGER NOT NULL UNIQUE,
        tool TEXT NOT NULL,
        args TEXT NOT NULL,
        status TEXT NOT NULL,
        started_ms INTEGER NOT NULL,
        ended_ms INTEGER,
        effect_status TEXT NOT NULL,
        recovery TEXT NOT NULL,
        expected_hash TEXT
     );
     CREATE TABLE IF NOT EXISTS provider_call (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        op_id INTEGER NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        started_ms INTEGER NOT NULL,
        ended_ms INTEGER,
        status TEXT NOT NULL,
        tokens_in INTEGER,
        tokens_out INTEGER,
        error TEXT
     );
     CREATE TABLE IF NOT EXISTS checkpoint (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        sequence INTEGER NOT NULL,
        path TEXT NOT NULL,
        before_hash TEXT NOT NULL,
        after_hash TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        restored_ms INTEGER
     );
     CREATE TABLE IF NOT EXISTS artifact (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        kind TEXT NOT NULL,
        cas_hash TEXT NOT NULL UNIQUE,
        summary TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        size INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS worktree (
        id INTEGER PRIMARY KEY,
        workspace_id INTEGER NOT NULL REFERENCES workspace(id),
        path TEXT NOT NULL UNIQUE,
        branch TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
     );
     CREATE TABLE IF NOT EXISTS memory_fact (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        kind TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        updated_ms INTEGER NOT NULL,
        UNIQUE (session_id, kind, key)
     );
     CREATE TABLE IF NOT EXISTS compaction (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        before_tokens INTEGER NOT NULL,
        after_tokens INTEGER NOT NULL,
        target_tokens INTEGER NOT NULL,
        accepted INTEGER NOT NULL,
        strategy TEXT NOT NULL,
        created_ms INTEGER NOT NULL
     );
     CREATE TABLE IF NOT EXISTS permission (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        op_id INTEGER NOT NULL,
        capability TEXT NOT NULL,
        decision TEXT NOT NULL,
        resolved_ms INTEGER,
        expires_ms INTEGER NOT NULL
     );
     CREATE INDEX IF NOT EXISTS idx_event_session_seq ON event(session_id, seq);
     CREATE INDEX IF NOT EXISTS idx_message_session_seq ON message(session_id, seq);
     CREATE INDEX IF NOT EXISTS idx_toolrun_session ON tool_run(session_id, status);
     CREATE INDEX IF NOT EXISTS idx_checkpoint_session ON checkpoint(session_id, sequence);",
    // v2 — session lifecycle (orthogonal to the turn state machine)
    "ALTER TABLE session ADD COLUMN lifecycle TEXT NOT NULL DEFAULT 'open';",
    // v3 — checkpoint rows carry the CAS hash of the AFTER-content blob, so
    // unrevert (redo) and diff can reconstruct what the edit wrote. NULL on
    // pre-v3 rows: those checkpoints refuse redo/diff honestly.
    "ALTER TABLE checkpoint ADD COLUMN after_cas_hash TEXT;",
    // v4 — durable per-session prompt queue with a FULL execution envelope
    // and a durable state machine (audit rounds 6+7): pending | claimed |
    // running | done | cancelled. The user conversation message is NOT
    // stored here — it is materialized at ADMISSION (after the preceding
    // turn's output) so conversation chronology is the insertion order.
    "CREATE TABLE IF NOT EXISTS prompt_queue (
        session_id INTEGER NOT NULL REFERENCES session(id),
        seq INTEGER NOT NULL,
        op_id INTEGER NOT NULL,
        message_seq INTEGER,
        delivered INTEGER NOT NULL DEFAULT 0,
        prompt TEXT NOT NULL DEFAULT '',
        files TEXT NOT NULL DEFAULT '[]',
        model TEXT,
        variant TEXT,
        agent TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        requested_at INTEGER NOT NULL DEFAULT 0,
        claimed_at INTEGER,
        completed_at INTEGER,
        PRIMARY KEY (session_id, seq)
     );",
    // v5 — durable loop signals (spec §28): repeated identical failing
    // tool calls across LOGICAL TURNS and daemon restarts are detected
    // from this table, never from memory.
    "CREATE TABLE IF NOT EXISTS loop_signal (
        session_id INTEGER NOT NULL REFERENCES session(id),
        key TEXT NOT NULL,
        count INTEGER NOT NULL,
        updated_ms INTEGER NOT NULL,
        PRIMARY KEY (session_id, key)
     );",
    // v6 — checkpoint rows carry per-side EXISTENCE flags. A hash alone
    // cannot distinguish a missing file from an empty one: before==after
    // ("no change") currently means the empty-file creation is skipped and
    // rollback of a missing→content checkpoint would recreate an empty file
    // instead of deleting. DEFAULT 1 keeps pre-v6 rows readable: old rows
    // were only recorded for real files (the caller had content on both
    // sides), so "hash present with no existence marker means exists:true".
    "ALTER TABLE checkpoint ADD COLUMN before_exists INTEGER NOT NULL DEFAULT 1;",
    "ALTER TABLE checkpoint ADD COLUMN after_exists INTEGER NOT NULL DEFAULT 1;",
    // v7 — exact per-turn operation identity + recovery descriptors.
    // `turn_record` fixes the durable identity of every ADMITTED logical
    // turn (op id, queue seq, prompt message, effective provider/model/
    // variant/tool mode, status) so crash recovery resumes the SAME turn
    // with the SAME recorded envelope instead of synthesizing an operation.
    // `tool_run` gains the crash-recovery machinery: the durable
    // `replay_descriptor` (the stored invocation an idempotent tool may be
    // re-executed from), the `attempt` counter (a replay is a new PHYSICAL
    // attempt of the SAME logical operation) and the `postcondition`
    // (workspace-write verification data computed from the actual bytes as
    // written — never from JSON-encoded args).
    "CREATE TABLE IF NOT EXISTS turn_record (
        id INTEGER PRIMARY KEY,
        session_id INTEGER NOT NULL REFERENCES session(id),
        turn_op_id INTEGER NOT NULL,
        queue_seq INTEGER,
        prompt_message_id INTEGER,
        effective_provider TEXT NOT NULL DEFAULT '',
        effective_model TEXT NOT NULL DEFAULT '',
        variant TEXT,
        tool_mode TEXT,
        started_at INTEGER NOT NULL,
        status TEXT NOT NULL,
        updated_ms INTEGER NOT NULL,
        UNIQUE (session_id, turn_op_id)
     );
     CREATE INDEX IF NOT EXISTS idx_turn_record_session_status ON turn_record(session_id, status);
     ALTER TABLE tool_run ADD COLUMN replay_descriptor TEXT;
     ALTER TABLE tool_run ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0;
     ALTER TABLE tool_run ADD COLUMN postcondition TEXT;",
    // v8 — durable worktree/task identity on sessions. Tool calls were
    // being handed fake identities (worktree 1/task 1) because the real
    // ones lived nowhere durable: the session row now records them, so the
    // agent runtime builds `ToolRunCtx.identity` from the session row and
    // every replay descriptor / postcondition rides the SAME ids. DEFAULT 1
    // preserves existing rows: 1/1 is the documented STANDALONE session
    // identity (no worktree/task adopted); WorktreeManager-created
    // worktrees adopt their sessions deliberately afterwards.
    // (This block is array index 8, i.e. schema target 9: the v6 checkpoint
    // block spans two array entries before it.)
    "ALTER TABLE session ADD COLUMN worktree_id INTEGER NOT NULL DEFAULT 1;
     ALTER TABLE session ADD COLUMN task_id INTEGER NOT NULL DEFAULT 1;",
];

/// Apply migrations transactionally; `PRAGMA user_version` is the cursor.
fn migrate(conn: &mut Connection) -> StoreResult<()> {
    let mut version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    for (i, sql) in MIGRATIONS.iter().enumerate() {
        let target = (i + 1) as i64;
        if version >= target {
            continue;
        }
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        tx.execute_batch(sql)
            .map_err(|e| StoreError::Migration(format!("v{target}: {e}")))?;
        tx.execute_batch(&format!("PRAGMA user_version = {target}"))
            .map_err(|e| StoreError::Migration(format!("v{target} version write: {e}")))?;
        tx.commit()
            .map_err(|e| StoreError::Migration(format!("v{target} commit: {e}")))?;
        version = target;
    }
    Ok(())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn kind_name(k: EventKind) -> &'static str {
    match k {
        EventKind::SessionCreated => "session_created",
        EventKind::PromptReceived => "prompt_received",
        EventKind::ContextPrepared => "context_prepared",
        EventKind::ModelStarted => "model_started",
        EventKind::ModelChunkReceived => "model_chunk_received",
        EventKind::ToolRequested => "tool_requested",
        EventKind::ToolStarted => "tool_started",
        EventKind::FileChanged => "file_changed",
        EventKind::ToolCompleted => "tool_completed",
        EventKind::ToolCancelled => "tool_cancelled",
        EventKind::CheckpointCreated => "checkpoint_created",
        EventKind::ContextCompacted => "context_compacted",
        EventKind::CompactRejected => "compact_rejected",
        EventKind::SubagentStarted => "subagent_started",
        EventKind::SubagentCompleted => "subagent_completed",
        EventKind::TurnCompleted => "turn_completed",
        EventKind::PermissionGranted => "permission_granted",
        EventKind::PermissionDenied => "permission_denied",
        EventKind::PhaseChanged => "phase_changed",
        EventKind::ReplayStarted => "replay_started",
        EventKind::PromptAdmitted => "prompt_admitted",
        EventKind::CrashDetected => "crash_detected",
        EventKind::RecoveryApplied => "recovery_applied",
        EventKind::SessionEnded => "session_ended",
        EventKind::Suspended => "suspended",
        EventKind::Resumed => "resumed",
        EventKind::Failed => "failed",
    }
}

fn kind_from_name(name: &str) -> Option<EventKind> {
    Some(match name {
        "session_created" => EventKind::SessionCreated,
        "prompt_received" => EventKind::PromptReceived,
        "context_prepared" => EventKind::ContextPrepared,
        "model_started" => EventKind::ModelStarted,
        "model_chunk_received" => EventKind::ModelChunkReceived,
        "tool_requested" => EventKind::ToolRequested,
        "tool_started" => EventKind::ToolStarted,
        "file_changed" => EventKind::FileChanged,
        "tool_completed" => EventKind::ToolCompleted,
        "tool_cancelled" => EventKind::ToolCancelled,
        "checkpoint_created" => EventKind::CheckpointCreated,
        "context_compacted" => EventKind::ContextCompacted,
        "compact_rejected" => EventKind::CompactRejected,
        "subagent_started" => EventKind::SubagentStarted,
        "subagent_completed" => EventKind::SubagentCompleted,
        "turn_completed" => EventKind::TurnCompleted,
        "permission_granted" => EventKind::PermissionGranted,
        "permission_denied" => EventKind::PermissionDenied,
        "phase_changed" => EventKind::PhaseChanged,
        "replay_started" => EventKind::ReplayStarted,
        "prompt_admitted" => EventKind::PromptAdmitted,
        "crash_detected" => EventKind::CrashDetected,
        "recovery_applied" => EventKind::RecoveryApplied,
        "session_ended" => EventKind::SessionEnded,
        "suspended" => EventKind::Suspended,
        "resumed" => EventKind::Resumed,
        "failed" => EventKind::Failed,
        _ => return None,
    })
}

fn message_map(r: &rusqlite::Row<'_>) -> StoreResult<MessageRow> {
    let id = r.get::<_, i64>(0)?;
    Ok(MessageRow {
        id,
        session_id: SessionId::new(r.get::<_, i64>(1)? as u64),
        seq: r.get(2)?,
        role: r.get(3)?,
        data: parse_json(&format!("message {id} data"), &r.get::<_, String>(4)?)?,
        created_ms: r.get(5)?,
    })
}

fn session_row_map(r: &rusqlite::Row<'_>) -> StoreResult<SessionRow> {
    let id = SessionId::new(r.get::<_, i64>(0)? as u64);
    Ok(SessionRow {
        id,
        workspace_id: WorkspaceId::new(r.get::<_, i64>(1)? as u64),
        worktree_id: WorktreeId::new(r.get::<_, i64>(2)? as u64),
        task_id: TaskId::new(r.get::<_, i64>(3)? as u64),
        title: r.get(4)?,
        provider: r.get(5)?,
        model: r.get(6)?,
        state: parse_json(&format!("session {id} state"), &r.get::<_, String>(7)?)?,
        lifecycle: parse_lifecycle(&format!("session {id} lifecycle"), &r.get::<_, String>(8)?)?,
        created_ms: r.get(9)?,
        updated_ms: r.get(10)?,
    })
}

/// Parse of a persisted lifecycle that FAILS CLOSED on corruption.
///
/// A session that was really Closed/FailedPermanent must never silently
/// become Open (an autonomous agent could accept work again), so unreadable
/// content surfaces as `StoreError::Corrupt` instead of defaulting to Open.
///
/// The one tolerated non-JSON spelling is the bare literal `open`: the v2
/// schema declares `lifecycle TEXT NOT NULL DEFAULT 'open'`, `create_session`
/// INSERTs exactly that SQL literal, and the v2 ALTER backfilled every
/// pre-existing row to it — it is the schema's own default representation of
/// `Open`, not corruption. The column is NOT NULL, so a NULL lifecycle cannot
/// occur; had one been read (e.g. constraints disabled), the decode would
/// fail via the `Sqlite` error rather than reopening the session.
fn parse_lifecycle(ctx: &str, raw: &str) -> StoreResult<kilop_core::state::SessionLifecycle> {
    if raw == "open" {
        return Ok(kilop_core::state::SessionLifecycle::Open);
    }
    serde_json::from_str(raw)
        .map_err(|e| StoreError::Corrupt(vec![format!("{ctx}: lifecycle {raw:?} is corrupt: {e}")]))
}

fn event_map(r: &rusqlite::Row<'_>, session_id: SessionId) -> StoreResult<Event> {
    let seq = EventSeq::new(r.get::<_, i64>(0)? as u64);
    let kind_raw = r.get::<_, String>(3)?;
    let kind = kind_from_name(&kind_raw).ok_or_else(|| {
        StoreError::Corrupt(vec![format!(
            "event {session_id}/{seq}: unknown kind {kind_raw:?}"
        )])
    })?;
    Ok(Event {
        seq,
        session_id,
        op_id: r.get::<_, Option<i64>>(2)?.map(|o| OpId::new(o as u64)),
        kind,
        state: parse_json(
            &format!("event {session_id}/{seq} state"),
            &r.get::<_, String>(4)?,
        )?,
        ts_ms: r.get(5)?,
        payload: match r.get::<_, Option<String>>(6)? {
            Some(raw) => Some(parse_json(
                &format!("event {session_id}/{seq} payload"),
                &raw,
            )?),
            None => None,
        },
    })
}

fn part_map(r: &rusqlite::Row<'_>) -> StoreResult<PartRow> {
    let id = r.get::<_, i64>(0)?;
    Ok(PartRow {
        id,
        message_id: r.get(1)?,
        kind: r.get(2)?,
        data: parse_json(&format!("part {id} data"), &r.get::<_, String>(3)?)?,
        created_ms: r.get(4)?,
    })
}

fn tool_run_map(r: &rusqlite::Row<'_>) -> StoreResult<ToolRunRow> {
    let id = r.get::<_, i64>(0)?;
    Ok(ToolRunRow {
        id,
        session_id: SessionId::new(r.get::<_, i64>(1)? as u64),
        op_id: OpId::new(r.get::<_, i64>(2)? as u64),
        tool: r.get(3)?,
        args: parse_json(&format!("tool_run {id} args"), &r.get::<_, String>(4)?)?,
        status: r.get(5)?,
        started_ms: r.get(6)?,
        ended_ms: r.get(7)?,
        effect_status: r.get(8)?,
        recovery: parse_json(&format!("tool_run {id} recovery"), &r.get::<_, String>(9)?)?,
        expected_hash: r.get(10)?,
        replay_descriptor: match r.get::<_, Option<String>>(11)? {
            Some(raw) => Some(parse_json(
                &format!("tool_run {id} replay_descriptor"),
                &raw,
            )?),
            None => None,
        },
        attempt: r.get(12)?,
        postcondition: match r.get::<_, Option<String>>(13)? {
            Some(raw) => Some(parse_json(&format!("tool_run {id} postcondition"), &raw)?),
            None => None,
        },
    })
}

fn turn_record_map(r: &rusqlite::Row<'_>) -> rusqlite::Result<TurnRecordRow> {
    let id = r.get::<_, i64>(0)?;
    Ok(TurnRecordRow {
        id,
        session_id: SessionId::new(r.get::<_, i64>(1)? as u64),
        turn_op_id: OpId::new(r.get::<_, i64>(2)? as u64),
        queue_seq: r.get(3)?,
        prompt_message_id: r.get(4)?,
        effective_provider: r.get(5)?,
        effective_model: r.get(6)?,
        variant: r.get(7)?,
        tool_mode: r.get(8)?,
        started_at: r.get(9)?,
        status: r.get(10)?,
        updated_ms: r.get(11)?,
    })
}

/// Fallible JSON parse of persisted data: corrupted or version-skewed rows
/// surface as `Corrupt`, never a panic.
fn parse_json<T: serde::de::DeserializeOwned>(ctx: &str, raw: &str) -> StoreResult<T> {
    serde_json::from_str(raw).map_err(|e| StoreError::Corrupt(vec![format!("{ctx}: {e}")]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kilop_core::capability::PermissionDecision;

    fn tmp_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let s = Store::open(dir.path(), true).unwrap();
        (dir, s)
    }

    #[test]
    fn loop_signal_survives_reopen_and_trips_at_threshold() {
        // Spec §28 durable signals: crash between turns must NOT lose the
        // count — the third identical failure after a daemon restart still
        // trips.
        let dir = tempfile::tempdir().unwrap();
        let ws: WorkspaceId;
        let sid: SessionId;
        {
            let store = Store::open(dir.path().join("store"), true).unwrap();
            ws = store.create_workspace("/w").unwrap();
            let row = store.create_session(ws, "t", "p", "m").unwrap();
            sid = row.id;
            for i in 1..=2 {
                let tripped = store
                    .bump_loop_signal(sid, "fail run_command", 3, i * 1000)
                    .unwrap();
                assert!(!tripped, "count {i} must not trip yet");
            }
            // Reopen happens when the store drops (crash simulation).
        }
        {
            let store = Store::open(dir.path().join("store"), true).unwrap();
            assert!(
                store
                    .bump_loop_signal(sid, "fail run_command", 3, 3000)
                    .unwrap(),
                "third identical failure after a restart must trip"
            );
            // Progress clears everything.
            store.reset_loop_signals(sid).unwrap();
            assert!(store.loop_signal_counts(sid).unwrap().is_empty());
        }
    }

    #[test]
    fn migrate_and_reopen_is_stable() {
        let dir = tempfile::tempdir().unwrap();
        {
            let s = Store::open(dir.path(), true).unwrap();
            s.create_workspace("/tmp/ws").unwrap();
            s.create_session(WorkspaceId::new(1), "t", "ollama", "qwen3.8")
                .unwrap();
        }
        // Reopen: migrations must be a no-op and data must survive.
        let s = Store::open(dir.path(), true).unwrap();
        assert!(s.get_session(SessionId::new(1)).unwrap().is_some());
        assert_eq!(
            s.last_event_seq(SessionId::new(1)).unwrap().unwrap().raw(),
            1
        );
    }

    #[test]
    fn corrupt_db_file_is_detected_on_open() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("kilo-plus.db"),
            b"this is not a sqlite database at all - definitely not valid magic header bytes",
        )
        .unwrap();
        match Store::open(dir.path(), true) {
            Err(StoreError::Sqlite(_)) | Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt db must fail cleanly, got {other:?}"),
        }
        // Without integrity_check it may open (SQLite lazy), but any query
        // must error, not panic.
        let s = Store::open(dir.path(), false);
        if let Ok(s) = s {
            let r = s.list_sessions(None);
            assert!(r.is_err() || r.is_ok(), "never panic");
        }
    }

    #[test]
    fn truncated_db_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("kilo-plus.db"), b"SQLite format 3\x00").unwrap();
        match Store::open(dir.path(), true) {
            Err(StoreError::Sqlite(_)) | Err(StoreError::Corrupt(_)) => {}
            other => panic!("truncated db must fail cleanly, got {other:?}"),
        }
    }

    #[test]
    fn journal_sequences_are_gapless_under_concurrent_append() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let session = store.create_session(ws, "c", "p", "m").unwrap();
        let sid = session.id;
        let store = std::sync::Arc::new(store);
        let mut handles = vec![];
        for t in 0..8 {
            let store = store.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..50 {
                    store
                        .append_event(
                            sid,
                            Some(OpId::new(1 + t * 100 + i)),
                            EventKind::ModelChunkReceived,
                            AgentState::Streaming,
                            now_ms(),
                            Some(serde_json::json!({"i": i})),
                        )
                        .unwrap();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let events = store.events_range(sid, 1, None).unwrap();
        // SessionCreated(1) + 400 chunks = 401 events, seq 1..=401 gapless.
        assert_eq!(events.len(), 401);
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e.seq.raw(), (i + 1) as u64, "gap at {i}");
        }
        // Resume cursor semantics: events_after(seq 400) returns exactly 1.
        let tail = store.events_after(sid, EventSeq::new(400)).unwrap();
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].seq.raw(), 401);
    }

    #[test]
    fn message_paging_is_fundamental_and_stable() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let session = store.create_session(ws, "t", "p", "m").unwrap();
        for i in 0..100 {
            store
                .put_message(
                    session.id,
                    i,
                    "user",
                    serde_json::json!({"text": format!("msg {i}")}),
                )
                .unwrap();
        }
        // Newest first, page size 10.
        let page1 = store.messages_before(session.id, None, 10).unwrap();
        assert_eq!(page1.len(), 10);
        assert_eq!(page1[0].seq, 99);
        // Cursor paging reaches everything exactly once.
        let mut seen = vec![];
        let mut cursor = None;
        loop {
            let page = store.messages_before(session.id, cursor, 7).unwrap();
            if page.is_empty() {
                break;
            }
            for m in &page {
                assert!(!seen.contains(&m.seq), "duplicate message in paging");
                seen.push(m.seq);
            }
            cursor = Some(page.last().unwrap().seq);
        }
        assert_eq!(seen.len(), 100);
    }

    #[test]
    fn crash_recovery_scanner_input_is_durable() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let session = store.create_session(ws, "t", "p", "m").unwrap();
        let op = OpId::new(77);
        store
            .start_tool_run(
                session.id,
                op,
                "write_file",
                serde_json::json!({"path": "/w/a.txt", "content": "x"}),
                serde_json::json!({"strategy": "verify_hash", "detail": {"path": "/w/a.txt", "expected": "ab".repeat(32)}}),
                Some("ab".repeat(32)),
                None,
            )
            .unwrap();
        // Crash: no finish. The scanner must find it with effect unknown.
        let pending = store.pending_tool_runs(session.id).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].op_id, op);
        assert_eq!(pending[0].effect_status, "unknown");
        assert_eq!(pending[0].status, "running");
        assert_eq!(
            pending[0].expected_hash.as_deref(),
            Some("ab".repeat(32).as_str())
        );
        // Finishing moves it out of the scanner set.
        store
            .finish_tool_run(session.id, op, "completed", "verified")
            .unwrap();
        assert!(store.pending_tool_runs(session.id).unwrap().is_empty());
        // finish on missing row is an error (loud, not silent)
        assert!(store
            .finish_tool_run(session.id, OpId::new(999), "completed", "verified")
            .is_err());
    }

    #[test]
    fn checkpoints_dedup_by_hash_and_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let session_id = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            for i in 0..5 {
                store
                    .put_checkpoint(s.id, i, "a.rs", "hash-before", "hash-after", None)
                    .unwrap();
            }
            // v3: the after-blob hash roundtrips when recorded.
            store
                .put_checkpoint(s.id, 5, "b.rs", "b1", "a1", Some("cas-after-blob"))
                .unwrap();
            s.id
        };
        let store = Store::open(dir.path(), true).unwrap();
        let cps = store.checkpoints_of(session_id).unwrap();
        assert_eq!(cps.len(), 6);
        assert_eq!(cps[0].sequence, 0);
        assert_eq!(cps[4].after_hash, "hash-after");
        assert_eq!(cps[4].after_cas_hash, None);
        assert_eq!(cps[5].after_cas_hash.as_deref(), Some("cas-after-blob"));
    }

    #[test]
    fn migration_v3_keeps_pre_v3_checkpoint_rows_readable() {
        // Simulate a store that was created at v2 (checkpoints without the
        // after-blob column): open a fresh store, record a checkpoint, then
        // downgrade the schema behind the API's back (DROP COLUMN + set the
        // version cursor back). Reopening must apply v3, leave the old row
        // readable, and surface after_cas_hash as NULL — never a panic and
        // never a lost row.
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            store
                .put_checkpoint(s.id, 3, "f.txt", "before", "after", Some("after-blob"))
                .unwrap();
            {
                let conn = store.write();
                conn.execute("ALTER TABLE checkpoint DROP COLUMN after_cas_hash", [])
                    .unwrap();
                // The v6 existence columns are post-v2 too: drop them so the
                // full migration chain (v3..v6) replays on reopen.
                conn.execute("ALTER TABLE checkpoint DROP COLUMN before_exists", [])
                    .unwrap();
                conn.execute("ALTER TABLE checkpoint DROP COLUMN after_exists", [])
                    .unwrap();
                // The v7 tool-run recovery columns + turn-record table are
                // post-v2 too: drop them so the full migration chain
                // (v3..v7) replays on reopen.
                conn.execute("ALTER TABLE tool_run DROP COLUMN replay_descriptor", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN postcondition", [])
                    .unwrap();
                conn.execute("DROP TABLE turn_record", []).unwrap();
                // The v8 session-identity columns are post-v2 too: drop them
                // so the full migration chain (v3..v8) replays on reopen.
                conn.execute("ALTER TABLE session DROP COLUMN worktree_id", [])
                    .unwrap();
                conn.execute("ALTER TABLE session DROP COLUMN task_id", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 2", []).unwrap();
            }
            s.id
        };
        // Reopen: v3 re-applies the column; the pre-v3 row must read back
        // intact with after_cas_hash = NULL.
        let store = Store::open(dir.path(), true).unwrap();
        let cps = store.checkpoints_of(sid).unwrap();
        assert_eq!(cps.len(), 1, "the old row must survive the v3 migration");
        assert_eq!(cps[0].path, "f.txt");
        assert_eq!(cps[0].before_hash, "before");
        assert_eq!(cps[0].after_hash, "after");
        assert_eq!(cps[0].after_cas_hash, None);
        assert!(cps[0].created_ms > 0);
        // And the column is writable again.
        store
            .put_checkpoint(sid, 4, "g.txt", "b", "a", Some("x"))
            .unwrap();
        assert_eq!(
            store.checkpoints_of(sid).unwrap()[1]
                .after_cas_hash
                .as_deref(),
            Some("x")
        );
    }

    #[test]
    fn checkpoint_sequence_allocation_is_atomic_under_concurrent_writers() {
        // P1 "checkpoint numbering race": the old flow derived the sequence
        // from rows.len()+1 OUTSIDE the store, so two racing writers could
        // both receive the same N+1. insert_checkpoint must allocate
        // MAX(sequence)+1 and insert in ONE transaction: 8 writers × 25
        // checkpoints = 200 rows, all sequences distinct and gapless.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let store = std::sync::Arc::new(store);
        let mut handles = Vec::new();
        for t in 0..8 {
            let store = store.clone();
            let sid = s.id;
            handles.push(std::thread::spawn(move || {
                for i in 0..25 {
                    let (id, seq) = store
                        .insert_checkpoint(
                            sid,
                            &format!("f{t}-{i}.rs"),
                            true,
                            "before-hash",
                            true,
                            "after-hash",
                            None,
                        )
                        .unwrap();
                    assert!(id > 0);
                    assert!(seq >= 1, "sequence must be >= 1, got {seq}");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let rows = store.checkpoints_of(s.id).unwrap();
        assert_eq!(rows.len(), 200, "every racing insert must land");
        let mut seqs: Vec<i64> = rows.iter().map(|c| c.sequence).collect();
        seqs.sort_unstable();
        for (i, seq) in seqs.iter().enumerate() {
            assert_eq!(
                *seq,
                (i + 1) as i64,
                "sequence {seq} at slot {i}: must be gapless"
            );
        }
        // The duplicate-guard invariant: no two rows share a sequence.
        let unique: std::collections::HashSet<i64> = seqs.iter().copied().collect();
        assert_eq!(
            unique.len(),
            200,
            "two writers must never receive the same sequence"
        );
    }

    #[test]
    fn insert_checkpoint_roundtrips_existence_flags() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // missing -> non-empty file: the before side has NO content, so its
        // hash column is the empty-string sentinel, not a hash of nothing.
        let (id, seq) = store
            .insert_checkpoint(
                s.id,
                "created.rs",
                false,
                "",
                true,
                "after-hex",
                Some("after-blob-hex"),
            )
            .unwrap();
        assert!(id > 0);
        assert_eq!(seq, 1);
        // file -> deleted (second row): the after side does not exist.
        let (_, seq2) = store
            .insert_checkpoint(s.id, "deleted.rs", true, "before-hex", false, "", None)
            .unwrap();
        assert_eq!(seq2, 2, "allocation must continue the session sequence");
        let rows = store.checkpoints_of(s.id).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(
            !rows[0].before_exists,
            "missing side must read back as not existing"
        );
        assert!(rows[0].after_exists);
        assert_eq!(rows[0].before_hash, "", "no content -> no hash");
        assert_eq!(rows[0].after_hash, "after-hex");
        assert!(rows[1].before_exists);
        assert!(
            !rows[1].after_exists,
            "deleted side must read back as not existing"
        );
        assert_eq!(rows[1].after_hash, "");
        assert_eq!(rows[1].after_cas_hash, None);
    }

    #[test]
    fn migration_v6_keeps_pre_v6_checkpoint_rows_readable_as_existing() {
        // A store created at v5 records a checkpoint without existence
        // flags. Downgrade the schema behind the API's back (DROP the new
        // columns + rewind the version cursor), then reopen: v6 re-adds the
        // columns with DEFAULT 1 and the old row must read back as
        // exists:true on both sides (old rows only ever recorded real
        // files) — never a lost row, never a panic.
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            store
                .put_checkpoint(s.id, 1, "f.txt", "before", "after", Some("blob"))
                .unwrap();
            {
                let conn = store.write();
                conn.execute("ALTER TABLE checkpoint DROP COLUMN before_exists", [])
                    .unwrap();
                conn.execute("ALTER TABLE checkpoint DROP COLUMN after_exists", [])
                    .unwrap();
                // The v7 tool-run recovery columns + turn-record table are
                // post-v5 too: drop them so the full migration chain
                // (v6..v7) replays on reopen.
                conn.execute("ALTER TABLE tool_run DROP COLUMN replay_descriptor", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN postcondition", [])
                    .unwrap();
                conn.execute("DROP TABLE turn_record", []).unwrap();
                // The v8 session-identity columns are post-v5 too: drop them
                // so the full migration chain (v6..v8) replays on reopen.
                conn.execute("ALTER TABLE session DROP COLUMN worktree_id", [])
                    .unwrap();
                conn.execute("ALTER TABLE session DROP COLUMN task_id", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 5", []).unwrap();
            }
            s.id
        };
        let store = Store::open(dir.path(), true).unwrap();
        let cps = store.checkpoints_of(sid).unwrap();
        assert_eq!(cps.len(), 1, "the old row must survive the v6 migration");
        assert!(
            cps[0].before_exists && cps[0].after_exists,
            "pre-v6 rows have no existence marker: hash present means exists:true"
        );
        assert_eq!(cps[0].before_hash, "before");
        // And the new columns are writable again.
        store
            .insert_checkpoint(sid, "g.txt", false, "", true, "a", Some("x"))
            .unwrap();
        let rows = store.checkpoints_of(sid).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(!rows[1].before_exists);
        assert_eq!(
            rows[1].sequence, 2,
            "allocation continues after legacy rows"
        );
    }

    #[test]
    fn message_created_ms_queries_known_and_unknown() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .put_message(s.id, 5, "user", serde_json::json!({"text": "hi"}))
            .unwrap();
        let ms = store.message_created_ms(s.id, 5).unwrap().unwrap();
        assert!(ms > 0);
        // The same value the message row itself carries.
        assert_eq!(
            ms,
            store.messages_before(s.id, None, 10).unwrap()[0].created_ms
        );
        // Unknown seq → None, never an error.
        assert_eq!(store.message_created_ms(s.id, 99).unwrap(), None);
        assert_eq!(
            store.message_created_ms(SessionId::new(999), 5).unwrap(),
            None
        );
    }

    #[test]
    fn workspace_root_roundtrip_and_unknown() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/root/x").unwrap();
        assert_eq!(
            store.workspace_root(ws).unwrap().as_deref(),
            Some("/root/x")
        );
        assert_eq!(store.workspace_root(WorkspaceId::new(999)).unwrap(), None);
    }

    #[test]
    fn workspace_create_is_idempotent() {
        let (_d, store) = tmp_store();
        let a = store.create_workspace("/same").unwrap();
        let b = store.create_workspace("/same").unwrap();
        assert_eq!(a, b);
        let c = store.create_workspace("/other").unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn adversarial_duplicate_event_append_is_structurally_impossible() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // Try to insert a duplicate (session, seq) by bypassing the API:
        // the PRIMARY KEY must reject it.
        let conn = store.write();
        let r = conn.execute(
            "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload) VALUES (1, ?1, NULL, 'model_started', '\"streaming\"', 0, NULL)",
            params![s.id.raw() as i64],
        );
        assert!(r.is_err(), "duplicate (session,seq) must be rejected by PK");
    }

    #[test]
    fn memory_facts_upsert_and_query() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .upsert_memory_fact(s.id, "decision", "framework", "rust")
            .unwrap();
        store
            .upsert_memory_fact(s.id, "decision", "framework", "rust+tokio")
            .unwrap();
        let facts = store.memory_facts(s.id).unwrap();
        assert_eq!(facts.len(), 1, "upsert must not duplicate");
        assert_eq!(facts[0].2, "rust+tokio");
    }

    #[test]
    fn permissions_resolve_once_and_expire() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let op = OpId::new(5);
        let pid = store.insert_permission(s.id, op, "execute_shell").unwrap();
        let pending = store.pending_permission(pid).unwrap().unwrap();
        assert_eq!(pending.0, s.id);
        assert_eq!(pending.1, op);
        assert_eq!(pending.2, "execute_shell");
        store.resolve_permission(pid, "allow").unwrap();
        assert!(store.pending_permission(pid).unwrap().is_none());
        // Resolving again must not change anything (first decision wins).
        store.resolve_permission(pid, "deny").unwrap();
        assert_eq!(
            PermissionDecision::Allow,
            PermissionDecision::Allow,
            "first decision wins"
        );
    }

    #[test]
    fn backup_restores_full_state() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .put_message(s.id, 1, "user", serde_json::json!({"text": "hi"}))
            .unwrap();
        let backup_path = dir.path().join("backup.db");
        store.backup_to(&backup_path).unwrap();
        // Reopen backup as a store; data must be complete.
        let restored_dir = tempfile::tempdir().unwrap();
        std::fs::copy(&backup_path, restored_dir.path().join("kilo-plus.db")).unwrap();
        let restored = Store::open(restored_dir.path(), true).unwrap();
        assert_eq!(restored.message_count(s.id).unwrap(), 1);
        assert_eq!(
            restored.messages_before(s.id, None, 10).unwrap()[0].data["text"],
            "hi"
        );
    }

    #[test]
    fn integrity_check_survives_normal_use_and_flags_corruption() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let _s = store.create_session(ws, "t", "p", "m").unwrap();
        assert!(store.integrity_check().unwrap().is_empty());
        // Corrupt the DB file on disk behind the store's back: replace the
        // main file with garbage AND remove the WAL/shm sidecars so nothing
        // can paper over the corruption. A lazy reopen must either refuse to
        // open or flag the corruption on the next integrity check — never
        // silently serve a fake store.
        drop(store);
        let path = dir.path().join("kilo-plus.db");
        std::fs::write(
            &path,
            b"this file is complete garbage now, no sqlite magic header at all - 1234567890",
        )
        .unwrap();
        let _ = std::fs::remove_file(dir.path().join("kilo-plus.db-wal"));
        let _ = std::fs::remove_file(dir.path().join("kilo-plus.db-shm"));
        match Store::open(dir.path(), false) {
            Err(e) => {
                assert!(
                    matches!(e, StoreError::Sqlite(_) | StoreError::Corrupt(_)),
                    "corrupt db must fail cleanly, got {e:?}"
                );
            }
            Ok(reopened) => {
                let issues = reopened.integrity_check();
                assert!(
                    issues.is_err() || !issues.unwrap().is_empty(),
                    "corruption must surface as an error or flagged rows"
                );
            }
        }
    }

    #[test]
    fn session_state_tracks_journal() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        assert_eq!(s.state, AgentState::Idle);
        store
            .append_event(
                s.id,
                None,
                EventKind::PromptReceived,
                AgentState::Preparing,
                now_ms(),
                None,
            )
            .unwrap();
        let s2 = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(s2.state, AgentState::Preparing);
        assert!(s2.updated_ms >= s.updated_ms);
    }

    #[test]
    fn worktree_crud() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let id = store.put_worktree(ws, "/w/wt1", "feat/x").unwrap();
        assert_eq!(id, store.put_worktree(ws, "/w/wt1", "feat/x").unwrap());
        let list = store.worktrees_of(ws).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].branch, "feat/x");
        store.remove_worktree("/w/wt1").unwrap();
        assert!(store.worktrees_of(ws).unwrap().is_empty());
    }

    #[test]
    fn artifact_hash_unique_across_sessions() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s1 = store.create_session(ws, "a", "p", "m").unwrap();
        let s2 = store.create_session(ws, "b", "p", "m").unwrap();
        store
            .put_artifact(s1.id, "command_output", "hash1", "sum", 10)
            .unwrap();
        store
            .put_artifact(s2.id, "command_output", "hash1", "sum", 10)
            .unwrap();
        let a = store.artifact("hash1").unwrap().unwrap();
        assert_eq!(a.0, "sum");
        assert_eq!(store.artifact("nope").unwrap(), None);
    }

    #[test]
    fn diagnostic_smoke() {
        let (_d, store) = tmp_store();
        let d = store.diagnostics().unwrap();
        assert_eq!(d["journal_mode"], "wal");
    }

    #[test]
    fn giant_payload_roundtrip_via_cas_hash_reference() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let blob_hash = format!("{:064x}", 7);
        let big = serde_json::json!({"blob_hash": blob_hash});
        store
            .put_artifact(
                s.id,
                "tool_output",
                big["blob_hash"].as_str().unwrap(),
                "300MB compiler log",
                300_000_000,
            )
            .unwrap();
        assert_eq!(
            store
                .artifact(big["blob_hash"].as_str().unwrap())
                .unwrap()
                .unwrap()
                .1,
            "tool_output"
        );
        // A different hash is not found.
        assert_eq!(store.artifact(&"0".repeat(63)).unwrap(), None);
    }

    #[test]
    fn reader_pool_is_concurrency_bounded() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let store = std::sync::Arc::new(store);
        let mut handles = vec![];
        for _ in 0..20 {
            let store = store.clone();
            let sid = s.id;
            handles.push(std::thread::spawn(move || {
                for _ in 0..25 {
                    let conn = store.read().unwrap();
                    let n: i64 = conn
                        .query_row(
                            "SELECT COUNT(*) FROM session WHERE id = ?1",
                            params![sid.raw() as i64],
                            |r| r.get(0),
                        )
                        .unwrap();
                    assert_eq!(n, 1);
                    drop(conn);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        // 20 readers finished with the pool intact: retention never exceeds
        // the cap, and — the strong invariant the semaphore guarantees — the
        // number of connections ever opened never exceeds the cap either
        // (the old pool opened a new connection whenever it was empty).
        assert!(
            store.reader_pool_len() <= READER_POOL,
            "idle pool exceeds cap: {}",
            store.reader_pool_len()
        );
        assert!(
            store.connections_created() <= READER_POOL as u64,
            "connections created {} exceeds cap {}",
            store.connections_created(),
            READER_POOL
        );
    }

    #[test]
    fn reader_pool_waits_and_never_starves() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let _s = store.create_session(ws, "t", "p", "m").unwrap();
        let store = std::sync::Arc::new(store);
        // Hold all 4 permits, then prove a 5th reader waits (bounded) and
        // succeeds once a permit frees.
        let held: Vec<ReadConn> = (0..READER_POOL).map(|_| store.read().unwrap()).collect();
        let store2 = store.clone();
        let late = std::thread::spawn(move || {
            let conn = store2.read().unwrap(); // must block, then succeed
            let n: i64 = conn
                .query_row("SELECT COUNT(*) FROM session", [], |r| r.get(0))
                .unwrap();
            assert_eq!(n, 1);
        });
        std::thread::sleep(Duration::from_millis(50));
        assert!(!late.is_finished(), "5th reader must wait for a permit");
        drop(held);
        late.join().unwrap();
    }

    #[test]
    fn corrupt_state_row_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        {
            let conn = store.write();
            conn.execute(
                "UPDATE session SET state = ?1 WHERE id = ?2",
                params!["\"not_a_state\"", s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.get_session(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt session state must error, not panic: {other:?}"),
        }
        match store.list_sessions(Some(ws)) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt session state must error in list too: {other:?}"),
        }
    }

    #[test]
    fn corrupt_event_kind_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        {
            let conn = store.write();
            conn.execute(
                "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload)
                 VALUES (2, ?1, NULL, 'bogus', '\"idle\"', 0, NULL)",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.events_range(s.id, 1, None) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("unknown event kind must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_event_state_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        {
            let conn = store.write();
            conn.execute(
                "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload)
                 VALUES (2, ?1, NULL, 'model_started', '\"not_a_state\"', 0, NULL)",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.events_range(s.id, 1, None) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt event state must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_payload_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        {
            let conn = store.write();
            conn.execute(
                "INSERT INTO event(seq, session_id, op_id, kind, state, ts_ms, payload)
                 VALUES (2, ?1, NULL, 'model_started', '\"streaming\"', 0, 'not json at all')",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.events_range(s.id, 1, None) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt payload must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_message_and_part_data_return_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let mid = store
            .put_message(s.id, 1, "user", serde_json::json!({"text": "hi"}))
            .unwrap();
        let pid = store
            .put_part(mid, "text", serde_json::json!({"t": "hi"}))
            .unwrap();
        {
            let conn = store.write();
            conn.execute(
                "UPDATE message SET data = 'broken{' WHERE id = ?1",
                params![mid],
            )
            .unwrap();
            conn.execute(
                "UPDATE part SET data = 'also broken' WHERE id = ?1",
                params![pid],
            )
            .unwrap();
        }
        match store.messages_before(s.id, None, 10) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt message data must error, not panic: {other:?}"),
        }
        match store.parts_of(mid) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt part data must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_tool_run_args_and_recovery_return_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .start_tool_run(
                s.id,
                OpId::new(1),
                "write_file",
                serde_json::json!({"path": "/a"}),
                serde_json::json!({"strategy": "verify_hash"}),
                None,
                None,
            )
            .unwrap();
        {
            let conn = store.write();
            conn.execute(
                "UPDATE tool_run SET args = 'broken{' WHERE session_id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.pending_tool_runs(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt tool_run args must error, not panic: {other:?}"),
        }
        {
            let conn = store.write();
            conn.execute(
                "UPDATE tool_run SET args = '{\"a\":1}', recovery = 'broken{' WHERE session_id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.pending_tool_runs(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt tool_run recovery must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupt_task_ledger_returns_error_not_panic() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .put_task_ledger(s.id, serde_json::json!({"tasks": []}))
            .unwrap();
        {
            let conn = store.write();
            conn.execute(
                "UPDATE task SET ledger = 'garbage' WHERE session_id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.get_task_ledger(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("corrupt ledger must error, not panic: {other:?}"),
        }
    }

    #[test]
    fn corrupted_lifecycle_fails_closed() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // Fresh rows hold the schema DEFAULT literal `open` (bare, non-JSON);
        // that spelling must still read back as Open, not Corrupt.
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, kilop_core::state::SessionLifecycle::Open);
        // (1) Not valid JSON at all: fail closed, never silently Open.
        {
            let conn = store.write();
            conn.execute(
                "UPDATE session SET lifecycle = 'garbage-not-json' WHERE id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.get_session(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("garbage lifecycle must fail closed, got {other:?}"),
        }
        // (2) Structurally valid JSON but an unknown variant: also fail
        // closed (a real Closed must not reopen as Open).
        {
            let conn = store.write();
            conn.execute(
                "UPDATE session SET lifecycle = '\"terminated\"' WHERE id = ?1",
                params![s.id.raw() as i64],
            )
            .unwrap();
        }
        match store.get_session(s.id) {
            Err(StoreError::Corrupt(_)) => {}
            other => panic!("unknown lifecycle variant must fail closed, got {other:?}"),
        }
        // (3) Regression: structurally valid JSON with a VALID variant still
        // parses.
        store
            .set_session_lifecycle(s.id, kilop_core::state::SessionLifecycle::Suspended)
            .unwrap();
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(
            row.lifecycle,
            kilop_core::state::SessionLifecycle::Suspended
        );
        // (4) NULL lifecycle cannot occur: the v2 schema declares the column
        // NOT NULL DEFAULT 'open', so SQLite itself rejects a NULL write;
        // parse_lifecycle never sees None. Prove the constraint holds.
        {
            let conn = store.write();
            let err = conn
                .execute(
                    "UPDATE session SET lifecycle = NULL WHERE id = ?1",
                    params![s.id.raw() as i64],
                )
                .unwrap_err();
            assert!(
                matches!(err, rusqlite::Error::SqliteFailure(e, _) if e.code == rusqlite::ErrorCode::ConstraintViolation),
                "NOT NULL lifecycle must reject NULL writes, got {err:?}"
            );
        }
    }

    fn end_transition() -> SessionTransition {
        SessionTransition {
            expected_lifecycle: Some(SessionLifecycle::Open),
            new_lifecycle: Some(SessionLifecycle::Closed),
            expected_state: None,
            new_state: AgentState::Completed,
            event_kind: EventKind::SessionEnded,
            event_payload: None,
        }
    }

    #[test]
    fn transition_session_commits_atomically() {
        // The crash window: the old code updated lifecycle in one transaction
        // and appended SessionEnded in a second; a crash between them left
        // Closed-without-event or event-without-Closed. One call must produce
        // BOTH, durably, and a reopen must see them together.
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let seq = store
                .transition_session(s.id, None, end_transition())
                .unwrap();
            assert_eq!(seq.raw(), 2, "SessionCreated(1) + SessionEnded(2)");
            // Fresh read (same store): both sides of the transition visible.
            let row = store.get_session(s.id).unwrap().unwrap();
            assert_eq!(row.lifecycle, SessionLifecycle::Closed);
            assert_eq!(row.state, AgentState::Completed);
            let events = store.events_range(s.id, 1, None).unwrap();
            assert_eq!(events.len(), 2);
            assert_eq!(events[1].kind, EventKind::SessionEnded);
            s.id
        };
        // "Daemon restart": both persisted in the single transaction.
        let store = Store::open(dir.path(), true).unwrap();
        let row = store.get_session(sid).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Closed);
        assert_eq!(row.state, AgentState::Completed);
        let events = store.events_range(sid, 1, None).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].kind, EventKind::SessionEnded);
        assert_eq!(events[1].seq.raw(), 2);
    }

    #[test]
    fn transition_session_conflict_aborts_atomically() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .set_session_lifecycle(s.id, SessionLifecycle::Suspended)
            .unwrap();
        let err = store
            .transition_session(s.id, None, end_transition())
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Conflict(_)),
            "expected lifecycle mismatch must be Conflict, got {err:?}"
        );
        // Rollback proven: no event row appeared and nothing moved.
        assert_eq!(store.events_range(s.id, 1, None).unwrap().len(), 1);
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Suspended);
        assert_eq!(row.state, AgentState::Idle);
    }

    #[test]
    fn expected_state_mismatch_same() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        store
            .append_event(
                s.id,
                None,
                EventKind::PromptReceived,
                AgentState::Preparing,
                now_ms(),
                None,
            )
            .unwrap();
        let mut t = end_transition();
        // The turn machine is mid-turn (Preparing), not Idle.
        t.expected_state = Some(AgentState::Idle);
        let err = store.transition_session(s.id, None, t).unwrap_err();
        assert!(
            matches!(err, StoreError::Conflict(_)),
            "expected state mismatch must be Conflict, got {err:?}"
        );
        // No SessionEnded row, lifecycle still Open, state still Preparing.
        assert_eq!(store.last_event_seq(s.id).unwrap().unwrap().raw(), 2);
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Open);
        assert_eq!(row.state, AgentState::Preparing);
    }

    #[test]
    fn concurrent_end_session_races() {
        // Two (well, eight) racers try to close one session. The writer lock
        // serializes the transactions; the expected_lifecycle guard means
        // exactly ONE wins and exactly ONE SessionEnded event exists.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let store = std::sync::Arc::new(store);
        let mut handles = Vec::new();
        for _ in 0..8 {
            let store = store.clone();
            let sid = s.id;
            handles.push(std::thread::spawn(move || {
                store.transition_session(sid, None, end_transition())
            }));
        }
        let results: Vec<StoreResult<EventSeq>> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();
        let wins = results.iter().filter(|r| r.is_ok()).count();
        let conflicts = results
            .iter()
            .filter(|r| matches!(r, Err(StoreError::Conflict(_))))
            .count();
        assert_eq!(wins, 1, "exactly one racer must win");
        assert_eq!(conflicts, 7, "the rest must conflict, got {results:?}");
        let events = store.events_range(s.id, 1, None).unwrap();
        let ended = events
            .iter()
            .filter(|e| e.kind == EventKind::SessionEnded)
            .count();
        assert_eq!(ended, 1, "exactly one SessionEnded event");
        assert_eq!(events.len(), 2);
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Closed);
        assert_eq!(row.state, AgentState::Completed);
    }

    #[test]
    fn gapless_seq_across_transition() {
        // A transition_session event must continue the journal sequence after
        // regular appends — the shared insert path must be the SAME path.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        for i in 0..3 {
            store
                .append_event(
                    s.id,
                    Some(OpId::new(1 + i)),
                    EventKind::ModelChunkReceived,
                    AgentState::Streaming,
                    now_ms(),
                    None,
                )
                .unwrap();
        }
        let seq = store
            .transition_session(s.id, None, end_transition())
            .unwrap();
        assert_eq!(seq.raw(), 5, "created + 3 chunks + transition = seq 5");
        let events = store.events_range(s.id, 1, None).unwrap();
        assert_eq!(events.len(), 5);
        for (i, e) in events.iter().enumerate() {
            assert_eq!(e.seq.raw(), (i + 1) as u64, "gap at {i}");
        }
        assert_eq!(events[4].kind, EventKind::SessionEnded);
    }

    #[test]
    fn set_lifecycle_if_is_conditional() {
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        // Auto-resume (Suspended -> Open): true and journal untouched.
        store
            .set_session_lifecycle(s.id, SessionLifecycle::Suspended)
            .unwrap();
        assert!(store
            .set_lifecycle_if(s.id, SessionLifecycle::Suspended, SessionLifecycle::Open)
            .unwrap());
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Open);
        assert_eq!(store.last_event_seq(s.id).unwrap().unwrap().raw(), 1);
        // Wrong expectation: no update, no error.
        assert!(!store
            .set_lifecycle_if(s.id, SessionLifecycle::Suspended, SessionLifecycle::Closed)
            .unwrap());
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.lifecycle, SessionLifecycle::Open);
    }

    #[test]
    fn transition_session_missing_session_conflicts() {
        let (_d, store) = tmp_store();
        let err = store
            .transition_session(SessionId::new(999), None, end_transition())
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Conflict(_)),
            "missing session must be Conflict, got {err:?}"
        );
    }

    #[test]
    fn turn_record_lifecycle_exclusive_active_and_envelope() {
        // v7: at most ONE active logical-turn record per session; a new
        // admission finalizes stragglers; the envelope is updateable while
        // active; finish is idempotent.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "ollama", "qwen3.8").unwrap();
        let a = OpId::new(11);
        let b = OpId::new(22);
        store
            .start_turn_record(s.id, a, None, Some(2), "ollama", "qwen3.8", None)
            .unwrap();
        let rec = store.active_turn_record(s.id).unwrap().unwrap();
        assert_eq!(rec.turn_op_id, a);
        assert_eq!(rec.status, TURN_RECORD_ACTIVE);
        assert_eq!(rec.prompt_message_id, Some(2));
        // A second admission while the first is active finalizes the first.
        store
            .start_turn_record(s.id, b, Some(1), Some(5), "ollama", "m2", Some("v1"))
            .unwrap();
        let recs = store.turn_records_of(s.id).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].status, TURN_RECORD_FAILED, "straggler finalized");
        assert_eq!(recs[1].status, TURN_RECORD_ACTIVE);
        assert_eq!(recs[1].queue_seq, Some(1));
        assert_eq!(recs[1].variant.as_deref(), Some("v1"));
        assert_eq!(
            store.active_turn_record(s.id).unwrap().unwrap().turn_op_id,
            b
        );
        // Envelope update (per-message model override at drive start).
        assert!(store
            .set_turn_record_envelope(
                s.id,
                b,
                "ollama",
                "override-model",
                Some("v2"),
                Some("native")
            )
            .unwrap());
        let rec = store.turn_record_of(s.id, b).unwrap().unwrap();
        assert_eq!(rec.effective_model, "override-model");
        assert_eq!(rec.tool_mode.as_deref(), Some("native"));
        // Re-admission of the SAME op upserts the SAME record (crash between
        // claim and drive; the queue row is re-admitted) — never a phantom.
        store
            .start_turn_record(s.id, b, Some(1), Some(5), "ollama", "m3", None)
            .unwrap();
        assert_eq!(store.turn_records_of(s.id).unwrap().len(), 2);
        assert_eq!(
            store
                .turn_record_of(s.id, b)
                .unwrap()
                .unwrap()
                .effective_model,
            "m3"
        );
        // Finish transitions + idempotence.
        assert!(store
            .finish_turn_record(s.id, b, TURN_RECORD_COMPLETED)
            .unwrap());
        assert!(!store
            .finish_turn_record(s.id, b, TURN_RECORD_COMPLETED)
            .unwrap());
        assert!(store.active_turn_record(s.id).unwrap().is_none());
        assert_eq!(
            store.turn_record_of(s.id, b).unwrap().unwrap().status,
            TURN_RECORD_COMPLETED
        );
        // Invalid statuses are rejected loudly.
        assert!(store.finish_turn_record(s.id, b, "bogus").is_err());
        // Unknown op: no record.
        assert!(store.turn_record_of(s.id, OpId::new(99)).unwrap().is_none());
    }

    #[test]
    fn tool_run_recovery_columns_roundtrip_and_attempts() {
        // v7: the replay descriptor, the physical-attempt counter and the
        // workspace-write postcondition ride the tool_run row.
        let (_d, store) = tmp_store();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let op = OpId::new(7);
        store
            .start_tool_run(
                s.id,
                op,
                "echo",
                serde_json::json!({"x": 1}),
                serde_json::json!({"strategy": "idempotent"}),
                None,
                Some(serde_json::json!({"tool_name": "echo", "validated_args": {"x": 1}})),
            )
            .unwrap();
        let pending = store.pending_tool_runs(s.id).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].attempt, 0, "the original run is attempt 0");
        assert_eq!(
            pending[0].replay_descriptor.as_ref().unwrap()["tool_name"],
            "echo"
        );
        // Postcondition annotation (recorded at execution end, pre-finish).
        store
            .record_tool_postcondition(
                s.id,
                op,
                &serde_json::json!({
                    "workspace_id": ws.raw(),
                    "worktree_id": 1,
                    "relative_path": "a.txt",
                    "expected_hash": "ab".repeat(32),
                }),
            )
            .unwrap();
        assert_eq!(
            store.pending_tool_runs(s.id).unwrap()[0]
                .postcondition
                .as_ref()
                .unwrap()["relative_path"],
            "a.txt"
        );
        // A replay bumps the attempt counter of the SAME logical row.
        assert_eq!(store.bump_tool_run_attempt(s.id, op).unwrap(), 1);
        // Hostile annotation on a finished row is loud.
        store
            .finish_tool_run(s.id, op, "completed", "applied")
            .unwrap();
        assert!(store.pending_tool_runs(s.id).unwrap().is_empty());
        assert!(store
            .record_tool_postcondition(s.id, op, &serde_json::json!({}))
            .is_err());
        assert!(store.bump_tool_run_attempt(s.id, op).is_err());
    }

    #[test]
    fn tool_run_recovery_columns_and_turn_records_survive_reopen() {
        // Requirement 2c: the descriptor + postcondition + attempt survive a
        // daemon restart and still drive a replay.
        let dir = tempfile::tempdir().unwrap();
        let (sid, op) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let op = OpId::new(31);
            store
                .start_tool_run(
                    s.id,
                    op,
                    "echo",
                    serde_json::json!({"x": 1}),
                    serde_json::json!({"strategy": "idempotent"}),
                    None,
                    Some(serde_json::json!({"tool_name": "echo"})),
                )
                .unwrap();
            store
                .record_tool_postcondition(s.id, op, &serde_json::json!({"relative_path": "a.txt"}))
                .unwrap();
            store
                .start_turn_record(s.id, OpId::new(99), None, Some(2), "p", "m", None)
                .unwrap();
            (s.id, op)
        };
        let store = Store::open(dir.path(), true).unwrap();
        let pending = store.pending_tool_runs(sid).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].op_id, op);
        assert_eq!(pending[0].attempt, 0);
        assert_eq!(
            pending[0].replay_descriptor.as_ref().unwrap()["tool_name"],
            "echo"
        );
        assert_eq!(
            pending[0].postcondition.as_ref().unwrap()["relative_path"],
            "a.txt"
        );
        assert_eq!(store.bump_tool_run_attempt(sid, op).unwrap(), 1);
        let rec = store.turn_record_of(sid, OpId::new(99)).unwrap().unwrap();
        assert_eq!(rec.status, TURN_RECORD_ACTIVE);
        assert_eq!(rec.effective_model, "m");
    }

    #[test]
    fn migration_v7_replays_cleanly_on_a_v6_store() {
        // Simulate a store created before v7 (no turn_record table, no
        // tool_run recovery columns), then reopen: the v7 migration must
        // re-create everything without touching existing rows.
        let dir = tempfile::tempdir().unwrap();
        let sid = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            store
                .start_tool_run(
                    s.id,
                    OpId::new(1),
                    "run",
                    serde_json::json!({}),
                    serde_json::json!({"strategy": "none"}),
                    None,
                    None,
                )
                .unwrap();
            {
                let conn = store.write();
                conn.execute("ALTER TABLE tool_run DROP COLUMN replay_descriptor", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN attempt", [])
                    .unwrap();
                conn.execute("ALTER TABLE tool_run DROP COLUMN postcondition", [])
                    .unwrap();
                conn.execute("DROP TABLE turn_record", []).unwrap();
                // The v8 session-identity columns are post-v7 too: drop them
                // so the migration chain past v7 replays on reopen.
                conn.execute("ALTER TABLE session DROP COLUMN worktree_id", [])
                    .unwrap();
                conn.execute("ALTER TABLE session DROP COLUMN task_id", [])
                    .unwrap();
                // Pre-v7 stores sit at machine version 7 (the v6 comment
                // block covers TWO ALTER entries: before_exists and
                // after_exists); rewinding to 7 replays ONLY the v7 entry.
                conn.execute("PRAGMA user_version = 7", []).unwrap();
            }
            s.id
        };
        let store = Store::open(dir.path(), true).unwrap();
        // The pre-v7 row survived and is readable.
        let pending = store.pending_tool_runs(sid).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].attempt, 0);
        assert!(pending[0].replay_descriptor.is_none());
        assert!(pending[0].postcondition.is_none());
        // And the new machinery works on the migrated store.
        let op = OpId::new(2);
        store
            .start_tool_run(
                sid,
                op,
                "echo",
                serde_json::json!({}),
                serde_json::json!({"strategy": "idempotent"}),
                None,
                Some(serde_json::json!({"tool_name": "echo"})),
            )
            .unwrap();
        assert_eq!(store.bump_tool_run_attempt(sid, op).unwrap(), 1);
        store
            .start_turn_record(sid, OpId::new(9), None, Some(2), "p", "m", None)
            .unwrap();
        assert_eq!(store.turn_records_of(sid).unwrap().len(), 1);
        // Reopen again: still stable.
        let store = Store::open(dir.path(), true).unwrap();
        assert_eq!(store.turn_records_of(sid).unwrap().len(), 1);
    }

    #[test]
    fn session_identity_defaults_to_standalone_and_adoption_is_durable() {
        // v8: sessions default to worktree 1 / task 1 (the DOCUMENTED
        // standalone identity) and adopt_identity persists the real
        // worktree/task ids durably — a reopen must read them back.
        let dir = tempfile::tempdir().unwrap();
        let (sid, ws) = {
            let store = Store::open(dir.path().join("store"), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            let row = store.get_session(s.id).unwrap().unwrap();
            assert_eq!(row.worktree_id, WorktreeId::new(1), "standalone default");
            assert_eq!(row.task_id, TaskId::new(1), "standalone default");
            // Adoption moves the row off the defaults.
            store
                .adopt_session_identity(s.id, WorktreeId::new(7), TaskId::new(9))
                .unwrap();
            let row = store.get_session(s.id).unwrap().unwrap();
            assert_eq!(row.worktree_id, WorktreeId::new(7));
            assert_eq!(row.task_id, TaskId::new(9));
            // Unknown sessions are loud, not silent.
            assert!(store
                .adopt_session_identity(SessionId::new(9999), WorktreeId::new(2), TaskId::new(2))
                .is_err());
            (s.id, ws)
        };
        let store = Store::open(dir.path().join("store"), true).unwrap();
        let row = store.get_session(sid).unwrap().unwrap();
        assert_eq!(
            row.worktree_id,
            WorktreeId::new(7),
            "adoption survives reopen"
        );
        assert_eq!(row.task_id, TaskId::new(9));
        // list_sessions carries the same columns.
        let listed = store.list_sessions(Some(ws)).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].worktree_id, WorktreeId::new(7));
    }

    #[test]
    fn migration_v8_replays_cleanly_on_a_v7_store() {
        // Simulate a v7 store (no worktree_id/task_id columns on session),
        // reopen: v8 must add the columns and existing rows must read back
        // as the standalone 1/1 default — never a lost or corrupt row.
        // (Note: the v6 checkpoint block spans TWO array entries, so the
        // session-identity migration is array index 8 = schema target 9;
        // rewinding to 8 replays exactly this one entry.)
        let dir = tempfile::tempdir().unwrap();
        let (sid, ws) = {
            let store = Store::open(dir.path(), true).unwrap();
            let ws = store.create_workspace("/w").unwrap();
            let s = store.create_session(ws, "t", "p", "m").unwrap();
            {
                let conn = store.write();
                conn.execute("ALTER TABLE session DROP COLUMN worktree_id", [])
                    .unwrap();
                conn.execute("ALTER TABLE session DROP COLUMN task_id", [])
                    .unwrap();
                conn.execute("PRAGMA user_version = 8", []).unwrap();
            }
            (s.id, ws)
        };
        let store = Store::open(dir.path(), true).unwrap();
        let row = store.get_session(sid).unwrap().unwrap();
        assert_eq!(row.workspace_id, ws, "row survived the migration");
        assert_eq!(
            row.worktree_id,
            WorktreeId::new(1),
            "v8 default on old rows"
        );
        assert_eq!(row.task_id, TaskId::new(1), "v8 default on old rows");
        assert_eq!(store.list_sessions(None).unwrap().len(), 1);
    }

    #[test]
    fn update_session_title_roundtrip_and_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "old", "p", "m").unwrap();
        assert!(store.update_session_title(s.id, "new title").unwrap());
        let row = store.get_session(s.id).unwrap().unwrap();
        assert_eq!(row.title, "new title");
        assert!(row.updated_ms >= s.updated_ms, "updated_ms must bump");
        // Unknown sessions report false (nothing updated).
        assert!(!store
            .update_session_title(SessionId::new(9999), "x")
            .unwrap());
        // Durable across reopen.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        assert_eq!(store.get_session(s.id).unwrap().unwrap().title, "new title");
    }

    #[test]
    fn delete_message_removes_rows_and_parts_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path(), true).unwrap();
        let ws = store.create_workspace("/w").unwrap();
        let s = store.create_session(ws, "t", "p", "m").unwrap();
        let sid = s.id;
        store
            .put_message(sid, 1, "user", serde_json::json!({"text": "a"}))
            .unwrap();
        let m2 = store
            .put_message(sid, 2, "assistant", serde_json::json!({"parts": []}))
            .unwrap();
        store
            .put_part(m2, "tool_call", serde_json::json!({"tool_call_id": "c1"}))
            .unwrap();
        store
            .put_part(m2, "text", serde_json::json!({"text": "body"}))
            .unwrap();
        store
            .put_message(sid, 3, "user", serde_json::json!({"text": "b"}))
            .unwrap();
        // Delete the middle message: its part rows go with it.
        assert!(store.delete_message(sid, 2).unwrap());
        assert_eq!(store.message_count(sid).unwrap(), 2);
        assert!(store.parts_of(m2).unwrap().is_empty());
        // No orphan part rows can survive (single transaction).
        let orphans: i64 = {
            let conn = store.read().unwrap();
            conn.query_row(
                "SELECT COUNT(*) FROM part WHERE message_id NOT IN (SELECT id FROM message)",
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        assert_eq!(orphans, 0, "parts are removed with their message");
        // Sequences of surviving rows are STABLE (no renumbering).
        let page = store.messages_before(sid, None, 10).unwrap();
        let seqs: Vec<i64> = page.iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![3, 1]);
        // Re-removal of the same message deletes nothing and says so.
        assert!(!store.delete_message(sid, 2).unwrap());
        assert!(!store.delete_message(sid, 99).unwrap());
        // The removal is durable across a reopen.
        drop(store);
        let store = Store::open(dir.path(), true).unwrap();
        assert_eq!(store.message_count(sid).unwrap(), 2);
        assert!(store.message_created_ms(sid, 2).unwrap().is_none());
        let page = store.messages_before(sid, None, 10).unwrap();
        let seqs: Vec<i64> = page.iter().map(|r| r.seq).collect();
        assert_eq!(seqs, vec![3, 1]);
    }
}
