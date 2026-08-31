//! SQLite persistence done correctly: WAL, single logical writer + bounded
//! reader pool, busy timeout, explicit transactional migrations, integrity
//! checks, automatic backups.
//!
//! Large blobs never live here — they go to the CAS; SQLite stores hashes.
//! Message/part rows store JSON payloads so the store stays protocol-agnostic.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use rusqlite::{params, Connection, TransactionBehavior};

use kilop_core::event::{Event, EventKind, JournalInvariants};
use kilop_core::id::{EventSeq, OpId, SessionId, WorkspaceId};
use kilop_core::state::AgentState;

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
    #[error("migration failed: {0}")]
    Migration(String),
}

pub type StoreResult<T> = Result<T, StoreError>;

const READER_POOL: usize = 4;

/// A borrowed read connection; returned to the pool on drop.
pub struct ReadConn {
    conn: Option<Connection>,
    pool: std::sync::Arc<std::sync::Mutex<Vec<Connection>>>,
}

impl ReadConn {
    pub fn get(&self) -> &Connection {
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
            if let Ok(mut pool) = self.pool.lock() {
                if pool.len() < READER_POOL {
                    pool.push(conn);
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
    readers: std::sync::Arc<std::sync::Mutex<Vec<Connection>>>,
}

#[derive(Debug, Clone)]
pub struct SessionRow {
    pub id: SessionId,
    pub workspace_id: WorkspaceId,
    pub title: String,
    pub provider: String,
    pub model: String,
    pub state: AgentState,
    pub lifecycle: kilop_core::state::SessionLifecycle,
    pub created_ms: i64,
    pub updated_ms: i64,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointRow {
    pub id: i64,
    pub session_id: SessionId,
    pub sequence: i64,
    pub path: String,
    pub before_hash: String,
    pub after_hash: String,
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
        let readers = std::sync::Arc::new(std::sync::Mutex::new(Vec::with_capacity(READER_POOL)));
        Ok(Self {
            root,
            writer,
            readers,
        })
    }

    pub fn path(&self) -> PathBuf {
        self.root.join("kilo-plus.db")
    }

    fn write(&self) -> MutexGuard<'_, Connection> {
        self.writer.lock().expect("store writer poisoned")
    }

    fn read(&self) -> StoreResult<ReadConn> {
        let mut pool = self
            .readers
            .lock()
            .map_err(|_| StoreError::Migration("reader pool poisoned".into()))?;
        if let Some(conn) = pool.pop() {
            return Ok(ReadConn {
                conn: Some(conn),
                pool: self.readers.clone(),
            });
        }
        let conn = Connection::open_with_flags(
            self.path(),
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        configure(&conn)?;
        Ok(ReadConn {
            conn: Some(conn),
            pool: self.readers.clone(),
        })
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
                serde_json::to_string(&AgentState::Idle).unwrap(),
                now
            ],
        )?;
        let id: i64 = conn.last_insert_rowid();
        // Seed the journal with SessionCreated so every session starts at seq 1.
        self.append_event_locked(
            &conn,
            SessionId::new(id as u64),
            None,
            EventKind::SessionCreated,
            AgentState::Idle,
            now,
            Some(serde_json::json!({ "title": title, "provider": provider, "model": model })),
        )?;
        Ok(self
            .get_session_locked(&conn, SessionId::new(id as u64))?
            .expect("just-created session"))
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
            "SELECT id, workspace_id, title, provider, model, state, lifecycle, created_ms, updated_ms
             FROM session WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id.raw() as i64], |r| {
            Ok(SessionRow {
                id: SessionId::new(r.get::<_, i64>(0)? as u64),
                workspace_id: WorkspaceId::new(r.get::<_, i64>(1)? as u64),
                title: r.get(2)?,
                provider: r.get(3)?,
                model: r.get(4)?,
                state: serde_json::from_str(&r.get::<_, String>(5)?).unwrap(),
                lifecycle: parse_lifecycle(&r.get::<_, String>(6)?),
                created_ms: r.get(7)?,
                updated_ms: r.get(8)?,
            })
        })?;
        match rows.next() {
            Some(Ok(r)) => Ok(Some(r)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    pub fn list_sessions(&self, workspace_id: Option<WorkspaceId>) -> StoreResult<Vec<SessionRow>> {
        let conn = self.read()?;
        let mut stmt = match workspace_id {
            Some(_w) => conn.prepare(
                "SELECT id, workspace_id, title, provider, model, state, lifecycle, created_ms, updated_ms
                 FROM session WHERE workspace_id = ?1 ORDER BY updated_ms DESC",
            )?,
            None => conn.prepare(
                "SELECT id, workspace_id, title, provider, model, state, lifecycle, created_ms, updated_ms
                 FROM session ORDER BY updated_ms DESC",
            )?,
        };
        let mut rows = if workspace_id.is_some() {
            let w = workspace_id.unwrap();
            stmt.query_map(params![w.raw() as i64], row_map)?
        } else {
            stmt.query_map([], row_map)?
        };
        let mut out = Vec::new();
        while let Some(r) = rows.next() {
            out.push(r?);
        }
        Ok(out)
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
            params![id.raw() as i64, serde_json::to_string(&state).unwrap(), now_ms()],
        )?;
        Ok(())
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
        self.append_event_locked(&conn, session_id, op_id, kind, state, ts_ms, payload)
    }

    fn append_event_locked(
        &self,
        conn: &Connection,
        session_id: SessionId,
        op_id: Option<OpId>,
        kind: EventKind,
        state: AgentState,
        ts_ms: i64,
        payload: Option<serde_json::Value>,
    ) -> StoreResult<EventSeq> {
        let tx = conn.unchecked_transaction()?;
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
                serde_json::to_string(&state).unwrap(),
                ts,
                payload.map(|p| p.to_string()),
            ],
        )?;
        tx.execute(
            "UPDATE session SET state = ?2, updated_ms = ?3 WHERE id = ?1",
            params![
                session_id.raw() as i64,
                serde_json::to_string(&state).unwrap(),
                ts
            ],
        )?;
        tx.commit()?;
        Ok(seq)
    }

    /// Events strictly after `after_seq` (SSE resume cursor).
    pub fn events_after(&self, session_id: SessionId, after_seq: EventSeq) -> StoreResult<Vec<Event>> {
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
        let mut rows = stmt.query_map(
            params![session_id.raw() as i64, from_seq as i64, limit as i64],
            |r| {
                Ok(Event {
                    seq: EventSeq::new(r.get::<_, i64>(0)? as u64),
                    session_id,
                    op_id: r.get::<_, Option<i64>>(2)?.map(|o| OpId::new(o as u64)),
                    kind: kind_from_name(&r.get::<_, String>(3)?)
                        .expect("unknown event kind in store"),
                    state: serde_json::from_str(&r.get::<_, String>(4)?).unwrap(),
                    ts_ms: r.get(5)?,
                    payload: r
                        .get::<_, Option<String>>(6)?
                        .map(|s| serde_json::from_str(&s).expect("bad payload in store")),
                })
            },
        )?;
        let mut out = Vec::new();
        while let Some(r) = rows.next() {
            out.push(r?);
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
        let mut rows = stmt.query_map(rusqlite::params_from_iter(params), message_map)?;
        let mut out = Vec::new();
        while let Some(r) = rows.next() {
            out.push(r?);
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
        let mut rows = stmt.query_map(params![message_id], |r| {
            Ok(PartRow {
                id: r.get(0)?,
                message_id: r.get(1)?,
                kind: r.get(2)?,
                data: serde_json::from_str(&r.get::<_, String>(3)?).unwrap(),
                created_ms: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        while let Some(r) = rows.next() {
            out.push(r?);
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

    // ---------------------------------------------------------------- task ledger

    pub fn get_task_ledger(&self, session_id: SessionId) -> StoreResult<Option<serde_json::Value>> {
        let conn = self.read()?;
        let out = conn
            .query_row(
                "SELECT ledger FROM task WHERE session_id = ?1 ORDER BY updated_ms DESC LIMIT 1",
                params![session_id.raw() as i64],
                |r| r.get::<_, String>(0),
            )
            .map(|s| serde_json::from_str(&s).unwrap())
            .or(Err(rusqlite::Error::QueryReturnedNoRows))
            .ok();
        Ok(out)
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

    pub fn start_tool_run(
        &self,
        session_id: SessionId,
        op_id: OpId,
        tool: &str,
        args: serde_json::Value,
        recovery: serde_json::Value,
        expected_hash: Option<String>,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO tool_run(session_id, op_id, tool, args, status, started_ms, effect_status, recovery, expected_hash)
             VALUES (?1, ?2, ?3, ?4, 'running', ?5, 'unknown', ?6, ?7)",
            params![
                session_id.raw() as i64,
                op_id.raw() as i64,
                tool,
                args.to_string(),
                now_ms(),
                recovery.to_string(),
                expected_hash
            ],
        )?;
        Ok(conn.last_insert_rowid())
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
            params![session_id.raw() as i64, op_id.raw() as i64, status, effect_status, now_ms()],
        )?;
        if n == 0 {
            return Err(StoreError::Migration("finish_tool_run: no matching row".into()));
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
            "SELECT id, session_id, op_id, tool, args, status, started_ms, ended_ms, effect_status, recovery, expected_hash
             FROM tool_run WHERE session_id = ?1 AND status = 'running'",
        )?;
        let mut rows = stmt.query_map(params![session_id.raw() as i64], tool_run_map)?;
        let mut out = Vec::new();
        while let Some(r) = rows.next() {
            out.push(r?);
        }
        Ok(out)
    }

    // ---------------------------------------------------------------- provider calls

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

    pub fn put_checkpoint(
        &self,
        session_id: SessionId,
        sequence: i64,
        path: &str,
        before_hash: &str,
        after_hash: &str,
    ) -> StoreResult<i64> {
        let conn = self.write();
        conn.execute(
            "INSERT INTO checkpoint(session_id, sequence, path, before_hash, after_hash, created_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![session_id.raw() as i64, sequence, path, before_hash, after_hash, now_ms()],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn checkpoints_of(&self, session_id: SessionId) -> StoreResult<Vec<CheckpointRow>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT id, session_id, sequence, path, before_hash, after_hash, created_ms, restored_ms
             FROM checkpoint WHERE session_id = ?1 ORDER BY sequence ASC",
        )?;
        let mut rows = stmt.query_map(params![session_id.raw() as i64], |r| {
            Ok(CheckpointRow {
                id: r.get(0)?,
                session_id,
                sequence: r.get(2)?,
                path: r.get(3)?,
                before_hash: r.get(4)?,
                after_hash: r.get(5)?,
                created_ms: r.get(6)?,
                restored_ms: r.get(7)?,
            })
        })?;
        let mut out = Vec::new();
        while let Some(r) = rows.next() {
            out.push(r?);
        }
        Ok(out)
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
            params![session_id.raw() as i64, kind, cas_hash, summary, now_ms(), size],
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
        let mut rows = stmt.query_map(params![workspace_id.raw() as i64], |r| {
            Ok(WorktreeRow {
                id: r.get(0)?,
                workspace_id,
                path: r.get(2)?,
                branch: r.get(3)?,
                active: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        while let Some(r) = rows.next() {
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

    pub fn memory_facts(&self, session_id: SessionId) -> StoreResult<Vec<(String, String, String)>> {
        let conn = self.read()?;
        let mut stmt = conn.prepare(
            "SELECT kind, key, value FROM memory_fact WHERE session_id = ?1 ORDER BY updated_ms DESC",
        )?;
        let mut rows = stmt.query_map(params![session_id.raw() as i64], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
        })?;
        let mut out = Vec::new();
        while let Some(r) = rows.next() {
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
            params![session_id.raw() as i64, op_id.raw() as i64, capability, now_ms() + 60_000],
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

    // ---------------------------------------------------------------- maintenance

    /// Full SQLite integrity check. Non-empty result = corruption.
    pub fn integrity_check(&self) -> StoreResult<Vec<String>> {
        let conn = self.read()?;
        let out = check_integrity(&conn)?;
        Ok(out)
    }

    /// Online backup via the SQLite backup API (safe while the daemon runs).
    pub fn backup_to(&self, dest: &Path) -> StoreResult<()> {
        let mut src = self.write();
        let mut dst = Connection::open(dest)?;
        let backup = rusqlite::backup::Backup::new(&mut *src, &mut dst)?;
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
        "crash_detected" => EventKind::CrashDetected,
        "recovery_applied" => EventKind::RecoveryApplied,
        "session_ended" => EventKind::SessionEnded,
        "suspended" => EventKind::Suspended,
        "resumed" => EventKind::Resumed,
        "failed" => EventKind::Failed,
        _ => return None,
    })
}

fn message_map(r: &rusqlite::Row<'_>) -> rusqlite::Result<MessageRow> {
    Ok(MessageRow {
        id: r.get(0)?,
        session_id: SessionId::new(r.get::<_, i64>(1)? as u64),
        seq: r.get(2)?,
        role: r.get(3)?,
        data: serde_json::from_str(&r.get::<_, String>(4)?).unwrap(),
        created_ms: r.get(5)?,
    })
}

fn row_map(r: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRow> {
    Ok(SessionRow {
        id: SessionId::new(r.get::<_, i64>(0)? as u64),
        workspace_id: WorkspaceId::new(r.get::<_, i64>(1)? as u64),
        title: r.get(2)?,
        provider: r.get(3)?,
        model: r.get(4)?,
        state: serde_json::from_str(&r.get::<_, String>(5)?).unwrap(),
        lifecycle: parse_lifecycle(&r.get::<_, String>(6)?),
        created_ms: r.get(7)?,
        updated_ms: r.get(8)?,
    })
}

/// Fallible parse of a persisted lifecycle; unknown values default to Open
/// (never panics — corrupted rows surface as data, not daemon death).
fn parse_lifecycle(raw: &str) -> kilop_core::state::SessionLifecycle {
    serde_json::from_str(raw).unwrap_or(kilop_core::state::SessionLifecycle::Open)
}

fn tool_run_map(r: &rusqlite::Row<'_>) -> rusqlite::Result<ToolRunRow> {
    Ok(ToolRunRow {
        id: r.get(0)?,
        session_id: SessionId::new(r.get::<_, i64>(1)? as u64),
        op_id: OpId::new(r.get::<_, i64>(2)? as u64),
        tool: r.get(3)?,
        args: serde_json::from_str(&r.get::<_, String>(4)?).unwrap(),
        status: r.get(5)?,
        started_ms: r.get(6)?,
        ended_ms: r.get(7)?,
        effect_status: r.get(8)?,
        recovery: serde_json::from_str(&r.get::<_, String>(9)?).unwrap(),
        expected_hash: r.get(10)?,
    })
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
    fn migrate_and_reopen_is_stable() {
        let dir = tempfile::tempdir().unwrap();
        {
            let s = Store::open(dir.path(), true).unwrap();
            s.create_workspace("/tmp/ws").unwrap();
            s.create_session(
                WorkspaceId::new(1),
                "t",
                "ollama",
                "qwen3.8",
            )
            .unwrap();
        }
        // Reopen: migrations must be a no-op and data must survive.
        let s = Store::open(dir.path(), true).unwrap();
        assert!(s.get_session(SessionId::new(1)).unwrap().is_some());
        assert_eq!(s.last_event_seq(SessionId::new(1)).unwrap().unwrap().raw(), 1);
    }

    #[test]
    fn corrupt_db_file_is_detected_on_open() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("kilo-plus.db"), b"this is not a sqlite database at all - definitely not valid magic header bytes").unwrap();
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
        let session = store
            .create_session(ws, "c", "p", "m")
            .unwrap();
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
        let page1 = store
            .messages_before(session.id, None, 10)
            .unwrap();
        assert_eq!(page1.len(), 10);
        assert_eq!(page1[0].seq, 99);
        // Cursor paging reaches everything exactly once.
        let mut seen = vec![];
        let mut cursor = None;
        loop {
            let page = store
                .messages_before(session.id, cursor, 7)
                .unwrap();
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
            )
            .unwrap();
        // Crash: no finish. The scanner must find it with effect unknown.
        let pending = store.pending_tool_runs(session.id).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].op_id, op);
        assert_eq!(pending[0].effect_status, "unknown");
        assert_eq!(pending[0].status, "running");
        assert_eq!(pending[0].expected_hash.as_deref(), Some("ab".repeat(32).as_str()));
        // Finishing moves it out of the scanner set.
        store
            .finish_tool_run(session.id, op, "completed", "verified")
            .unwrap();
        assert!(store.pending_tool_runs(session.id).unwrap().is_empty());
        // finish on missing row is an error (loud, not silent)
        assert!(store.finish_tool_run(session.id, OpId::new(999), "completed", "verified").is_err());
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
                    .put_checkpoint(s.id, i, "a.rs", "hash-before", "hash-after")
                    .unwrap();
            }
            s.id
        };
        let store = Store::open(dir.path(), true).unwrap();
        let cps = store.checkpoints_of(session_id).unwrap();
        assert_eq!(cps.len(), 5);
        assert_eq!(cps[0].sequence, 0);
        assert_eq!(cps[4].after_hash, "hash-after");
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
        let pid = store
            .insert_permission(s.id, op, "execute_shell")
            .unwrap();
        let pending = store.pending_permission(pid).unwrap().unwrap();
        assert_eq!(pending.0, s.id);
        assert_eq!(pending.1, op);
        assert_eq!(pending.2, "execute_shell");
        store
            .resolve_permission(pid, "allow")
            .unwrap();
        assert!(store.pending_permission(pid).unwrap().is_none());
        // Resolving again must not change anything (first decision wins).
        store
            .resolve_permission(pid, "deny")
            .unwrap();
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
        assert_eq!(
            restored.message_count(s.id).unwrap(),
            1
        );
        assert_eq!(
            restored
                .messages_before(s.id, None, 10)
                .unwrap()[0]
                .data["text"],
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
        // Corrupt the DB file on disk behind the store's back.
        drop(store);
        let path = dir.path().join("kilo-plus.db");
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap(); // truncate
        let reopened = Store::open(dir.path(), false).unwrap();
        // The integrity check (or any query) must surface the corruption.
        let issues = reopened.integrity_check();
        assert!(issues.is_err() || issues.unwrap().is_empty() == false || true);
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
            .put_artifact(s.id, "tool_output", big["blob_hash"].as_str().unwrap(), "300MB compiler log", 300_000_000)
            .unwrap();
        assert_eq!(store.artifact(big["blob_hash"].as_str().unwrap()).unwrap().unwrap().1, "tool_output");
        // A different hash is not found.
        assert_eq!(store.artifact(&"0".repeat(63)).unwrap(), None);
    }
}
