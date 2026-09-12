//! The durable agent-coordination board (parent/descendant scoped).
//!
//! One board exists per RUN FAMILY: its id is derived from the family ROOT
//! session, and every board row lives in the root session's append-only
//! typed ledger as an additive ledger entry ([`crate::ledger::ENTRY_BOARD_POST`],
//! [`crate::ledger::ENTRY_BOARD_READ`], [`crate::ledger::ENTRY_BOARD_RECEIPT`],
//! [`crate::ledger::ENTRY_BOARD_RESET`]). No migration, no new table: board
//! rows are typed wave-11 ledger entries, strictly bounded at append AND at
//! decode, and PINNED across compaction — the pinned stream is the durable
//! authority a board read reconstructs the live surface from after a crash
//! or reopen.
//!
//! Semantics
//! ---------
//! - A `BoardPost` carries a per-board monotonic `revision`; its durable id
//!   EQUALS that revision (revisions are never reused — a reset consumes
//!   one), so the identity is stable and recoverable from the pinned stream.
//! - A `BoardRead { child, post_id, read_ms }` is recorded (idempotently)
//!   the first time a child reads a post, which is what
//!   [`SessionHandle::board_unread_counts`] folds over.
//! - A `BoardReceipt { child, post_id, action, note }` records an explicit
//!   action (`ack` / `task_update` / `blocked` / `question`).
//! - [`SessionHandle::board_reset`] CAS-checks `expected_revision` and
//!   appends ONE durable reset marker that bumps the revision by exactly
//!   one (`board_reset` payload). History before the marker is retained in
//!   the stream but hidden from reads: readers see only posts whose
//!   revision is ABOVE the newest marker's `new_revision`. A concurrent
//!   reset with a stale `expected_revision` is a typed `Conflict`.
//!
//! Scoping
//! -------
//! Every operation resolves the actor's family root by walking the durable
//! child identity rows (bounded depth, cycle-detecting). A board/target id
//! from ANOTHER run family is a typed permission refusal (`access denied`)
//! even when the caller knows every raw id; a child may act only for
//! itself, the root may act for any descendant. Posting (and action
//! receipts) from a TERMINAL child are refused with the same lifecycle rule
//! the steer path applies (session lifecycle/state terminal, or the child's
//! durable runtime projection reporting a terminal state) — before any
//! byte is written.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use faktor_core::id::SessionId;

use crate::handle::SessionHandle;
use crate::ledger::{
    validate_board_post, validate_board_receipt, validate_board_reset, LedgerPayload,
};
use crate::{map_store_err, SessionError};

pub use crate::ledger::{
    BOARD_RECEIPT_ACK, BOARD_RECEIPT_BLOCKED, BOARD_RECEIPT_QUESTION, BOARD_RECEIPT_TASK_UPDATE,
    MAX_BOARD_BODY_BYTES, MAX_BOARD_RECEIPT_NOTE_BYTES, MAX_BOARD_REFS, MAX_BOARD_REF_BYTES,
    MAX_BOARD_SUBJECT_BYTES,
};

/// Bounded page size of one `board_read` (paging is fundamental).
pub const MAX_BOARD_PAGE: usize = 100;
/// Hard cap on rows one board page read may scan (a hostile read surface
/// cannot stall a turn with an unbounded descending walk); exceeding it is
/// a typed `Oversized` refusal, never a silent partial page.
pub const MAX_BOARD_SCAN_ROWS: usize = 100_000;
/// Max ancestors walked when resolving a run family root (zero-orphan
/// contract: deeper chains are hostile/cyclic and refuse loudly).
pub const MAX_BOARD_FAMILY_DEPTH: usize = 64;
/// Hard cap on receipts returned for one `(child, post)` pair.
pub const MAX_BOARD_RECEIPTS_PER_POST: usize = crate::ledger::MAX_BOARD_RECEIPTS_PER_POST;

// ---------------------------------------------------------------- model

/// Durable identity of one run-family board: the ROOT session's raw id (one
/// board per run family).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BoardId(SessionId);

impl BoardId {
    /// The board of a family root.
    pub fn of_root(root: SessionId) -> Self {
        Self(root)
    }

    pub fn root(self) -> SessionId {
        self.0
    }
}

impl fmt::Display for BoardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "board:{}", self.0)
    }
}

/// Durable identity of an acting child: the child SESSION's raw id. `None`
/// in a post/receipt means the run ROOT (the parent agent itself).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ChildId(SessionId);

impl ChildId {
    pub fn of_session(session: SessionId) -> Self {
        Self(session)
    }

    pub fn session(self) -> SessionId {
        self.0
    }
}

impl fmt::Display for ChildId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "child:{}", self.0)
    }
}

/// Durable id of one board post: equals the revision that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BoardPostId(u64);

impl BoardPostId {
    /// Construct from a non-zero raw post id (internal hot path).
    ///
    /// # Panics
    /// Panics when `raw == 0`.
    pub const fn new(raw: u64) -> Self {
        assert!(raw != 0, "BoardPostId cannot be 0");
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for BoardPostId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Typed action vocabulary of a board receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoardAction {
    Ack,
    TaskUpdate,
    Blocked,
    Question,
}

impl BoardAction {
    pub const fn as_str(self) -> &'static str {
        match self {
            BoardAction::Ack => BOARD_RECEIPT_ACK,
            BoardAction::TaskUpdate => BOARD_RECEIPT_TASK_UPDATE,
            BoardAction::Blocked => BOARD_RECEIPT_BLOCKED,
            BoardAction::Question => BOARD_RECEIPT_QUESTION,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            BOARD_RECEIPT_ACK => Some(BoardAction::Ack),
            BOARD_RECEIPT_TASK_UPDATE => Some(BoardAction::TaskUpdate),
            BOARD_RECEIPT_BLOCKED => Some(BoardAction::Blocked),
            BOARD_RECEIPT_QUESTION => Some(BoardAction::Question),
            _ => None,
        }
    }
}

/// One durable board post, as read from the pinned ledger stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardPost {
    pub id: BoardPostId,
    pub board_id: BoardId,
    /// The acting child's session id; `None` = the root agent posted.
    pub author_child: Option<ChildId>,
    pub author_session: SessionId,
    pub subject: String,
    pub body: String,
    /// Evidence / path / artifact references (bounded).
    pub refs: Vec<String>,
    /// Per-board monotonic revision (`id == revision`).
    pub revision: u64,
    pub created_ms: i64,
}

/// One durable read receipt: `child` read `post_id` at `read_ms`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardRead {
    pub child: ChildId,
    pub post_id: BoardPostId,
    pub read_ms: i64,
}

/// One durable action receipt on a board post.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardReceipt {
    /// `None` = the root agent acted.
    pub child: Option<ChildId>,
    pub post_id: BoardPostId,
    pub action: BoardAction,
    pub note: String,
    pub created_ms: i64,
}

/// Both receipt kinds of one `(child, post)` pair: the (idempotent) read
/// marker plus the ordered action receipts.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BoardReceipts {
    pub read: Option<BoardRead>,
    pub actions: Vec<BoardReceipt>,
}

/// The outcome of a successful CAS reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardReset {
    pub board_id: BoardId,
    pub previous_revision: u64,
    pub new_revision: u64,
}

/// One bounded newest-first page of the live board surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardPage {
    pub board_id: BoardId,
    /// The board revision at read time (newest post/reset revision).
    pub revision: u64,
    /// Visible posts, newest first.
    pub posts: Vec<BoardPost>,
    /// Exclusive cursor for the next OLDER page: pass it as
    /// `since_revision` to [`SessionHandle::board_read_posts`]. `None` when
    /// the page exhausted the live surface.
    pub next_before_revision: Option<u64>,
    pub has_more: bool,
}

/// The DERIVED delivery view of one live post for one recipient. No new
/// ledger stream is introduced: every state is folded from rows that already
/// exist (posts/reads/receipts) plus the recipient's durable lifecycle.
///
/// Precedence (strongest evidence first):
/// 1. [`BoardDeliveryView::ActedUpon`] — an action receipt from the
///    recipient exists (read or not);
/// 2. [`BoardDeliveryView::Read`] — a durable read marker exists;
/// 3. [`BoardDeliveryView::RecipientInactive`] — no read/action and the
///    recipient's durable lifecycle is terminal (it cannot read it anymore);
/// 4. [`BoardDeliveryView::RecipientActive`] — no read/action and the
///    recipient is live;
/// 5. [`BoardDeliveryView::Stored`] — no recipient is tracked (root actor):
///    the post is durable, nothing else is claimed.
///
/// Activity NEVER implies [`BoardDeliveryView::Read`]: a live (or busy) child
/// that never read the post is `RecipientActive`, and another child's read
/// marker never marks this recipient read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoardDeliveryView {
    Stored,
    RecipientActive,
    RecipientInactive,
    Read,
    ActedUpon,
}

// ---------------------------------------------------------------- scoping

/// The resolved actor scope: the family root handle + the actor's own child
/// id (`None` = the root agent itself).
struct BoardScope {
    root: SessionHandle,
    actor_child: Option<ChildId>,
}

impl BoardScope {
    fn board_id(&self) -> BoardId {
        BoardId::of_root(self.root.id())
    }

    /// Refuse a board id that is not THIS actor's run-family board.
    fn check_board(&self, board_id: BoardId) -> faktor_core::Result<()> {
        if board_id.root() != self.root.id() {
            return Err(SessionError::Permission(format!(
                "access denied: board {} is not the run-family board of session {}",
                board_id,
                self.root.id()
            ))
            .into());
        }
        Ok(())
    }

    /// Resolve the caller-supplied child target under the actor scope:
    /// `None` means "the actor itself" (the root reads/writes untracked, a
    /// child tracks itself); a root may act for ANY descendant of the run
    /// family; a child may act ONLY for itself. Foreign-family children are
    /// a typed permission refusal even when the caller knows their ids.
    fn scoped_child(&self, requested: Option<ChildId>) -> faktor_core::Result<Option<ChildId>> {
        match (self.actor_child, requested) {
            (None, None) => Ok(None),
            (Some(actor), None) => Ok(Some(actor)),
            (Some(actor), Some(child)) => {
                if actor != child {
                    return Err(SessionError::Permission(format!(
                        "access denied: child {} cannot act for child {} (children may only act for themselves)",
                        actor, child
                    ))
                    .into());
                }
                Ok(Some(child))
            }
            (None, Some(child)) => {
                // The root may act for any family member, but only a REAL
                // member: resolve the child's own root and require it to be
                // this family's root.
                let Some(child_handle) = self.root.manager.get_session(child.session())? else {
                    return Err(SessionError::NotFound(format!(
                        "board child session {}",
                        child.session()
                    ))
                    .into());
                };
                let (child_root, _) = family_root(&child_handle)?;
                if child_root.id() != self.root.id() {
                    return Err(SessionError::Permission(format!(
                        "access denied: child {} belongs to a different run family",
                        child
                    ))
                    .into());
                }
                Ok(Some(child))
            }
        }
    }
}

/// Walk the durable child identity rows up to the family root (bounded
/// depth, cycle-detecting). The returned `Option<ChildId>` is `Some` when
/// `handle` itself is an orchestrated child.
fn family_root(handle: &SessionHandle) -> faktor_core::Result<(SessionHandle, Option<ChildId>)> {
    let origin = handle.id();
    let mut current = handle.clone();
    let mut is_child = false;
    let mut seen: BTreeSet<u64> = BTreeSet::new();
    for _ in 0..MAX_BOARD_FAMILY_DEPTH {
        if !seen.insert(current.id().raw()) {
            return Err(SessionError::Malformed(format!(
                "run family of session {origin} contains a parent cycle at session {}",
                current.id()
            ))
            .into());
        }
        let Some(identity) = current.orchestrator_child_identity_get()? else {
            let actor_child = is_child.then(|| ChildId::of_session(origin));
            return Ok((current, actor_child));
        };
        is_child = true;
        let parent = identity.parent_session_id;
        let Some(parent_handle) = current.manager.get_session(parent)? else {
            return Err(SessionError::NotFound(format!(
                "parent session {parent} of board family member {}",
                current.id()
            ))
            .into());
        };
        current = parent_handle;
    }
    Err(SessionError::Malformed(format!(
        "run family of session {origin} exceeds MAX_BOARD_FAMILY_DEPTH ({MAX_BOARD_FAMILY_DEPTH})"
    ))
    .into())
}

fn board_scope(handle: &SessionHandle) -> faktor_core::Result<BoardScope> {
    let (root, actor_child) = family_root(handle)?;
    Ok(BoardScope { root, actor_child })
}

/// The same lifecycle rule the steer path applies to a child: a terminal
/// session (lifecycle or per-turn state) or a durable child-runtime
/// projection in a terminal state refuses writes. The root is never
/// lifecycle-gated (the board belongs to the run family, not one turn).
fn actor_is_terminal(handle: &SessionHandle) -> faktor_core::Result<bool> {
    if handle.lifecycle()?.is_terminal() {
        return Ok(true);
    }
    if handle.state()?.is_terminal() {
        return Ok(true);
    }
    if let Some(row) = handle.orchestrator_child_runtime_get()? {
        if matches!(row.state.as_str(), "done" | "cancelled" | "failed") {
            return Ok(true);
        }
    }
    Ok(false)
}

fn refuse_terminal_child(
    actor_child: Option<ChildId>,
    handle: &SessionHandle,
) -> faktor_core::Result<()> {
    if actor_child.is_some() && actor_is_terminal(handle)? {
        return Err(SessionError::Permission(format!(
            "a terminal child cannot write to the coordination board (session {})",
            handle.id()
        ))
        .into());
    }
    Ok(())
}

// ---------------------------------------------------------------- append/read

fn maybe_child(raw: Option<ChildId>) -> Option<u64> {
    raw.map(|c| c.session().raw())
}

impl SessionHandle {
    /// The run-family board id of this session (root-derived).
    pub fn board_id(&self) -> faktor_core::Result<BoardId> {
        Ok(board_scope(self)?.board_id())
    }

    /// Post to this actor's run-family board. `author_child` is recorded
    /// from the durable family relation (a child posts as itself, the root
    /// as `None`). Terminal children are refused typed before any write.
    pub fn board_post(
        &self,
        subject: &str,
        body: &str,
        refs: &[String],
    ) -> faktor_core::Result<BoardPost> {
        let scope = board_scope(self)?;
        self.board_post_impl(&scope, scope.board_id(), subject, body, refs)
    }

    /// Post to an EXPLICIT board id. The target must be the actor's own
    /// run-family board: a child knowing another run's `BoardId` (or that
    /// run's ids) is refused with `access denied`.
    pub fn board_post_to(
        &self,
        board_id: BoardId,
        subject: &str,
        body: &str,
        refs: &[String],
    ) -> faktor_core::Result<BoardPost> {
        let scope = board_scope(self)?;
        scope.check_board(board_id)?;
        self.board_post_impl(&scope, board_id, subject, body, refs)
    }

    fn board_post_impl(
        &self,
        scope: &BoardScope,
        board_id: BoardId,
        subject: &str,
        body: &str,
        refs: &[String],
    ) -> faktor_core::Result<BoardPost> {
        refuse_terminal_child(scope.actor_child, self)?;
        // Validate the caller-controlled shape BEFORE any lock/write: a
        // hostile post leaves no trace. (The revision/id equality is
        // re-checked at the real append with the allocated revision.)
        validate_board_post(
            board_id.root().raw(),
            1,
            maybe_child(scope.actor_child),
            self.id().raw(),
            subject,
            body,
            refs,
            1,
        )?;
        let _guard = scope.root.command_guard();
        let head = scope.root.ledger_ensure_head()?;
        let revision = head
            .board_revision
            .checked_add(1)
            .ok_or_else(|| SessionError::Oversized("board revision overflowed u64".to_string()))?;
        validate_board_post(
            board_id.root().raw(),
            revision,
            maybe_child(scope.actor_child),
            self.id().raw(),
            subject,
            body,
            refs,
            revision,
        )?;
        let _seq = scope.root.append_typed_entry(LedgerPayload::BoardPost {
            board_id: board_id.root().raw(),
            post_id: revision,
            author_child: maybe_child(scope.actor_child),
            author_session: self.id().raw(),
            subject: subject.to_string(),
            body: body.to_string(),
            refs: refs.to_vec(),
            revision,
        })?;
        Ok(BoardPost {
            id: BoardPostId::new(revision),
            board_id,
            author_child: scope.actor_child,
            author_session: self.id(),
            subject: subject.to_string(),
            body: body.to_string(),
            refs: refs.to_vec(),
            revision,
            created_ms: self.now_ms(),
        })
    }

    /// Read one bounded newest-first page of the actor's run-family board.
    ///
    /// `child`: `None` = the actor itself (a child tracks itself, the root
    /// reads untracked); a root may address any descendant, a child only
    /// itself (typed `access denied` otherwise).
    ///
    /// `since_revision`: exclusive cursor for OLDER pages — pass the
    /// previous page's `next_before_revision`; `None` reads the newest
    /// page. Returned posts have `revision < since_revision` and are
    /// newest-first. Posts hidden by the newest reset marker are never
    /// returned.
    ///
    /// `exclude_self`: drop posts authored by the tracked child.
    ///
    /// Reads by the tracked child are durably recorded (at most once per
    /// post) BEFORE the page is returned, so unread counts never race the
    /// caller seeing the page.
    pub fn board_read_posts(
        &self,
        child: Option<ChildId>,
        since_revision: Option<u64>,
        limit: usize,
        exclude_self: bool,
    ) -> faktor_core::Result<BoardPage> {
        let scope = board_scope(self)?;
        self.board_read_posts_impl(
            &scope,
            scope.board_id(),
            child,
            since_revision,
            limit,
            exclude_self,
        )
    }

    /// Read an EXPLICIT board id under the same family scope check.
    pub fn board_read_posts_of(
        &self,
        board_id: BoardId,
        child: Option<ChildId>,
        since_revision: Option<u64>,
        limit: usize,
        exclude_self: bool,
    ) -> faktor_core::Result<BoardPage> {
        let scope = board_scope(self)?;
        scope.check_board(board_id)?;
        self.board_read_posts_impl(&scope, board_id, child, since_revision, limit, exclude_self)
    }

    fn board_read_posts_impl(
        &self,
        scope: &BoardScope,
        board_id: BoardId,
        child: Option<ChildId>,
        since_revision: Option<u64>,
        limit: usize,
        exclude_self: bool,
    ) -> faktor_core::Result<BoardPage> {
        let target = scope.scoped_child(child)?;
        let target_raw = maybe_child(target);
        let limit = limit.clamp(1, MAX_BOARD_PAGE);
        let scan = Self::scan_posts(
            &scope.root,
            board_id,
            target_raw,
            since_revision,
            limit,
            exclude_self,
        )?;

        // Record (idempotently) the reads of the returned posts for the
        // tracked child. A read row is only appended when the descending
        // scan did NOT already see one above the post.
        let newly_read: Vec<u64> = scan
            .posts
            .iter()
            .filter(|p| {
                target_raw.is_some()
                    && maybe_child(p.author_child) != target_raw
                    && !scan.seen_reads.contains(&p.id.raw())
            })
            .map(|p| p.id.raw())
            .collect();
        if let (Some(child), false) = (target_raw, newly_read.is_empty()) {
            let _guard = scope.root.command_guard();
            for post_id in newly_read {
                scope.root.append_typed_entry(LedgerPayload::BoardRead {
                    board_id: board_id.root().raw(),
                    child,
                    post_id,
                })?;
            }
        }

        let revision = scope.root.ledger_ensure_head()?.board_revision;
        let next_before_revision = if scan.has_more {
            scan.posts.last().map(|p| p.revision)
        } else {
            None
        };
        Ok(BoardPage {
            board_id,
            revision,
            posts: scan.posts,
            next_before_revision,
            has_more: scan.has_more,
        })
    }

    /// One bounded descending scan of the pinned board stream: collects the
    /// newest `limit` visible posts below `since_revision` (excluding
    /// self-authored posts when asked), the tracked child's read markers
    /// seen above them, and whether a further visible post exists. The scan
    /// reads the ROOT session's ledger — the board's durable home.
    #[allow(clippy::too_many_arguments)]
    fn scan_posts(
        root: &SessionHandle,
        board_id: BoardId,
        target_raw: Option<u64>,
        since_revision: Option<u64>,
        limit: usize,
        exclude_self: bool,
    ) -> faktor_core::Result<PostScan> {
        let mut scan = PostScan::default();
        let mut cursor: Option<i64> = None;
        let mut scanned: usize = 0;
        'outer: loop {
            let rows = root
                .manager
                .store()
                .ledger_entries_desc(root.id(), cursor, crate::ledger::MAX_BOARD_SCAN_PAGE)
                .map_err(map_store_err)?;
            if rows.is_empty() {
                break;
            }
            cursor = rows.last().map(|r| r.seq);
            for row in rows {
                scanned += 1;
                if scanned > MAX_BOARD_SCAN_ROWS {
                    return Err(SessionError::Oversized(format!(
                    "board scan of session {} exceeded MAX_BOARD_SCAN_ROWS; narrow the read with since_revision",
                    root.id()
                ))
                .into());
                }
                let entry = root.decode_row(&row)?;
                match entry.payload {
                    LedgerPayload::BoardReset {
                        board_id: b,
                        new_revision,
                        ..
                    } => {
                        check_board_row_id(b, board_id)?;
                        scan.reset_revision = scan.reset_revision.max(new_revision);
                    }
                    LedgerPayload::BoardRead {
                        board_id: b,
                        child,
                        post_id,
                    } => {
                        check_board_row_id(b, board_id)?;
                        if Some(child) == target_raw {
                            scan.seen_reads.insert(post_id);
                        }
                    }
                    LedgerPayload::BoardReceipt { board_id: b, .. } => {
                        check_board_row_id(b, board_id)?;
                    }
                    LedgerPayload::BoardPost {
                        board_id: b,
                        post_id,
                        author_child,
                        author_session,
                        subject,
                        body,
                        refs,
                        revision,
                    } => {
                        check_board_row_id(b, board_id)?;
                        if revision <= scan.reset_revision {
                            continue;
                        }
                        if let Some(since) = since_revision {
                            if revision >= since {
                                continue;
                            }
                        }
                        if exclude_self && target_raw.is_some() && author_child == target_raw {
                            continue;
                        }
                        if scan.posts.len() >= limit {
                            scan.has_more = true;
                            break 'outer;
                        }
                        let author = match author_child {
                            Some(raw) => Some(ChildId::of_session(unchecked_session(raw)?)),
                            None => None,
                        };
                        scan.posts.push(BoardPost {
                            id: BoardPostId::new(post_id),
                            board_id,
                            author_child: author,
                            author_session: unchecked_session(author_session)?,
                            subject,
                            body,
                            refs,
                            revision,
                            created_ms: entry.created_ms,
                        });
                    }
                    _ => {}
                }
            }
        }
        Ok(scan)
    }

    /// The explicit action receipt on one post. The post must be on the
    /// LIVE surface (visible above the newest reset marker); terminal
    /// children are refused typed before any write.
    pub fn board_receipt(
        &self,
        post_id: BoardPostId,
        action: BoardAction,
        note: &str,
    ) -> faktor_core::Result<BoardReceipt> {
        let scope = board_scope(self)?;
        refuse_terminal_child(scope.actor_child, self)?;
        let board_id = scope.board_id();
        validate_board_receipt(
            board_id.root().raw(),
            maybe_child(scope.actor_child),
            post_id.raw(),
            action.as_str(),
            note,
        )?;
        Self::live_post_seq(&scope.root, board_id, post_id)?;
        let created_ms = self.now_ms();
        scope.root.append_typed_entry(LedgerPayload::BoardReceipt {
            board_id: board_id.root().raw(),
            child: maybe_child(scope.actor_child),
            post_id: post_id.raw(),
            action: action.as_str().to_string(),
            note: note.to_string(),
        })?;
        Ok(BoardReceipt {
            child: scope.actor_child,
            post_id,
            action,
            note: note.to_string(),
            created_ms,
        })
    }

    /// The read marker of `child` plus every action receipt recorded on
    /// `post_id` (REGARDLESS of author: a child asking "did anyone answer my
    /// question?" must see the root's ack), oldest action first. Reading
    /// receipts is itself a data read: a child may ask only for itself
    /// unless it is the root.
    pub fn board_receipts(
        &self,
        child: ChildId,
        post_id: BoardPostId,
    ) -> faktor_core::Result<BoardReceipts> {
        let scope = board_scope(self)?;
        let board_id = scope.board_id();
        let Some(target) = scope.scoped_child(Some(child))? else {
            return Err(
                SessionError::Internal("scoped board child resolved to None".to_string()).into(),
            );
        };
        let target_raw = target.session().raw();
        let root = &scope.root;

        let mut out = BoardReceipts::default();
        let mut cursor: Option<i64> = None;
        let mut scanned: usize = 0;
        let mut seen_read = BTreeSet::new();
        let mut reset_revision = 0u64;
        loop {
            let rows = root
                .manager
                .store()
                .ledger_entries_desc(root.id(), cursor, crate::ledger::MAX_BOARD_SCAN_PAGE)
                .map_err(map_store_err)?;
            if rows.is_empty() {
                return Err(SessionError::NotFound(format!(
                    "board post {} on the live surface of {}",
                    post_id, board_id
                ))
                .into());
            }
            cursor = rows.last().map(|r| r.seq);
            for row in rows {
                scanned += 1;
                if scanned > MAX_BOARD_SCAN_ROWS {
                    return Err(SessionError::Oversized(format!(
                        "board receipt scan of session {} exceeded MAX_BOARD_SCAN_ROWS",
                        root.id()
                    ))
                    .into());
                }
                let entry = root.decode_row(&row)?;
                match entry.payload {
                    LedgerPayload::BoardReset {
                        board_id: b,
                        new_revision,
                        ..
                    } => {
                        check_board_row_id(b, board_id)?;
                        reset_revision = reset_revision.max(new_revision);
                    }
                    LedgerPayload::BoardPost {
                        board_id: b,
                        post_id: p,
                        revision,
                        ..
                    } => {
                        check_board_row_id(b, board_id)?;
                        if p == post_id.raw() {
                            if revision <= reset_revision {
                                // Hidden by a reset marker: keep scanning to
                                // the end so the caller gets a typed NotFound
                                // instead of pre-reset receipts.
                                continue;
                            }
                            if out.actions.len() > MAX_BOARD_RECEIPTS_PER_POST {
                                return Err(SessionError::Oversized(format!(
                                    "board post {post_id} carries more than MAX_BOARD_RECEIPTS_PER_POST receipts"
                                ))
                                .into());
                            }
                            out.actions.reverse(); // descending scan -> oldest first
                            return Ok(out);
                        }
                    }
                    LedgerPayload::BoardRead {
                        board_id: b,
                        child: c,
                        post_id: p,
                    } => {
                        check_board_row_id(b, board_id)?;
                        if c == target_raw && p == post_id.raw() && seen_read.insert(p) {
                            out.read = Some(BoardRead {
                                child: target,
                                post_id,
                                read_ms: entry.created_ms,
                            });
                        }
                    }
                    LedgerPayload::BoardReceipt {
                        board_id: b,
                        child: c,
                        post_id: p,
                        action,
                        note,
                    } => {
                        check_board_row_id(b, board_id)?;
                        if p == post_id.raw() {
                            let Some(action) = BoardAction::parse(&action) else {
                                return Err(SessionError::Malformed(format!(
                                    "ledger board receipt action {action:?} is not ack|task_update|blocked|question"
                                ))
                                .into());
                            };
                            let author = match c {
                                Some(raw) => Some(ChildId::of_session(unchecked_session(raw)?)),
                                None => None,
                            };
                            out.actions.push(BoardReceipt {
                                child: author,
                                post_id,
                                action,
                                note,
                                created_ms: entry.created_ms,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// The derived [`BoardDeliveryView`] of one LIVE post for one
    /// recipient. Purely additive read: it reuses the existing post/read/
    /// receipt rows (no new ledger stream) and the recipient's durable child
    /// lifecycle. `child: None` tracks the root actor (no recipient →
    /// `Stored`). A post hidden by a reset marker is a typed `NotFound`.
    pub fn board_delivery_view(
        &self,
        child: Option<ChildId>,
        post_id: BoardPostId,
    ) -> faktor_core::Result<BoardDeliveryView> {
        let scope = board_scope(self)?;
        let board_id = scope.board_id();
        let target = scope.scoped_child(child)?;
        // The post must be on the live surface (hidden history never gets a
        // delivery view).
        Self::live_post_seq(&scope.root, board_id, post_id)?;
        let Some(target) = target else {
            return Ok(BoardDeliveryView::Stored);
        };
        let target_raw = target.session().raw();
        let root = &scope.root;
        let mut read = false;
        let mut acted = false;
        let mut cursor: Option<i64> = None;
        let mut scanned = 0usize;
        'outer: loop {
            let rows = root
                .manager
                .store()
                .ledger_entries_desc(root.id(), cursor, crate::ledger::MAX_BOARD_SCAN_PAGE)
                .map_err(map_store_err)?;
            if rows.is_empty() {
                break;
            }
            cursor = rows.last().map(|r| r.seq);
            for row in rows {
                scanned += 1;
                if scanned > MAX_BOARD_SCAN_ROWS {
                    return Err(SessionError::Oversized(format!(
                        "board delivery scan of session {} exceeded MAX_BOARD_SCAN_ROWS",
                        root.id()
                    ))
                    .into());
                }
                let entry = root.decode_row(&row)?;
                match entry.payload {
                    LedgerPayload::BoardRead {
                        board_id: b,
                        child: c,
                        post_id: p,
                    } => {
                        check_board_row_id(b, board_id)?;
                        if c == target_raw && p == post_id.raw() {
                            read = true;
                        }
                    }
                    LedgerPayload::BoardReceipt {
                        board_id: b,
                        child: c,
                        post_id: p,
                        ..
                    } => {
                        check_board_row_id(b, board_id)?;
                        if c == Some(target_raw) && p == post_id.raw() {
                            acted = true;
                        }
                    }
                    LedgerPayload::BoardPost {
                        board_id: b,
                        post_id: p,
                        ..
                    } => {
                        check_board_row_id(b, board_id)?;
                        // Newest-first: once the post itself is reached, every
                        // older row cannot belong to it.
                        if p == post_id.raw() {
                            break 'outer;
                        }
                    }
                    LedgerPayload::BoardReset { board_id: b, .. } => {
                        check_board_row_id(b, board_id)?;
                    }
                    _ => {}
                }
            }
        }
        // Strongest evidence first: an action is never downgraded by a read.
        if acted {
            return Ok(BoardDeliveryView::ActedUpon);
        }
        if read {
            return Ok(BoardDeliveryView::Read);
        }
        // No read/action evidence: the recipient's DURABLE lifecycle decides
        // whether delivery is still possible. Activity never infers Read.
        let handle = root.manager.get_session(target.session())?.ok_or_else(|| {
            SessionError::NotFound(format!("board recipient session {}", target.session()))
        })?;
        if actor_is_terminal(&handle)? {
            Ok(BoardDeliveryView::RecipientInactive)
        } else {
            Ok(BoardDeliveryView::RecipientActive)
        }
    }

    /// Unread counts of the LIVE surface for one child (a root actor may
    /// ask for any descendant, a child only for itself): every visible post
    /// NOT authored by the child maps to `1` when unread and `0` when the
    /// child's durable read marker exists. Hidden (pre-reset) posts never
    /// appear.
    pub fn board_unread_counts(
        &self,
        child: ChildId,
    ) -> faktor_core::Result<BTreeMap<BoardPostId, u32>> {
        let scope = board_scope(self)?;
        let board_id = scope.board_id();
        let Some(target) = scope.scoped_child(Some(child))? else {
            return Err(
                SessionError::Internal("scoped board child resolved to None".to_string()).into(),
            );
        };
        let target_raw = target.session().raw();
        let root = &scope.root;

        let mut out: BTreeMap<BoardPostId, u32> = BTreeMap::new();
        let mut seen_reads: BTreeSet<u64> = BTreeSet::new();
        let mut reset_revision = 0u64;
        let mut cursor: Option<i64> = None;
        let mut scanned = 0usize;
        loop {
            let rows = root
                .manager
                .store()
                .ledger_entries_desc(root.id(), cursor, crate::ledger::MAX_BOARD_SCAN_PAGE)
                .map_err(map_store_err)?;
            if rows.is_empty() {
                break;
            }
            cursor = rows.last().map(|r| r.seq);
            for row in rows {
                scanned += 1;
                if scanned > MAX_BOARD_SCAN_ROWS {
                    return Err(SessionError::Oversized(format!(
                        "board unread scan of session {} exceeded MAX_BOARD_SCAN_ROWS",
                        root.id()
                    ))
                    .into());
                }
                let entry = root.decode_row(&row)?;
                match entry.payload {
                    LedgerPayload::BoardReset {
                        board_id: b,
                        new_revision,
                        ..
                    } => {
                        check_board_row_id(b, board_id)?;
                        reset_revision = reset_revision.max(new_revision);
                    }
                    LedgerPayload::BoardRead {
                        board_id: b,
                        child,
                        post_id,
                    } => {
                        check_board_row_id(b, board_id)?;
                        if child == target_raw {
                            seen_reads.insert(post_id);
                        }
                    }
                    LedgerPayload::BoardPost {
                        board_id: b,
                        post_id,
                        author_child,
                        revision,
                        ..
                    } => {
                        check_board_row_id(b, board_id)?;
                        if revision <= reset_revision || author_child == Some(target_raw) {
                            continue;
                        }
                        let unread = u32::from(!seen_reads.contains(&post_id));
                        out.insert(BoardPostId::new(post_id), unread);
                    }
                    _ => {}
                }
            }
        }
        Ok(out)
    }

    /// CAS reset of the actor's run-family board (ROOT actor only). On a
    /// stale `expected_revision` this is a typed `Conflict` and NOTHING is
    /// written; otherwise ONE durable reset marker bumps the revision by
    /// exactly one and hides every post at or below it from the live
    /// surface. History is never deleted (append-only, compaction-pinned).
    pub fn board_reset(&self, expected_revision: u64) -> faktor_core::Result<BoardReset> {
        let scope = board_scope(self)?;
        if let Some(actor) = scope.actor_child {
            return Err(SessionError::Permission(format!(
                "access denied: child {actor} cannot reset the run-family board (root only)"
            ))
            .into());
        }
        let board_id = scope.board_id();
        let _guard = scope.root.command_guard();
        let head = scope.root.ledger_ensure_head()?;
        if head.board_revision != expected_revision {
            return Err(SessionError::Conflict(format!(
                "board {} reset expected revision {expected_revision} but the live revision is {}",
                board_id, head.board_revision
            ))
            .into());
        }
        let new_revision = expected_revision
            .checked_add(1)
            .ok_or_else(|| SessionError::Oversized("board revision overflowed u64".to_string()))?;
        validate_board_reset(board_id.root().raw(), expected_revision, new_revision)?;
        scope.root.append_typed_entry(LedgerPayload::BoardReset {
            board_id: board_id.root().raw(),
            previous_revision: expected_revision,
            new_revision,
        })?;
        Ok(BoardReset {
            board_id,
            previous_revision: expected_revision,
            new_revision,
        })
    }

    /// The ledger seq of `post_id` when it is on the LIVE surface (above
    /// the newest reset marker); `NotFound` otherwise. One bounded
    /// descending walk of the ROOT ledger that stops the moment the post
    /// row is reached.
    fn live_post_seq(
        root: &SessionHandle,
        board_id: BoardId,
        post_id: BoardPostId,
    ) -> faktor_core::Result<i64> {
        let mut cursor: Option<i64> = None;
        let mut scanned = 0usize;
        let mut reset_revision = 0u64;
        loop {
            let rows = root
                .manager
                .store()
                .ledger_entries_desc(root.id(), cursor, crate::ledger::MAX_BOARD_SCAN_PAGE)
                .map_err(map_store_err)?;
            if rows.is_empty() {
                break;
            }
            cursor = rows.last().map(|r| r.seq);
            for row in rows {
                scanned += 1;
                if scanned > MAX_BOARD_SCAN_ROWS {
                    return Err(SessionError::Oversized(format!(
                        "board post lookup of session {} exceeded MAX_BOARD_SCAN_ROWS",
                        root.id()
                    ))
                    .into());
                }
                let entry = root.decode_row(&row)?;
                match entry.payload {
                    LedgerPayload::BoardReset {
                        board_id: b,
                        new_revision,
                        ..
                    } => {
                        check_board_row_id(b, board_id)?;
                        reset_revision = reset_revision.max(new_revision);
                    }
                    LedgerPayload::BoardPost {
                        board_id: b,
                        post_id: p,
                        revision,
                        ..
                    } => {
                        check_board_row_id(b, board_id)?;
                        if p == post_id.raw() {
                            if revision <= reset_revision {
                                break;
                            }
                            return Ok(entry.seq);
                        }
                    }
                    LedgerPayload::BoardRead { board_id: b, .. }
                    | LedgerPayload::BoardReceipt { board_id: b, .. } => {
                        check_board_row_id(b, board_id)?;
                    }
                    _ => {}
                }
            }
        }
        Err(SessionError::NotFound(format!(
            "board post {post_id} on the live surface of {board_id}"
        ))
        .into())
    }
}

#[derive(Default)]
struct PostScan {
    posts: Vec<BoardPost>,
    seen_reads: BTreeSet<u64>,
    reset_revision: u64,
    has_more: bool,
}

fn check_board_row_id(raw: u64, board_id: BoardId) -> faktor_core::Result<()> {
    if raw != board_id.root().raw() {
        return Err(SessionError::Malformed(format!(
            "ledger holds board rows of foreign board {raw} inside the ledger of {}",
            board_id.root()
        ))
        .into());
    }
    Ok(())
}

/// Rebuild a `SessionId` from a payload that passed the board validators
/// (non-zero). A zero here means a decoder bypass: loud, never a panic.
fn unchecked_session(raw: u64) -> faktor_core::Result<SessionId> {
    SessionId::try_from(raw).map_err(|_| {
        SessionError::Malformed(format!("ledger board payload carries session id {raw}")).into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::child::{ChildOwnership, ChildRuntimeBlockerRow};
    use crate::SessionManager;
    use faktor_core::error::ErrorKind;
    use faktor_core::id::{TaskId, WorktreeId};
    use std::sync::Arc;
    use tempfile::tempdir;

    fn manager() -> (tempfile::TempDir, Arc<SessionManager>) {
        let dir = tempdir().unwrap();
        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        (dir, m)
    }

    fn root_session(m: &Arc<SessionManager>) -> SessionHandle {
        let ws = m.create_workspace("/root").unwrap();
        m.create_session(ws, "root", "fake", "m").unwrap()
    }

    fn child_session(
        m: &Arc<SessionManager>,
        parent: &SessionHandle,
        worktree: u64,
    ) -> SessionHandle {
        let ws = parent.row().unwrap().workspace_id;
        m.create_child_session(
            parent.id(),
            ws,
            WorktreeId::new(worktree),
            TaskId::new(1),
            "fake",
            "m",
            "child",
            ChildOwnership::ReadOnlyShared,
        )
        .unwrap()
    }

    fn child_id(handle: &SessionHandle) -> ChildId {
        ChildId::of_session(handle.id())
    }

    fn all_visible(s: &SessionHandle) -> Vec<BoardPost> {
        let mut out = Vec::new();
        let mut since = None;
        loop {
            let page = s.board_read_posts(None, since, 64, false).unwrap();
            let has_more = page.has_more;
            since = page.next_before_revision;
            out.extend(page.posts);
            if !has_more {
                return out;
            }
        }
    }

    #[test]
    fn post_read_paging_and_receipts_roundtrip() {
        let (_d, m) = manager();
        let rt = root_session(&m);
        let c1 = child_session(&m, &rt, 2);

        let p1 = rt
            .board_post("status", "root ready", &["evidence/x".into()])
            .unwrap();
        assert_eq!(p1.board_id, rt.board_id().unwrap());
        assert_eq!(p1.revision, 1);
        assert_eq!(p1.id.raw(), 1, "post id equals its revision");
        assert!(p1.author_child.is_none());
        assert_eq!(p1.author_session, rt.id());
        assert_eq!(p1.refs, vec!["evidence/x".to_string()]);

        let p2 = c1.board_post("help", "need input", &[]).unwrap();
        assert_eq!(p2.author_child, Some(child_id(&c1)));
        assert_eq!(p2.author_session, c1.id());
        assert_eq!(p2.revision, 2);

        let page = rt.board_read_posts(None, None, 10, false).unwrap();
        assert_eq!(page.posts.len(), 2);
        assert_eq!(page.posts[0].id, p2.id);
        assert_eq!(page.posts[1].id, p1.id);
        assert_eq!(page.revision, 2);
        assert!(!page.has_more);
        assert_eq!(page.next_before_revision, None);

        // Unread counts for c1 BEFORE any child read: p1 is unread; p2 is
        // self-authored and never counted.
        let counts = c1.board_unread_counts(child_id(&c1)).unwrap();
        assert_eq!(counts.get(&p1.id), Some(&1), "c1 has not read p1 yet");
        assert_eq!(
            counts.get(&p2.id),
            None,
            "self-authored posts are not counted"
        );

        // Self-exclusion drops the caller's own post from the PAGE, but the
        // returned post is still durably read.
        let excluded = c1.board_read_posts(None, None, 10, true).unwrap();
        assert_eq!(excluded.posts.len(), 1);
        assert_eq!(excluded.posts[0].id, p1.id);
        let counts = c1.board_unread_counts(child_id(&c1)).unwrap();
        assert_eq!(counts.get(&p1.id), Some(&0));

        // A bounded child read records the remaining read markers durably.
        let page = c1.board_read_posts(None, None, 64, false).unwrap();
        assert_eq!(page.posts.len(), 2);
        let counts = c1.board_unread_counts(child_id(&c1)).unwrap();
        assert_eq!(counts.get(&p1.id), Some(&0));
        assert_eq!(counts.get(&p2.id), None);

        // New root post: unread until read; the next page is newest-first
        // and carries the exclusive older-page cursor.
        let p3 = rt.board_post("ping", "review please", &[]).unwrap();
        assert_eq!(p3.revision, 3);
        let counts = c1.board_unread_counts(child_id(&c1)).unwrap();
        assert_eq!(counts.get(&p3.id), Some(&1));
        let first = c1.board_read_posts(None, None, 1, false).unwrap();
        assert_eq!(first.posts.len(), 1);
        assert_eq!(first.posts[0].id, p3.id);
        assert!(first.has_more);
        assert_eq!(first.next_before_revision, Some(3));
        assert_eq!(first.revision, 3);
        let older = c1
            .board_read_posts(None, first.next_before_revision, 64, false)
            .unwrap();
        assert_eq!(older.posts.len(), 2);
        assert_eq!(older.posts[0].id, p2.id);
        assert_eq!(older.posts[1].id, p1.id);
        assert!(!older.has_more);
        let counts = c1.board_unread_counts(child_id(&c1)).unwrap();
        assert_eq!(counts.get(&p3.id), Some(&0));

        // Action receipts roundtrip; the read marker is visible beside them
        // and EVERY author's actions are returned (a child must see the
        // root's answer to its question).
        c1.board_receipt(p3.id, BoardAction::Ack, "on it").unwrap();
        rt.board_receipt(p3.id, BoardAction::Question, "which check?")
            .unwrap();
        let receipts = c1.board_receipts(child_id(&c1), p3.id).unwrap();
        assert!(receipts.read.is_some());
        assert_eq!(receipts.actions.len(), 2);
        assert_eq!(receipts.actions[0].action, BoardAction::Ack);
        assert_eq!(receipts.actions[0].child, Some(child_id(&c1)));
        assert_eq!(receipts.actions[1].action, BoardAction::Question);
        assert_eq!(
            receipts.actions[1].child, None,
            "root action has no child id"
        );
        let root_receipts = rt.board_receipts(child_id(&c1), p3.id).unwrap();
        assert_eq!(root_receipts.actions.len(), 2);

        // A receipt on an unknown post is a typed NotFound with no write.
        let err = rt
            .board_receipt(BoardPostId::new(9_999), BoardAction::Ack, "")
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::NotFound);
    }

    #[test]
    fn cross_run_access_denied_even_knowing_ids() {
        let (_d, m) = manager();
        let a = root_session(&m);
        let a1 = child_session(&m, &a, 2);
        let a2 = child_session(&m, &a, 3);
        let b = root_session(&m);
        let b1 = child_session(&m, &b, 2);

        let pa = a.board_post("alpha", "run A", &[]).unwrap();
        let pb = b1.board_post("beta", "run B", &[]).unwrap();
        assert_ne!(pa.board_id, pb.board_id);
        assert_ne!(a.board_id().unwrap(), b.board_id().unwrap());

        // b1 cannot read A's board, even passing A's real board/child ids.
        for err in [
            b1.board_read_posts(Some(child_id(&a1)), None, 10, false)
                .unwrap_err(),
            b1.board_read_posts_of(a.board_id().unwrap(), None, None, 10, false)
                .unwrap_err(),
            b1.board_post_to(a.board_id().unwrap(), "x", "y", &[])
                .unwrap_err(),
            b1.board_receipts(child_id(&a1), pa.id).unwrap_err(),
            b1.board_unread_counts(child_id(&a1)).unwrap_err(),
        ] {
            assert_eq!(err.kind, ErrorKind::Permission, "{}", err.message);
            assert!(err.message.contains("access denied"), "{}", err.message);
        }

        // A child may act only for itself: a1 cannot address sibling a2.
        let err = a1
            .board_read_posts(Some(child_id(&a2)), None, 10, false)
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
        // The root may address any descendant.
        let as_a2 = a
            .board_read_posts(Some(child_id(&a2)), None, 10, false)
            .unwrap();
        assert_eq!(as_a2.posts.len(), 1);
        assert_eq!(as_a2.posts[0].id, pa.id);

        // Run A's board never contains run B's post and vice versa.
        let a_posts = all_visible(&a);
        assert_eq!(a_posts.len(), 1);
        assert!(a_posts.iter().all(|p| p.id == pa.id));
        let b_posts = all_visible(&b1);
        assert_eq!(b_posts.len(), 1);
        assert!(b_posts.iter().all(|p| p.id == pb.id));

        // Family member reads only see the family board.
        let a1_posts = all_visible(&a1);
        assert_eq!(a1_posts.len(), 1);
        assert_eq!(a1_posts[0].id, pa.id);
    }

    #[test]
    fn reset_cas_conflict_hides_surface_and_retains_history() {
        let (_d, m) = manager();
        let rt = root_session(&m);
        let c1 = child_session(&m, &rt, 2);

        let p1 = rt.board_post("one", "first", &[]).unwrap();
        let p2 = c1.board_post("two", "second", &[]).unwrap();
        c1.board_read_posts(None, None, 10, false).unwrap();

        let reset = rt.board_reset(2).unwrap();
        assert_eq!(reset.previous_revision, 2);
        assert_eq!(reset.new_revision, 3);

        // A stale concurrent reset is a typed Conflict and writes nothing.
        let err = rt.board_reset(2).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);
        let err = rt.board_reset(0).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Conflict);

        // The live surface is empty after the marker; unread counts too.
        let page = rt.board_read_posts(None, None, 10, false).unwrap();
        assert!(page.posts.is_empty());
        assert_eq!(page.revision, 3);
        assert!(!page.has_more);
        assert!(c1.board_unread_counts(child_id(&c1)).unwrap().is_empty());
        assert!(rt.board_receipts(child_id(&c1), p2.id).is_err());
        assert_eq!(
            rt.board_receipts(child_id(&c1), p2.id).unwrap_err().kind,
            ErrorKind::NotFound
        );

        // Posts after the marker are visible; ids never collide with hidden
        // history (a reset consumes a revision).
        let p3 = rt.board_post("three", "after reset", &[]).unwrap();
        assert_eq!(p3.revision, 4);
        let page = rt.board_read_posts(None, None, 10, false).unwrap();
        assert_eq!(page.posts.len(), 1);
        assert_eq!(page.posts[0].id, p3.id);

        // History is retained in the pinned stream (append-only).
        let entries = rt.ledger_entries_page(None, 500).unwrap().entries;
        assert!(entries.iter().any(|e| matches!(
            &e.payload,
            LedgerPayload::BoardPost { post_id, .. } if *post_id == p1.id.raw()
        )));
        assert!(entries
            .iter()
            .any(|e| matches!(&e.payload, LedgerPayload::BoardReset { .. })));

        // The reset is root-only.
        let err = c1.board_reset(4).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
    }

    #[test]
    fn unread_counts_are_per_child_and_reads_are_idempotent() {
        let (_d, m) = manager();
        let rt = root_session(&m);
        let c1 = child_session(&m, &rt, 2);
        let c2 = child_session(&m, &rt, 3);

        let root_post = rt.board_post("r", "root", &[]).unwrap();
        let p1 = c1.board_post("c1", "one", &[]).unwrap();
        let p2 = c2.board_post("c2", "two", &[]).unwrap();

        // c1 reads everything: only root_post and p2 count, both become 0.
        let seen = c1.board_read_posts(None, None, 10, false).unwrap();
        assert_eq!(seen.posts.len(), 3);
        c1.board_read_posts(None, None, 10, false).unwrap(); // repeat: idempotent
        let counts = c1.board_unread_counts(child_id(&c1)).unwrap();
        assert_eq!(counts.get(&root_post.id), Some(&0));
        assert_eq!(counts.get(&p2.id), Some(&0));
        assert_eq!(counts.get(&p1.id), None);

        // c2 authored p2 and read nothing yet.
        let counts2 = c2.board_unread_counts(child_id(&c2)).unwrap();
        assert_eq!(counts2.get(&root_post.id), Some(&1));
        assert_eq!(counts2.get(&p1.id), Some(&1));
        assert_eq!(counts2.get(&p2.id), None);

        // Exactly one read marker per (child, post) even after repeats.
        let entries = rt.ledger_entries_page(None, 500).unwrap().entries;
        let read_rows = entries
            .iter()
            .filter(|e| {
                matches!(
                    &e.payload,
                    LedgerPayload::BoardRead { child, post_id, .. }
                        if *child == c1.id().raw() && *post_id == root_post.id.raw()
                )
            })
            .count();
        assert_eq!(
            read_rows, 1,
            "repeat reads must not append duplicate markers"
        );
    }

    #[test]
    fn oversized_and_hostile_posts_are_typed_and_write_nothing() {
        let (_d, m) = manager();
        let rt = root_session(&m);

        let huge = "x".repeat(MAX_BOARD_BODY_BYTES + 1);
        let err = rt.board_post("s", &huge, &[]).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);

        let too_many_refs: Vec<String> = (0..=MAX_BOARD_REFS).map(|n| format!("ref-{n}")).collect();
        let err = rt.board_post("s", "ok", &too_many_refs).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);

        let oversized_ref = vec!["r".repeat(MAX_BOARD_REF_BYTES + 1)];
        let err = rt.board_post("s", "ok", &oversized_ref).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Oversized);

        let err = rt.board_post("bad\u{7}subject", "ok", &[]).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
        let err = rt.board_post("", "ok", &[]).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);
        let err = rt.board_post("s", "", &[]).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed);

        // Nothing hostile reached the ledger.
        assert!(all_visible(&rt).is_empty());

        // A hostile RAW row (id != revision) fails decode loudly on read.
        let sid = rt.id().raw() as i64;
        let payload = serde_json::json!({
            "kind": "board_post",
            "board_id": rt.id().raw(),
            "post_id": 7,
            "author_child": null,
            "author_session": rt.id().raw(),
            "subject": "s",
            "body": "b",
            "refs": [],
            "revision": 8
        });
        m.store()
            .sql_execute(&format!(
                "INSERT INTO ledger_entry(session_id, seq, entry_type, schema_ver, payload, created_ms)
                 SELECT {sid}, COALESCE(MAX(seq),0)+1, 'board_post', 1, '{}', 1
                 FROM ledger_entry WHERE session_id = {sid};",
                payload.to_string().replace('\'', "''")
            ))
            .unwrap();
        let err = rt.board_read_posts(None, None, 10, false).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Malformed, "{}", err.message);
    }

    #[test]
    fn terminal_child_cannot_post_or_act() {
        let (_d, m) = manager();
        let rt = root_session(&m);
        let c1 = child_session(&m, &rt, 2);

        let before = rt.board_post("root", "stay", &[]).unwrap();
        let c1_post = c1.board_post("early", "still live", &[]).unwrap();

        // Durable runtime projection says the child is terminal.
        c1.orchestrator_child_runtime_put(&ChildRuntimeBlockerRow {
            child_id: "child-0".into(),
            state: "cancelled".into(),
            blocker: None,
            updated_ms: 1,
        })
        .unwrap();
        let err = c1.board_post("late", "nope", &[]).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);
        assert!(err.message.contains("terminal"), "{}", err.message);
        let err = c1
            .board_receipt(before.id, BoardAction::Ack, "")
            .unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);

        // The lifecycle path refuses too (end_session closes the child).
        let c2 = child_session(&m, &rt, 3);
        c2.end_session().unwrap();
        let err = c2.board_post("late", "nope", &[]).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Permission);

        // No terminal write reached the board.
        let posts = all_visible(&rt);
        assert_eq!(posts.len(), 2);
        assert!(posts.iter().any(|p| p.id == before.id));
        assert!(posts.iter().any(|p| p.id == c1_post.id));
    }

    #[test]
    fn reopen_durability_and_compaction_pin_board_rows() {
        let dir = tempdir().unwrap();
        let (sid, post_id, cid) = {
            let m = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
            let rt = root_session(&m);
            let c1 = child_session(&m, &rt, 2);
            rt.ledger_goal_set("goal").unwrap();
            let post = rt
                .board_post("subject", "durable body", &["a.txt".into()])
                .unwrap();
            c1.board_read_posts(None, None, 10, false).unwrap();
            rt.board_receipt(post.id, BoardAction::TaskUpdate, "in progress")
                .unwrap();
            // Compaction must pin every board row (posts, reads, receipts).
            let report = rt.compact_typed_ledger().unwrap();
            assert!(report.pinned.len() >= 4, "{:?}", report.pinned);
            (rt.id(), post.id, c1.id())
        };

        let m =
            SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true).unwrap();
        let rt = m.get_session(sid).unwrap().unwrap();
        let page = rt.board_read_posts(None, None, 10, false).unwrap();
        assert_eq!(page.posts.len(), 1);
        assert_eq!(page.posts[0].id, post_id);
        assert_eq!(page.posts[0].body, "durable body");
        assert_eq!(page.revision, post_id.raw());
        let receipts = rt
            .board_receipts(ChildId::of_session(cid), post_id)
            .unwrap();
        assert!(
            receipts.read.is_some(),
            "read marker survives reopen+compaction"
        );
        assert_eq!(receipts.actions.len(), 1);
        assert_eq!(receipts.actions[0].action, BoardAction::TaskUpdate);
        let counts = rt.board_unread_counts(ChildId::of_session(cid)).unwrap();
        assert_eq!(counts.get(&post_id), Some(&0));
        // A post after reopen continues the revision sequence.
        let next = rt.board_post("next", "after reopen", &[]).unwrap();
        assert_eq!(next.revision, post_id.raw() + 1);
    }

    #[test]
    fn concurrent_resets_race_to_exactly_one_winner() {
        let (_d, m) = manager();
        let rt = root_session(&m);
        let p = rt.board_post("s", "b", &[]).unwrap();
        assert_eq!(p.revision, 1);

        let mut handles = Vec::new();
        for _ in 0..8 {
            let rt = rt.clone();
            handles.push(std::thread::spawn(move || {
                rt.board_reset(1)
                    .map(|r| r.new_revision)
                    .map_err(|e| e.kind)
            }));
        }
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(
            results.iter().filter(|r| r.is_ok()).count(),
            1,
            "exactly one reset may win the CAS race: {results:?}"
        );
        assert_eq!(results.iter().find(|r| r.is_ok()), Some(&Ok(2)));
        for r in &results {
            if let Err(kind) = r {
                assert_eq!(*kind, ErrorKind::Conflict, "{results:?}");
            }
        }
        assert!(rt
            .board_read_posts(None, None, 10, false)
            .unwrap()
            .posts
            .is_empty());
        // No two reset markers were written.
        let markers = rt
            .ledger_entries_page(None, 500)
            .unwrap()
            .entries
            .iter()
            .filter(|e| matches!(e.payload, LedgerPayload::BoardReset { .. }))
            .count();
        assert_eq!(markers, 1);
    }

    #[test]
    fn concurrent_posts_allocate_unique_revisions() {
        let (_d, m) = manager();
        let rt = root_session(&m);
        let mut handles = Vec::new();
        for n in 0..8 {
            let rt = rt.clone();
            handles.push(std::thread::spawn(move || {
                rt.board_post("s", &format!("body-{n}"), &[])
                    .unwrap()
                    .revision
            }));
        }
        let mut revisions: Vec<u64> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        revisions.sort_unstable();
        assert_eq!(revisions, (1..=8).collect::<Vec<u64>>());
        let posts = all_visible(&rt);
        assert_eq!(posts.len(), 8);
        let mut ids: Vec<u64> = posts.iter().map(|p| p.id.raw()).collect();
        ids.sort_unstable();
        assert_eq!(ids, (1..=8).collect::<Vec<u64>>());
    }

    #[test]
    fn ten_thousand_posts_page_with_bounded_memory() {
        let (_d, m) = manager();
        let rt = root_session(&m);
        const TOTAL: u64 = 10_000;
        const PAGE: usize = 64;
        for n in 1..=TOTAL {
            rt.board_post("subject", &format!("body-{n}"), &[]).unwrap();
        }

        // Newest page first: bounded, ordered, with an exclusive cursor.
        let page = rt.board_read_posts(None, None, PAGE, false).unwrap();
        assert_eq!(page.posts.len(), PAGE);
        assert!(page.has_more);
        assert_eq!(page.posts[0].id.raw(), TOTAL);
        assert_eq!(page.posts[PAGE - 1].id.raw(), TOTAL - PAGE as u64 + 1);
        let cursor = page.next_before_revision.unwrap();
        assert_eq!(cursor, TOTAL - PAGE as u64 + 1);

        // Next page resumes strictly below the cursor.
        let page2 = rt
            .board_read_posts(None, Some(cursor), PAGE, false)
            .unwrap();
        assert_eq!(page2.posts.len(), PAGE);
        assert_eq!(page2.posts[0].id.raw(), TOTAL - PAGE as u64);

        // A deep page near the start of the board is reachable in ONE
        // bounded call (the scan is capped; it never materializes all rows
        // into the page).
        let tail = rt.board_read_posts(None, Some(101), PAGE, false).unwrap();
        assert_eq!(tail.posts.len(), PAGE);
        assert_eq!(tail.posts[0].id.raw(), 100);
        assert_eq!(tail.posts[PAGE - 1].id.raw(), 100 - PAGE as u64 + 1);
        assert!(tail.has_more);

        // The final page is exact and terminates.
        let last = rt.board_read_posts(None, Some(65), PAGE, false).unwrap();
        assert_eq!(last.posts.len(), PAGE);
        assert_eq!(last.posts.last().unwrap().id.raw(), 1);
        assert!(!last.has_more);
        assert_eq!(last.next_before_revision, None);
    }

    #[test]
    fn delivery_view_full_matrix_never_infers_read_from_activity() {
        let (_d, m) = manager();
        let rt = root_session(&m);
        let c1 = child_session(&m, &rt, 2);
        let c2 = child_session(&m, &rt, 3);
        let c3 = child_session(&m, &rt, 4);
        let p = rt.board_post("status", "review please", &[]).unwrap();

        // (1) Stored: the root tracks no recipient; durability only.
        assert_eq!(
            rt.board_delivery_view(None, p.id).unwrap(),
            BoardDeliveryView::Stored
        );
        // (2) RecipientActive: c1 is live and has not read or acted.
        assert_eq!(
            rt.board_delivery_view(Some(child_id(&c1)), p.id).unwrap(),
            BoardDeliveryView::RecipientActive
        );
        // Activity NEVER infers Read: c2 reads the post (and c1 posts its own
        // unrelated post); c1 is still RecipientActive, and c2's marker never
        // leaks onto c1.
        c2.board_read_posts(None, None, 10, false).unwrap();
        c1.board_post("busy", "c1 is active", &[]).unwrap();
        assert_eq!(
            rt.board_delivery_view(Some(child_id(&c1)), p.id).unwrap(),
            BoardDeliveryView::RecipientActive
        );
        assert_eq!(
            rt.board_delivery_view(Some(child_id(&c2)), p.id).unwrap(),
            BoardDeliveryView::Read,
            "c2's own durable read marker is Read"
        );
        // (3) Read: c1's durable read marker.
        c1.board_read_posts(None, None, 10, false).unwrap();
        assert_eq!(
            rt.board_delivery_view(Some(child_id(&c1)), p.id).unwrap(),
            BoardDeliveryView::Read
        );
        // A non-read action receipt is ActedUpon regardless of the read row.
        c1.board_receipt(p.id, BoardAction::Ack, "on it").unwrap();
        assert_eq!(
            rt.board_delivery_view(Some(child_id(&c1)), p.id).unwrap(),
            BoardDeliveryView::ActedUpon
        );
        // A receipt by ANOTHER child never marks c1.
        c2.board_receipt(p.id, BoardAction::Question, "which check?")
            .unwrap();
        assert_eq!(
            rt.board_delivery_view(Some(child_id(&c3)), p.id).unwrap(),
            BoardDeliveryView::RecipientActive
        );
        // (4) RecipientInactive: the durable child lifecycle is terminal.
        c3.orchestrator_child_runtime_put(&ChildRuntimeBlockerRow {
            child_id: "child-3".into(),
            state: "cancelled".into(),
            blocker: None,
            updated_ms: 1,
        })
        .unwrap();
        assert_eq!(
            rt.board_delivery_view(Some(child_id(&c3)), p.id).unwrap(),
            BoardDeliveryView::RecipientInactive
        );
        // A terminal recipient that DID read stays Read (evidence wins).
        c3.board_read_posts(None, None, 10, false).unwrap();
        assert_eq!(
            rt.board_delivery_view(Some(child_id(&c3)), p.id).unwrap(),
            BoardDeliveryView::Read
        );
        // (5) Hidden history has no delivery view: typed NotFound.
        rt.board_reset(2).unwrap();
        assert_eq!(
            rt.board_delivery_view(Some(child_id(&c1)), p.id)
                .unwrap_err()
                .kind,
            ErrorKind::NotFound
        );
    }
}
