//! The durable monetary budget ledger (P0-6/12 + P0-1 settlement + P0-2):
//! the single authority over how much a task's MODEL CALLS may cost.
//!
//! The agent runtime used to keep a private in-memory spend map keyed by
//! session (`AgentDeps.budget_micro` + a `spent` HashMap): a second budget
//! authority that vanished on restart and never recorded WHY money moved.
//! This module replaces it with a durable ledger over the session store
//! (schema v17): one `cost_reservation` row per paid model call attempt,
//! settled against the task row's monetary columns in the same transaction.
//!
//! # Invariants (locked by adversarial tests in this file)
//!
//! - `reserve` is atomic: spent + in-flight predictions + unresolved
//!   UNCERTAIN holds > cap refuses with a typed `BudgetError::BudgetExceeded`
//!   and writes NOTHING (no row, no spend). `max_cost_micro` NULL/0 =
//!   unlimited. When the routing decision priced the call, its immutable
//!   [`faktor_core::model::PricingSnapshot`] is persisted on the row
//!   (`pricing_snapshot_json`) — settlement prices usage against the frozen
//!   route-time capture, never against a later catalog repricing.
//! - `mark_dispatched` writes the durable `dispatched_ms` marker
//!   immediately BEFORE the provider request is sent. Crash recovery splits
//!   surviving OPEN rows on that marker:
//!   - marker NULL (dispatch never provably began) -> REFUNDED: the
//!     provider was never contacted, the reservation is safe to release;
//!   - marker set (the request was sent and the provider MAY have billed) ->
//!     UNCERTAIN: the reservation KEEPS consuming the reserved amount
//!     (`free = max - spent - open - uncertain`) until a reconcile settles it
//!     from the op's durable provider-call rows or the task-completion
//!     finalize charges the reserved estimate. The old v15 rule (OPEN ->
//!     ABANDONED, charged $0) undercounted spend when a crash hit between
//!     provider billing and the local settle; `abandoned` no longer exists.
//! - `settle_usage` closes an OPEN reservation exactly once (OPEN ->
//!   SETTLED) and adds the CHOSEN actual to `task.spent_cost_micro` in ONE
//!   transaction. The chosen actual is the provider-reported cost when the
//!   usage frame carried one (authoritative), else the usage token
//!   categories (uncached input / cache reads / cache writes / output —
//!   reasoning billed at the output line) x the reservation's STORED
//!   snapshot. There is NO `tokens x 1 microUSD` local-price fallback
//!   anywhere: no price authority (no snapshot, or an Unknown-source
//!   snapshot) plus no provider-reported cost under a hard task cap is a
//!   typed [`BudgetError::UnknownPrice`] refusal (NOTHING written — the row
//!   stays OPEN and recovery/finalize resolve it conservatively); without a
//!   cap the row settles as a documented Unknown spend (amount columns NULL,
//!   nothing folded — never a fabricated zero or one). Both the locally
//!   calculated and the provider-reported amounts are recorded on the row.
//!   A settlement that would push spent past the cap is recorded honestly —
//!   the money WAS spent; the NEXT reservation is what refuses.
//! - A second settle / a settle after refund is a typed `NotOpen` error
//!   (exactly-once semantics per reservation).
//! - `refund` releases an OPEN reservation without spending (exactly-once).
//! - `reconcile_uncertain` settles every UNCERTAIN reservation of one task
//!   whose op has a completed durable `provider_call` row FROM that row:
//!   the completed call's tokens at the reservation's stored snapshot. A
//!   later settle for the same op id settles the crashed attempt (each
//!   dispatched attempt may have been billed). Rows without a completed row
//!   — or unpriced under a hard cap — stay UNCERTAIN.
//! - `finalize_uncertain` is the conservative task-completion backstop:
//!   every still-UNCERTAIN reservation of the task settles AT ITS RESERVED
//!   ESTIMATE (`predicted_micro`) exactly once. Idempotent.
//! - Crash recovery ([`DurableBudgetLedger::recover_after_restart`]) is
//!   idempotent: a second run finds nothing OPEN.
//! - Reservation ids are SQLite AUTOINCREMENT row ids: monotonic across
//!   daemon restarts, never reused.
//!
//! The trait ([`BudgetAuthority`]) is what the runtime consumes; tests that
//! never set a cap inject [`NoopBudget`]. The task row's monetary columns
//! are READ-ONLY through the task machine (upsert_task never writes them);
//! this ledger is their only writer.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use faktor_core::id::{OpId, SessionId, TaskId};
use faktor_core::model::PricingSnapshot;
use tokio::task::JoinError;

use crate::manager::SessionManager;
use crate::SessionError;

/// Bound on the routing decision JSON stored on a reservation row
/// (bounded everything: an audit string is never stored unbounded).
pub const MAX_ROUTE_DECISION_JSON_BYTES: usize = 8192;

/// Bound on the pricing-snapshot JSON persisted at reserve time (the
/// snapshot is four u64 price lines + a source tag — far smaller — but a
/// hostile capture must never grow the row without limit).
pub const MAX_PRICING_SNAPSHOT_JSON_BYTES: usize = 1024;

/// One reservation's durable id (the store row's AUTOINCREMENT id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReservationId(i64);

impl ReservationId {
    /// The NOOP ledger's canonical id (never a store row).
    pub const NOOP: ReservationId = ReservationId(0);

    pub fn new(raw: i64) -> Self {
        Self(raw)
    }

    pub fn raw(&self) -> i64 {
        self.0
    }
}

impl std::fmt::Display for ReservationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Typed refusal of a ledger operation. Every variant is machine-readable;
/// prose-only errors are rejected in review.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BudgetError {
    #[error(
        "budget exceeded: reserving {predicted} micro would push spend past the task cap (free: {free})"
    )]
    BudgetExceeded { free: u64, predicted: u64 },
    #[error("reservation {0} does not exist")]
    UnknownReservation(i64),
    #[error("reservation {reservation} is {status}, not open: exactly-once per reservation")]
    NotOpen { reservation: i64, status: String },
    #[error(
        "reservation {reservation} has no price authority (no snapshot or an Unknown price source) \
         and no provider-reported cost, under a hard task cap: settlement is refused — nothing was \
         written (recovery/finalize resolve the row conservatively)"
    )]
    UnknownPrice { reservation: i64 },
    #[error("task {task_id} of session {session_id} has no row (the task machine owns creation)")]
    MissingTask {
        session_id: SessionId,
        task_id: TaskId,
    },
    #[error("malformed ledger input: {0}")]
    Malformed(String),
    #[error("store failure: {0}")]
    Store(String),
    #[error("ledger worker failed: {0}")]
    Worker(String),
}

impl From<faktor_store::StoreError> for BudgetError {
    fn from(e: faktor_store::StoreError) -> Self {
        // The store's only Conflict on these paths is a missing task row.
        if let faktor_store::StoreError::Conflict(m) = &e {
            if m.contains("has no row") {
                return BudgetError::Malformed(m.clone());
            }
        }
        BudgetError::Store(e.to_string())
    }
}

impl From<JoinError> for BudgetError {
    fn from(e: JoinError) -> Self {
        BudgetError::Worker(e.to_string())
    }
}

impl From<BudgetError> for SessionError {
    fn from(e: BudgetError) -> Self {
        match e {
            BudgetError::BudgetExceeded { .. } => SessionError::Conflict(e.to_string()),
            BudgetError::UnknownReservation(_)
            | BudgetError::NotOpen { .. }
            | BudgetError::UnknownPrice { .. } => SessionError::Conflict(e.to_string()),
            BudgetError::MissingTask { .. } => SessionError::NotFound(e.to_string()),
            BudgetError::Malformed(m) => SessionError::Malformed(m),
            BudgetError::Store(m) => SessionError::Store(faktor_store::StoreError::Conflict(m)),
            BudgetError::Worker(m) => SessionError::Internal(m),
        }
    }
}

impl From<BudgetError> for faktor_core::Error {
    fn from(e: BudgetError) -> Self {
        SessionError::from(e).into()
    }
}

/// The task's durable monetary picture at one moment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetView {
    /// The task's cap in microUSD; `None` = unlimited.
    pub max_cost_micro: Option<u64>,
    /// Durable settled spend (the task row's `spent_cost_micro`).
    pub spent_cost_micro: u64,
    /// Sum of the predicted micro of every OPEN reservation (in flight).
    pub open_reserved_micro: u64,
    /// Count of OPEN reservations.
    pub open_reservations: usize,
    /// Sum of the predicted micro of every UNCERTAIN reservation (a crashed
    /// daemon may have dispatched them; unresolved by reconcile/finalize).
    /// These KEEP consuming the reserved amount: the provider may have
    /// billed the attempt, so its prediction is never silently released.
    pub uncertain_reserved_micro: u64,
    /// Count of UNCERTAIN reservations.
    pub uncertain_reservations: usize,
    /// Count of SETTLED reservations.
    pub settled_count: usize,
}

impl BudgetView {
    /// Free balance: cap - spent - in-flight predictions - unresolved
    /// UNCERTAIN holds (saturating; the ledger refuses reservations that
    /// would cross it).
    pub fn free(&self) -> u64 {
        match self.max_cost_micro {
            None => u64::MAX,
            Some(cap) => cap
                .saturating_sub(self.spent_cost_micro)
                .saturating_sub(self.open_reserved_micro)
                .saturating_sub(self.uncertain_reserved_micro),
        }
    }
}

pub(crate) type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The budget authority the agent runtime consumes before every paid model
/// call. Async operations carry explicit futures (no hidden spawns); every
/// implementation is `Send + Sync` and must make its own durability story.
pub trait BudgetAuthority: Send + Sync {
    /// Reserve `predicted_micro` of the task's budget for one operation.
    /// `pricing_snapshot` is the immutable route-time price capture the
    /// settlement math prices the call's usage against; `None` = no pricing
    /// authority was consulted (settlement then fails closed under a hard
    /// cap, or records a documented Unknown spend without one). Fails with
    /// typed `BudgetError::BudgetExceeded` when
    /// spent + in-flight + uncertain > cap (`None`/0 cap = unlimited); a
    /// refusal writes NOTHING.
    fn reserve(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        op_id: OpId,
        predicted_micro: u64,
        pricing_snapshot: Option<PricingSnapshot>,
    ) -> BoxFut<'_, Result<ReservationId, BudgetError>>;

    /// Write the durable `dispatched_ms` marker of one reservation
    /// (immediately BEFORE the provider request is sent): crash recovery
    /// can then tell "dispatch never provably began" (marker NULL ->
    /// refunded) from "the provider may have billed" (marker set ->
    /// UNCERTAIN). A second mark of the same still-open reservation is
    /// idempotent; anything not `open` is a typed refusal.
    fn mark_dispatched(
        &self,
        session_id: SessionId,
        reservation: ReservationId,
    ) -> BoxFut<'_, Result<(), BudgetError>>;

    /// Settle one reservation from a provider usage frame at the chosen
    /// actual: the provider-reported cost when the frame carried one
    /// (authoritative), else the token categories x the reservation's
    /// stored route-time [`PricingSnapshot`]. Both amounts and the routing
    /// decision's JSON are recorded on the row; the chosen actual is folded
    /// into the task's spent total in the same transaction. `Ok(Some(m))`
    /// = `m` microUSD was folded; `Ok(None)` = the row settled as a
    /// documented Unknown spend (no price authority, no hard cap — nothing
    /// folded, never a fabricated number). A price-less settle under a hard
    /// cap is the typed [`BudgetError::UnknownPrice`] refusal (nothing
    /// written; the row stays OPEN for recovery/finalize).
    #[allow(clippy::too_many_arguments)]
    fn settle_usage(
        &self,
        session_id: SessionId,
        reservation: ReservationId,
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
        provider_reported_micro: Option<u64>,
        route_decision_json: Option<String>,
    ) -> BoxFut<'_, Result<Option<u64>, BudgetError>>;

    /// Release one reservation without spending (OPEN -> REFUNDED,
    /// exactly-once).
    fn refund(
        &self,
        session_id: SessionId,
        reservation: ReservationId,
    ) -> BoxFut<'_, Result<(), BudgetError>>;

    /// The task's durable monetary picture (sync read; used by route
    /// requests for the remaining-budget axis and by diagnostics).
    fn session_budget_view(&self, session_id: SessionId, task_id: TaskId) -> BudgetView;

    /// Crash recovery of every reservation a crashed process left OPEN,
    /// split on the durable dispatch marker (marker NULL -> REFUNDED,
    /// marker set -> UNCERTAIN which keeps consuming). Idempotent.
    fn recover_after_restart(&self);

    /// Reconcile every UNCERTAIN reservation of one task whose op has a
    /// completed durable `provider_call` row: the completed call's tokens,
    /// priced at each reservation's stored snapshot, settle the crashed
    /// attempt (a later settle for the same op id settles it). Rows without
    /// a completed row — or unpriced under a hard cap — stay UNCERTAIN for
    /// the task-completion finalize. Idempotent.
    fn reconcile_uncertain(
        &self,
        session_id: SessionId,
        task_id: TaskId,
    ) -> BoxFut<'_, Result<faktor_store::CostReconcileReport, BudgetError>>;

    /// Conservative task-completion finalize: every still-UNCERTAIN
    /// reservation of one task settles AT its reserved estimate (the
    /// provider may have billed for a dispatched attempt whose actual never
    /// reconciled). Called once at task completion; idempotent — a second
    /// pass finds nothing UNCERTAIN and never double-charges.
    fn finalize_uncertain(
        &self,
        session_id: SessionId,
        task_id: TaskId,
    ) -> BoxFut<'_, Result<faktor_store::CostFinalizeReport, BudgetError>>;
}

/// The durable ledger: reservations + the task row's monetary columns, both
/// owned by this type over one [`SessionManager`]'s store. Cheap to build;
/// shares the manager's store and CAS.
pub struct DurableBudgetLedger {
    session: Arc<SessionManager>,
}

impl std::fmt::Debug for DurableBudgetLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableBudgetLedger")
            .finish_non_exhaustive()
    }
}

impl DurableBudgetLedger {
    pub fn new(session: Arc<SessionManager>) -> Arc<Self> {
        Arc::new(Self { session })
    }

    fn store(&self) -> Arc<faktor_store::Store> {
        self.session.store()
    }

    fn now_ms(&self) -> i64 {
        self.session.now_ms()
    }

    /// Set (or clear) the durable monetary cap of one task. `None` =
    /// unlimited. The task row must exist (the task machine owns creation).
    /// The cap is only ever consulted by reservations; spend never rewinds.
    pub fn set_task_max_cost(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        max_cost_micro: Option<u64>,
    ) -> Result<(), BudgetError> {
        let store = self.store();
        let (session_id, task_id) = (session_id, task_id);
        store
            .cost_task_cap_set(session_id, task_id, max_cost_micro)
            .map_err(|e| {
                if matches!(e, faktor_store::StoreError::Conflict(_)) {
                    BudgetError::MissingTask {
                        session_id,
                        task_id,
                    }
                } else {
                    BudgetError::Store(e.to_string())
                }
            })
    }

    /// Resolve every reservation a crashed process left OPEN (split on the
    /// durable dispatch marker, see [`BudgetAuthority::recover_after_restart`]).
    /// Inherent alias of the [`BudgetAuthority`] method so daemon boot code
    /// can call it without importing the trait. Idempotent.
    pub fn recover_after_restart(&self) {
        BudgetAuthority::recover_after_restart(self)
    }

    fn map_row_err(
        e: faktor_store::StoreError,
        session_id: SessionId,
        task_id: TaskId,
    ) -> BudgetError {
        match e {
            faktor_store::StoreError::Conflict(m)
                if m.contains("cost reserve: task") && m.contains("has no row") =>
            {
                BudgetError::MissingTask {
                    session_id,
                    task_id,
                }
            }
            other => BudgetError::Store(other.to_string()),
        }
    }
}

/// Future-returning helpers so the trait impls stay one-liners.
impl DurableBudgetLedger {
    async fn reserve_inner(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        op_id: OpId,
        predicted_micro: u64,
        pricing_snapshot: Option<PricingSnapshot>,
    ) -> Result<ReservationId, BudgetError> {
        let store = self.store();
        let created_ms = self.now_ms();
        // Bounded: a pricing snapshot is four u64 lines + a source tag;
        // anything larger is malformed input, never stored.
        let snapshot_json = match pricing_snapshot {
            Some(s) => {
                let json = serde_json::to_string(&s)
                    .map_err(|e| BudgetError::Malformed(format!("pricing snapshot: {e}")))?;
                if json.len() > MAX_PRICING_SNAPSHOT_JSON_BYTES {
                    return Err(BudgetError::Malformed(format!(
                        "pricing snapshot JSON of {} bytes exceeds MAX_PRICING_SNAPSHOT_JSON_BYTES",
                        json.len()
                    )));
                }
                Some(json)
            }
            None => None,
        };
        tokio::task::spawn_blocking(move || {
            store.cost_reserve_priced(
                session_id,
                task_id,
                op_id,
                predicted_micro,
                created_ms,
                snapshot_json.as_deref(),
            )
        })
        .await
        .map_err(BudgetError::from)?
        .map_err(|e| Self::map_row_err(e, session_id, task_id))
        .and_then(|out| match out {
            faktor_store::CostReserveOutcome::Granted(id) => Ok(ReservationId::new(id)),
            faktor_store::CostReserveOutcome::Exceeded { free } => {
                Err(BudgetError::BudgetExceeded {
                    free,
                    predicted: predicted_micro,
                })
            }
        })
    }

    async fn mark_dispatched_inner(&self, reservation: ReservationId) -> Result<(), BudgetError> {
        let store = self.store();
        let at_ms = self.now_ms();
        tokio::task::spawn_blocking(move || store.cost_mark_dispatched(reservation.raw(), at_ms))
            .await
            .map_err(BudgetError::from)?
            .map_err(BudgetError::from)
            .and_then(map_reservation_state(reservation))
    }

    #[allow(clippy::too_many_arguments)]
    async fn settle_usage_inner(
        &self,
        reservation: ReservationId,
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
        provider_reported_micro: Option<u64>,
        route_decision_json: Option<String>,
    ) -> Result<Option<u64>, BudgetError> {
        if let Some(j) = &route_decision_json {
            if j.len() > MAX_ROUTE_DECISION_JSON_BYTES {
                return Err(BudgetError::Malformed(format!(
                    "route decision JSON of {} bytes exceeds MAX_ROUTE_DECISION_JSON_BYTES",
                    j.len()
                )));
            }
        }
        let store = self.store();
        let settled_ms = self.now_ms();
        // The chosen actual is decided in the STORE's settlement transaction
        // (reported wins; else categories x the row's stored snapshot) —
        // never recomputed here from a caller-supplied micro number: there
        // is no tokens x 1 microUSD fallback anywhere on this path.
        tokio::task::spawn_blocking(move || {
            store.cost_settle_usage(
                reservation.raw(),
                uncached_input_tokens,
                cache_read_tokens,
                cache_write_tokens,
                output_tokens,
                provider_reported_micro,
                route_decision_json.as_deref(),
                settled_ms,
            )
        })
        .await
        .map_err(BudgetError::from)?
        .map_err(BudgetError::from)
        .and_then(|out| match out {
            faktor_store::CostSettleOutcome::Applied { actual_micro } => Ok(Some(actual_micro)),
            // Settled as a documented Unknown spend (no price authority, no
            // hard cap): NOTHING was folded — never a fabricated zero.
            faktor_store::CostSettleOutcome::AppliedUnknown => Ok(None),
            faktor_store::CostSettleOutcome::Missing => {
                Err(BudgetError::UnknownReservation(reservation.raw()))
            }
            faktor_store::CostSettleOutcome::NotOpen { current } => Err(BudgetError::NotOpen {
                reservation: reservation.raw(),
                status: current,
            }),
            faktor_store::CostSettleOutcome::UnknownPrice { reservation: raw } => {
                Err(BudgetError::UnknownPrice { reservation: raw })
            }
        })
    }

    async fn refund_inner(&self, reservation: ReservationId) -> Result<(), BudgetError> {
        let store = self.store();
        let settled_ms = self.now_ms();
        tokio::task::spawn_blocking(move || store.cost_refund(reservation.raw(), settled_ms))
            .await
            .map_err(BudgetError::from)?
            .map_err(BudgetError::from)
            .and_then(map_reservation_state(reservation))
    }

    /// The durable reservations of one task (newest first, bounded) —
    /// observability/adversarial-test surface.
    pub fn reservations_of(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        limit: i64,
    ) -> Result<Vec<faktor_store::CostReservationRow>, BudgetError> {
        self.store()
            .cost_reservations_of(session_id, task_id, limit)
            .map_err(BudgetError::from)
    }

    fn view_inner(&self, session_id: SessionId, task_id: TaskId) -> BudgetView {
        let store = self.store();
        let (max, spent) = store
            .cost_task_row(session_id, task_id)
            .ok()
            .flatten()
            .map(|r| (r.max_cost_micro, r.spent_cost_micro))
            .unwrap_or((None, 0));
        let (open_micro, open_count, uncertain_micro, uncertain_count, settled_count) = store
            .cost_reservations_of(session_id, task_id, i64::MAX)
            .ok()
            .map(|rows| {
                let mut open_micro = 0u64;
                let mut open_count = 0usize;
                let mut uncertain_micro = 0u64;
                let mut uncertain_count = 0usize;
                let mut settled_count = 0usize;
                for r in &rows {
                    match r.status.as_str() {
                        "open" => {
                            open_micro = open_micro.saturating_add(r.predicted_micro);
                            open_count += 1;
                        }
                        "uncertain" => {
                            uncertain_micro = uncertain_micro.saturating_add(r.predicted_micro);
                            uncertain_count += 1;
                        }
                        "settled" => settled_count += 1,
                        _ => {}
                    }
                }
                (
                    open_micro,
                    open_count,
                    uncertain_micro,
                    uncertain_count,
                    settled_count,
                )
            })
            .unwrap_or((0, 0, 0, 0, 0));
        BudgetView {
            max_cost_micro: max,
            spent_cost_micro: spent,
            open_reserved_micro: open_micro,
            open_reservations: open_count,
            uncertain_reserved_micro: uncertain_micro,
            uncertain_reservations: uncertain_count,
            settled_count,
        }
    }

    fn recover_inner(&self) {
        let store = self.store();
        let at = self.now_ms();
        match store.cost_recover_open_reservations(at) {
            Ok((refunded, uncertain)) => {
                tracing::debug!(
                    "budget ledger crash recovery: {refunded} never-dispatched reservations \
                     refunded, {uncertain} may-have-dispatched reservations marked UNCERTAIN"
                );
            }
            Err(e) => tracing::warn!("budget ledger crash recovery failed: {e}"),
        }
    }

    async fn reconcile_uncertain_inner(
        &self,
        session_id: SessionId,
        task_id: TaskId,
    ) -> Result<faktor_store::CostReconcileReport, BudgetError> {
        let store = self.store();
        let at_ms = self.now_ms();
        tokio::task::spawn_blocking(move || {
            store.cost_reconcile_uncertain(session_id, task_id, at_ms)
        })
        .await
        .map_err(BudgetError::from)?
        .map_err(BudgetError::from)
    }

    async fn finalize_uncertain_inner(
        &self,
        session_id: SessionId,
        task_id: TaskId,
    ) -> Result<faktor_store::CostFinalizeReport, BudgetError> {
        let store = self.store();
        let at_ms = self.now_ms();
        tokio::task::spawn_blocking(move || {
            store.cost_finalize_uncertain(session_id, task_id, at_ms)
        })
        .await
        .map_err(BudgetError::from)?
        .map_err(BudgetError::from)
    }
}

fn map_reservation_state(
    reservation: ReservationId,
) -> impl FnOnce(faktor_store::CostReservationState) -> Result<(), BudgetError> {
    move |out| match out {
        faktor_store::CostReservationState::Applied => Ok(()),
        faktor_store::CostReservationState::Missing => {
            Err(BudgetError::UnknownReservation(reservation.raw()))
        }
        faktor_store::CostReservationState::NotOpen { current } => Err(BudgetError::NotOpen {
            reservation: reservation.raw(),
            status: current,
        }),
    }
}

impl BudgetAuthority for DurableBudgetLedger {
    fn reserve(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        op_id: OpId,
        predicted_micro: u64,
        pricing_snapshot: Option<PricingSnapshot>,
    ) -> BoxFut<'_, Result<ReservationId, BudgetError>> {
        Box::pin(self.reserve_inner(
            session_id,
            task_id,
            op_id,
            predicted_micro,
            pricing_snapshot,
        ))
    }

    fn mark_dispatched(
        &self,
        _session_id: SessionId,
        reservation: ReservationId,
    ) -> BoxFut<'_, Result<(), BudgetError>> {
        Box::pin(self.mark_dispatched_inner(reservation))
    }

    fn settle_usage(
        &self,
        _session_id: SessionId,
        reservation: ReservationId,
        uncached_input_tokens: u64,
        cache_read_tokens: u64,
        cache_write_tokens: u64,
        output_tokens: u64,
        provider_reported_micro: Option<u64>,
        route_decision_json: Option<String>,
    ) -> BoxFut<'_, Result<Option<u64>, BudgetError>> {
        Box::pin(self.settle_usage_inner(
            reservation,
            uncached_input_tokens,
            cache_read_tokens,
            cache_write_tokens,
            output_tokens,
            provider_reported_micro,
            route_decision_json,
        ))
    }

    fn refund(
        &self,
        _session_id: SessionId,
        reservation: ReservationId,
    ) -> BoxFut<'_, Result<(), BudgetError>> {
        Box::pin(self.refund_inner(reservation))
    }

    fn session_budget_view(&self, session_id: SessionId, task_id: TaskId) -> BudgetView {
        self.view_inner(session_id, task_id)
    }

    fn recover_after_restart(&self) {
        self.recover_inner();
    }

    fn reconcile_uncertain(
        &self,
        session_id: SessionId,
        task_id: TaskId,
    ) -> BoxFut<'_, Result<faktor_store::CostReconcileReport, BudgetError>> {
        Box::pin(self.reconcile_uncertain_inner(session_id, task_id))
    }

    fn finalize_uncertain(
        &self,
        session_id: SessionId,
        task_id: TaskId,
    ) -> BoxFut<'_, Result<faktor_store::CostFinalizeReport, BudgetError>> {
        Box::pin(self.finalize_uncertain_inner(session_id, task_id))
    }
}

/// The unlimited ledger of test graphs that never set a cap: every reserve
/// grants, every settle/refund records nothing. `session_budget_view`
/// reports an unlimited budget with no activity.
#[derive(Debug, Default)]
pub struct NoopBudget;

impl BudgetAuthority for NoopBudget {
    fn reserve(
        &self,
        _session_id: SessionId,
        _task_id: TaskId,
        _op_id: OpId,
        _predicted_micro: u64,
        _pricing_snapshot: Option<PricingSnapshot>,
    ) -> BoxFut<'_, Result<ReservationId, BudgetError>> {
        Box::pin(async { Ok(ReservationId::NOOP) })
    }

    fn mark_dispatched(
        &self,
        _session_id: SessionId,
        _reservation: ReservationId,
    ) -> BoxFut<'_, Result<(), BudgetError>> {
        Box::pin(async { Ok(()) })
    }

    fn settle_usage(
        &self,
        _session_id: SessionId,
        _reservation: ReservationId,
        _uncached_input_tokens: u64,
        _cache_read_tokens: u64,
        _cache_write_tokens: u64,
        _output_tokens: u64,
        _provider_reported_micro: Option<u64>,
        _route_decision_json: Option<String>,
    ) -> BoxFut<'_, Result<Option<u64>, BudgetError>> {
        Box::pin(async { Ok(None) })
    }

    fn refund(
        &self,
        _session_id: SessionId,
        _reservation: ReservationId,
    ) -> BoxFut<'_, Result<(), BudgetError>> {
        Box::pin(async { Ok(()) })
    }

    fn session_budget_view(&self, _session_id: SessionId, _task_id: TaskId) -> BudgetView {
        BudgetView {
            max_cost_micro: None,
            spent_cost_micro: 0,
            open_reserved_micro: 0,
            open_reservations: 0,
            uncertain_reserved_micro: 0,
            uncertain_reservations: 0,
            settled_count: 0,
        }
    }

    fn recover_after_restart(&self) {}

    fn reconcile_uncertain(
        &self,
        _session_id: SessionId,
        _task_id: TaskId,
    ) -> BoxFut<'_, Result<faktor_store::CostReconcileReport, BudgetError>> {
        Box::pin(async { Ok(faktor_store::CostReconcileReport::default()) })
    }

    fn finalize_uncertain(
        &self,
        _session_id: SessionId,
        _task_id: TaskId,
    ) -> BoxFut<'_, Result<faktor_store::CostFinalizeReport, BudgetError>> {
        Box::pin(async { Ok(faktor_store::CostFinalizeReport::default()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use faktor_core::state::TaskState;

    /// The audit's settlement-truth identity: $15/1M in + $60/1M out ==
    /// 15/60 microUSD per token.
    fn known_snapshot() -> PricingSnapshot {
        PricingSnapshot {
            input_micro_per_token: 15,
            output_micro_per_token: 60,
            cache_read_micro_per_token: 3,
            cache_write_micro_per_token: 7,
            pricing_epoch: 7,
            source: faktor_core::model::PriceSource::Known,
        }
    }

    fn seeded_task(s: &crate::SessionHandle, max_cost: Option<u64>) -> TaskId {
        let task = crate::Task {
            task_id: s.task_id().unwrap(),
            session_id: s.id,
            goal: "budgeted goal".into(),
            acceptance_criteria: vec![],
            plan: vec![],
            budget: crate::TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: 1,
            updated_ms: 1,
        };
        let created = s.create_task(task).unwrap();
        let ledger = DurableBudgetLedger::new(s.manager.clone());
        ledger
            .set_task_max_cost(s.id, created.task_id, max_cost)
            .unwrap();
        created.task_id
    }

    fn fresh_ledger() -> (
        tempfile::TempDir,
        Arc<SessionManager>,
        Arc<DurableBudgetLedger>,
    ) {
        let (_d, m) = test_manager();
        let ledger = DurableBudgetLedger::new(m.clone());
        (_d, m, ledger)
    }

    #[tokio::test]
    async fn reserve_over_budget_is_typed_and_writes_nothing() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(100));
        let err = ledger
            .reserve(s.id, task, OpId::new(1), 101, None)
            .await
            .unwrap_err();
        assert_eq!(
            err,
            BudgetError::BudgetExceeded {
                free: 100,
                predicted: 101
            }
        );
        // No row, no spend: a refusal never leaves a trace.
        assert!(ledger.reservations_of(s.id, task, 10).unwrap().is_empty());
        assert_eq!(ledger.session_budget_view(s.id, task).spent_cost_micro, 0);
        // Unlimited (cap cleared) grants anything.
        ledger.set_task_max_cost(s.id, task, None).unwrap();
        let granted = ledger
            .reserve(s.id, task, OpId::new(1), u64::MAX / 2, None)
            .await
            .unwrap();
        assert!(granted.raw() > 0);
        ledger.refund(s.id, granted).await.unwrap();
    }

    #[tokio::test]
    async fn settle_usage_updates_spent_exactly_once_and_refuses_a_second_settle() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        // Cap 2_000_000 microUSD ($2): the exactness example settles 1_620_000
        // within it and still leaves a tested refusal band.
        let task = seeded_task(&s, Some(2_000_000));
        let r1 = ledger
            .reserve(s.id, task, OpId::new(10), 400, Some(known_snapshot()))
            .await
            .unwrap();
        let r2 = ledger
            .reserve(s.id, task, OpId::new(11), 400, Some(known_snapshot()))
            .await
            .unwrap();
        // No provider report: categories x the stored snapshot settle the
        // local actual (100k uncached in @15 + 2k out @60 == 1_500_000 +
        // 120_000 == 1_620_000 microUSD == $1.62 — exact multiply-add).
        let actual = ledger
            .settle_usage(s.id, r1, 100_000, 0, 0, 2_000, None, None)
            .await
            .unwrap();
        assert_eq!(actual, Some(1_620_000), "exact category x snapshot math");
        let view = ledger.session_budget_view(s.id, task);
        assert_eq!(view.spent_cost_micro, 1_620_000);
        assert_eq!(view.open_reservations, 1);
        assert_eq!(view.open_reserved_micro, 400);
        // Double settle: typed NotOpen, nothing changes.
        let err = ledger
            .settle_usage(s.id, r1, 0, 0, 0, 1, None, None)
            .await
            .unwrap_err();
        assert!(matches!(err, BudgetError::NotOpen { .. }));
        assert_eq!(
            ledger.session_budget_view(s.id, task).spent_cost_micro,
            1_620_000
        );
        // Reserve over the remaining cap is refused (free = 2_000_000 -
        // 1_620_000 - 400 = 379_600).
        let err = ledger
            .reserve(s.id, task, OpId::new(12), 400_000, None)
            .await
            .unwrap_err();
        assert_eq!(
            err,
            BudgetError::BudgetExceeded {
                free: 379_600,
                predicted: 400_000
            }
        );
        ledger
            .settle_usage(s.id, r2, 0, 0, 0, 0, Some(400), None)
            .await
            .unwrap();
        assert_eq!(
            ledger.session_budget_view(s.id, task).spent_cost_micro,
            1_620_400
        );
    }

    #[tokio::test]
    async fn provider_reported_cost_wins_and_both_columns_are_recorded() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(50_000));
        let r = ledger
            .reserve(s.id, task, OpId::new(20), 100, Some(known_snapshot()))
            .await
            .unwrap();
        // Locally calculated: 100k x 15 + 2k x 60 == 1_620_000 microUSD
        // ($1.62). The provider reports 9_999: the REPORTED cost is
        // authoritative.
        let chosen = ledger
            .settle_usage(
                s.id,
                r,
                100_000,
                0,
                0,
                2_000,
                Some(9_999),
                Some(r#"{"provider":"p","model":"m"}"#.into()),
            )
            .await
            .unwrap();
        assert_eq!(chosen, Some(9_999));
        let rows = ledger.reservations_of(s.id, task, 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, "settled");
        assert_eq!(rows[0].provider_cost_micro, Some(1_620_000));
        assert_eq!(rows[0].provider_reported_micro, Some(9_999));
        assert_eq!(rows[0].pricing_snapshot, Some(known_snapshot()));
        assert_eq!(
            rows[0].route_decision_json.as_deref(),
            Some(r#"{"provider":"p","model":"m"}"#)
        );
        // The CHOSEN actual is the provider-reported cost.
        assert_eq!(
            ledger.session_budget_view(s.id, task).spent_cost_micro,
            9_999,
            "provider-reported cost is authoritative over the local actual"
        );
    }

    #[tokio::test]
    async fn cache_lines_bill_at_their_own_snapshot_lines() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(50_000));
        let r = ledger
            .reserve(s.id, task, OpId::new(21), 100, Some(known_snapshot()))
            .await
            .unwrap();
        // 10k cache reads @3 + 4k cache writes @7 + 2k out @60 == 30_000 +
        // 28_000 + 120_000 == 178_000 microUSD; no provider report.
        let chosen = ledger
            .settle_usage(s.id, r, 0, 10_000, 4_000, 2_000, None, None)
            .await
            .unwrap();
        assert_eq!(chosen, Some(178_000));
        assert_eq!(
            ledger.session_budget_view(s.id, task).spent_cost_micro,
            178_000
        );
    }

    #[tokio::test]
    async fn price_less_settle_under_a_cap_is_a_typed_refusal_that_writes_nothing() {
        // Unknown price source (or no snapshot at all) + no provider cost +
        // a hard budget: typed UnknownPrice, NOTHING written — the row stays
        // OPEN and keeps consuming (recovery/finalize resolve it).
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(1_000));
        // No snapshot was persisted at reserve (passthrough route).
        let r = ledger
            .reserve(s.id, task, OpId::new(30), 600, None)
            .await
            .unwrap();
        let err = ledger
            .settle_usage(s.id, r, 1_000, 0, 0, 2_000, None, None)
            .await
            .unwrap_err();
        assert_eq!(
            err,
            BudgetError::UnknownPrice {
                reservation: r.raw()
            }
        );
        // Nothing was written: the row is still OPEN, still consuming.
        let rows = ledger.reservations_of(s.id, task, 10).unwrap();
        assert_eq!(rows[0].status, "open");
        assert_eq!(rows[0].provider_cost_micro, None);
        let view = ledger.session_budget_view(s.id, task);
        assert_eq!(view.spent_cost_micro, 0);
        assert_eq!(view.open_reservations, 1);
        assert_eq!(view.open_reserved_micro, 600);
        // An Unknown-SOURCE snapshot behaves identically (never a number).
        let r2 = ledger
            .reserve(
                s.id,
                task,
                OpId::new(31),
                100,
                Some(PricingSnapshot {
                    input_micro_per_token: 15,
                    output_micro_per_token: 60,
                    cache_read_micro_per_token: 0,
                    cache_write_micro_per_token: 0,
                    pricing_epoch: 1,
                    source: faktor_core::model::PriceSource::Unknown,
                }),
            )
            .await
            .unwrap();
        let err = ledger
            .settle_usage(s.id, r2, 1_000, 0, 0, 2_000, None, None)
            .await
            .unwrap_err();
        assert_eq!(
            err,
            BudgetError::UnknownPrice {
                reservation: r2.raw()
            }
        );
        ledger.refund(s.id, r).await.unwrap();
        ledger.refund(s.id, r2).await.unwrap();
    }

    #[tokio::test]
    async fn price_less_settle_without_a_cap_records_an_unknown_spend_never_a_number() {
        // No price authority + no provider cost + NO hard cap: the row
        // settles as a documented Unknown spend — amount columns NULL,
        // nothing folded into the task total, never a fabricated 0 or 1.
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, None);
        let r = ledger
            .reserve(s.id, task, OpId::new(40), 100, None)
            .await
            .unwrap();
        let chosen = ledger
            .settle_usage(s.id, r, 1_000, 0, 0, 2_000, None, None)
            .await
            .unwrap();
        assert_eq!(chosen, None, "Ok(None) = unknown spend, nothing folded");
        let view = ledger.session_budget_view(s.id, task);
        assert_eq!(view.spent_cost_micro, 0, "nothing was folded");
        let rows = ledger.reservations_of(s.id, task, 10).unwrap();
        assert_eq!(rows[0].status, "settled");
        assert_eq!(rows[0].provider_cost_micro, None);
        assert_eq!(rows[0].provider_reported_micro, None);
    }

    #[tokio::test]
    async fn refund_is_exactly_once_and_releases_without_spending() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(100));
        let r = ledger
            .reserve(s.id, task, OpId::new(30), 60, None)
            .await
            .unwrap();
        ledger.refund(s.id, r).await.unwrap();
        assert_eq!(ledger.session_budget_view(s.id, task).spent_cost_micro, 0);
        // A refunded reservation refuses settle AND a second refund.
        assert!(matches!(
            ledger
                .settle_usage(s.id, r, 1, 0, 0, 0, None, None)
                .await
                .unwrap_err(),
            BudgetError::NotOpen { .. }
        ));
        assert!(matches!(
            ledger.refund(s.id, r).await.unwrap_err(),
            BudgetError::NotOpen { .. }
        ));
        // Space is available again.
        let again = ledger
            .reserve(s.id, task, OpId::new(31), 90, None)
            .await
            .unwrap();
        ledger.refund(s.id, again).await.unwrap();
    }

    #[tokio::test]
    async fn settle_overshoot_is_recorded_honestly_and_the_next_reserve_refuses() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(100));
        // Reserve 60 (fits), provider reports 150: settlement is NOT
        // refused — the money was spent — the spent total honestly exceeds
        // the cap and the NEXT reserve refuses.
        let r = ledger
            .reserve(s.id, task, OpId::new(40), 60, None)
            .await
            .unwrap();
        ledger
            .settle_usage(s.id, r, 0, 0, 0, 0, Some(150), None)
            .await
            .unwrap();
        let view = ledger.session_budget_view(s.id, task);
        assert_eq!(
            view.spent_cost_micro, 150,
            "overspend is recorded, not hidden"
        );
        let err = ledger
            .reserve(s.id, task, OpId::new(41), 1, None)
            .await
            .unwrap_err();
        assert_eq!(
            err,
            BudgetError::BudgetExceeded {
                free: 0,
                predicted: 1
            }
        );
    }

    #[tokio::test]
    async fn crash_before_the_dispatch_marker_is_refunded_and_free_is_restored() {
        // Crash BEFORE the provider request was sent (the durable marker
        // was never written): recovery REFUNDS the open reservation — the
        // provider was provably never contacted — and the free budget is
        // fully restored. Idempotent.
        let dir = tempfile::tempdir().unwrap();
        let (sid, task);
        {
            let m =
                crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let s = session(&m);
            let ledger = DurableBudgetLedger::new(m.clone());
            task = seeded_task(&s, Some(1_000));
            sid = s.id;
            // Reserved but NEVER marked dispatched: the crash hit before
            // the marker (the runtime writes it immediately before the
            // provider transport call).
            let open = ledger
                .reserve(sid, task, OpId::new(50), 200, None)
                .await
                .unwrap();
            assert!(open.raw() > 0);
        }
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let ledger2 = DurableBudgetLedger::new(m2.clone());
        let view_before = ledger2.session_budget_view(sid, task);
        assert_eq!(view_before.open_reservations, 1);
        ledger2.recover_after_restart();
        let view = ledger2.session_budget_view(sid, task);
        assert_eq!(view.open_reservations, 0, "no OPEN rows survive recovery");
        assert_eq!(view.open_reserved_micro, 0);
        assert_eq!(
            view.uncertain_reservations, 0,
            "pre-marker rows are REFUNDED, never UNCERTAIN"
        );
        assert_eq!(view.spent_cost_micro, 0);
        let rows = ledger2.reservations_of(sid, task, 10).unwrap();
        assert_eq!(rows[0].status, "refunded");
        assert_eq!(rows[0].dispatched_ms, None);
        // The refunded prediction is free again: a 900-micro reservation
        // fits the 1000 cap.
        let again = ledger2
            .reserve(sid, task, OpId::new(51), 900, None)
            .await
            .unwrap();
        ledger2.refund(sid, again).await.unwrap();
        // Idempotent recovery: a second run finds nothing open.
        ledger2.recover_after_restart();
        assert_eq!(ledger2.session_budget_view(sid, task).open_reservations, 0);
    }

    #[tokio::test]
    async fn crash_after_the_dispatch_marker_is_uncertain_and_keeps_consuming_free() {
        // Crash AFTER the provider request was sent (marker written) but
        // before the local settle: the provider may have billed. Recovery
        // marks the reservation UNCERTAIN and it KEEPS consuming the
        // reserved amount — the next reservation correctly refuses.
        let dir = tempfile::tempdir().unwrap();
        let (sid, task);
        {
            let m =
                crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let s = session(&m);
            let ledger = DurableBudgetLedger::new(m.clone());
            task = seeded_task(&s, Some(1_000));
            sid = s.id;
            let open = ledger
                .reserve(sid, task, OpId::new(60), 800, None)
                .await
                .unwrap();
            ledger.mark_dispatched(sid, open).await.unwrap();
            // Crash: the reservation never settled.
        }
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let ledger2 = DurableBudgetLedger::new(m2.clone());
        ledger2.recover_after_restart();
        let view = ledger2.session_budget_view(sid, task);
        assert_eq!(view.open_reservations, 0);
        assert_eq!(view.uncertain_reservations, 1);
        assert_eq!(
            view.uncertain_reserved_micro, 800,
            "the may-have-billed prediction keeps consuming free"
        );
        assert_eq!(view.spent_cost_micro, 0);
        // Free = 1000 - 800 = 200: a 201-micro reserve refuses.
        let err = ledger2
            .reserve(sid, task, OpId::new(61), 201, None)
            .await
            .unwrap_err();
        assert_eq!(
            err,
            BudgetError::BudgetExceeded {
                free: 200,
                predicted: 201
            }
        );
        // The row carries the durable marker and its snapshot verbatim.
        let rows = ledger2.reservations_of(sid, task, 10).unwrap();
        assert_eq!(rows[0].status, "uncertain");
        assert!(rows[0].dispatched_ms.is_some());
    }

    #[tokio::test]
    async fn reconcile_settles_an_uncertain_reservation_from_the_ops_completed_call() {
        // (iii) A later settle for the same op id settles the crashed
        // attempt: once the op's durable provider_call row completes, the
        // reconcile closes the UNCERTAIN reservation FROM that row — tokens
        // at the reservation's own frozen snapshot.
        let dir = tempfile::tempdir().unwrap();
        let (sid, task, op);
        {
            let m =
                crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let s = session(&m);
            let ledger = DurableBudgetLedger::new(m.clone());
            task = seeded_task(&s, Some(50_000));
            sid = s.id;
            op = OpId::new(70);
            let r = ledger
                .reserve(sid, task, op, 10_000, Some(known_snapshot()))
                .await
                .unwrap();
            ledger.mark_dispatched(sid, r).await.unwrap();
            // Crash mid-flight: the reservation never settled.
        }
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let ledger2 = DurableBudgetLedger::new(m2.clone());
        ledger2.recover_after_restart();
        let view = ledger2.session_budget_view(sid, task);
        assert_eq!(view.uncertain_reservations, 1);
        // Reconcile before anything completed: the row has no completed
        // provider-call row to settle FROM, so it stays UNCERTAIN for the
        // task-completion finalize (the report counts only rows it examined).
        let report = ledger2.reconcile_uncertain(sid, task).await.unwrap();
        assert_eq!(report, faktor_store::CostReconcileReport::default());
        assert_eq!(
            ledger2
                .session_budget_view(sid, task)
                .uncertain_reservations,
            1,
            "nothing to reconcile: the row is still UNCERTAIN and consuming"
        );
        assert_eq!(ledger2.session_budget_view(sid, task).spent_cost_micro, 0);
        // The resumed op completes: its durable provider_call row records
        // 100k in + 2k out (a later settle for the same op id).
        let s2 = m2.get_session(sid).unwrap().unwrap();
        s2.settle_usage(op, "p", "m", "completed", Some(100_000), Some(2_000), None)
            .await
            .unwrap();
        let report = ledger2.reconcile_uncertain(sid, task).await.unwrap();
        assert_eq!(report.settled, 1);
        assert_eq!(
            report.charged_micro, 1_620_000,
            "100k @15 + 2k @60 == 1_620_000 micro at the frozen snapshot"
        );
        let view = ledger2.session_budget_view(sid, task);
        assert_eq!(view.spent_cost_micro, 1_620_000);
        assert_eq!(view.uncertain_reservations, 0);
        let rows = ledger2.reservations_of(sid, task, 10).unwrap();
        assert_eq!(rows[0].status, "settled");
        assert_eq!(rows[0].provider_cost_micro, Some(1_620_000));
        // Idempotent: a second pass settles nothing more.
        let report = ledger2.reconcile_uncertain(sid, task).await.unwrap();
        assert_eq!(report, faktor_store::CostReconcileReport::default());
        assert_eq!(
            ledger2.session_budget_view(sid, task).spent_cost_micro,
            1_620_000
        );
    }

    #[tokio::test]
    async fn reconcile_without_a_completed_call_leaves_the_row_for_the_finalize() {
        // A crashed dispatch whose op NEVER completed (no provider_call
        // row, no later settle) stays UNCERTAIN consuming free until the
        // task-completion finalize charges the reserved estimate.
        let dir = tempfile::tempdir().unwrap();
        let (sid, task);
        {
            let m =
                crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let s = session(&m);
            let ledger = DurableBudgetLedger::new(m.clone());
            task = seeded_task(&s, Some(10_000));
            sid = s.id;
            let r = ledger
                .reserve(sid, task, OpId::new(80), 7_500, Some(known_snapshot()))
                .await
                .unwrap();
            ledger.mark_dispatched(sid, r).await.unwrap();
            let _ = ledger
                .reserve(sid, task, OpId::new(81), 1_000, None)
                .await
                .unwrap();
            // Crash: both reservations never settled.
        }
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let ledger2 = DurableBudgetLedger::new(m2.clone());
        ledger2.recover_after_restart();
        // Reconcile finds no completed provider_call row to settle FROM:
        // the report is empty (rows without a completed row are never
        // examined) and the row stays UNCERTAIN, still consuming free. (The
        // unmarked second reservation was REFUNDED by recovery — dispatch
        // never provably began.)
        let report = ledger2.reconcile_uncertain(sid, task).await.unwrap();
        assert_eq!(report, faktor_store::CostReconcileReport::default());
        let view = ledger2.session_budget_view(sid, task);
        assert_eq!(view.uncertain_reservations, 1);
        assert_eq!(view.uncertain_reserved_micro, 7_500);
        assert_eq!(view.spent_cost_micro, 0);
    }

    #[tokio::test]
    async fn completion_finalize_charges_the_estimate_exactly_once() {
        // (iv) Task completion: outstanding UNCERTAIN rows settle
        // conservatively AT their reserved estimates — once, idempotently.
        let dir = tempfile::tempdir().unwrap();
        let (sid, task);
        {
            let m =
                crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let s = session(&m);
            let ledger = DurableBudgetLedger::new(m.clone());
            task = seeded_task(&s, Some(50_000));
            sid = s.id;
            // A settled call that DID run carries its own row.
            let settled = ledger
                .reserve(sid, task, OpId::new(90), 500, Some(known_snapshot()))
                .await
                .unwrap();
            ledger.mark_dispatched(sid, settled).await.unwrap();
            ledger
                .settle_usage(sid, settled, 0, 0, 0, 0, Some(120), None)
                .await
                .unwrap();
            // Two dispatched-but-never-settled attempts of crashed runs.
            for (i, predicted) in [(91, 2_000u64), (92, 3_000u64)] {
                let r = ledger
                    .reserve(sid, task, OpId::new(i), predicted, None)
                    .await
                    .unwrap();
                ledger.mark_dispatched(sid, r).await.unwrap();
            }
            // Crash: both outstanding reservations never settled.
        }
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let ledger2 = DurableBudgetLedger::new(m2.clone());
        ledger2.recover_after_restart();
        let view = ledger2.session_budget_view(sid, task);
        assert_eq!(view.spent_cost_micro, 120);
        assert_eq!(view.uncertain_reservations, 2);
        assert_eq!(view.uncertain_reserved_micro, 5_000);
        // Task completion: the finalize charges each at its estimate.
        let report = ledger2.finalize_uncertain(sid, task).await.unwrap();
        assert_eq!(report.settled, 2);
        assert_eq!(report.charged_micro, 5_000);
        let view = ledger2.session_budget_view(sid, task);
        assert_eq!(
            view.spent_cost_micro, 5_120,
            "120 settled + 2_000 + 3_000 finalized at the estimates"
        );
        assert_eq!(view.uncertain_reservations, 0);
        assert_eq!(view.uncertain_reserved_micro, 0);
        let rows = ledger2.reservations_of(sid, task, 10).unwrap();
        for row in &rows {
            match row.op_id.raw() {
                91 | 92 => {
                    assert_eq!(row.status, "settled");
                    assert_eq!(row.provider_cost_micro, Some(row.predicted_micro));
                    assert_eq!(row.provider_reported_micro, None);
                }
                _ => {}
            }
        }
        // Idempotent: a second finalize finds nothing UNCERTAIN and never
        // double-charges.
        let report = ledger2.finalize_uncertain(sid, task).await.unwrap();
        assert_eq!(report.settled, 0);
        assert_eq!(report.charged_micro, 0);
        assert_eq!(
            ledger2.session_budget_view(sid, task).spent_cost_micro,
            5_120
        );
    }

    #[tokio::test]
    async fn dispatch_marker_is_written_before_the_provider_call_and_second_mark_is_idempotent() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(1_000));
        let r = ledger
            .reserve(s.id, task, OpId::new(95), 10, None)
            .await
            .unwrap();
        // A second mark of the same still-open reservation is idempotent.
        ledger.mark_dispatched(s.id, r).await.unwrap();
        ledger.mark_dispatched(s.id, r).await.unwrap();
        let rows = ledger.reservations_of(s.id, task, 10).unwrap();
        assert_eq!(rows[0].status, "open");
        assert!(rows[0].dispatched_ms.is_some());
        // A marked reservation refuses settle-after-refund etc. exactly
        // once: settle closes it normally.
        ledger
            .settle_usage(s.id, r, 0, 0, 0, 0, Some(5), None)
            .await
            .unwrap();
        assert!(matches!(
            ledger.mark_dispatched(s.id, r).await.unwrap_err(),
            BudgetError::NotOpen { .. }
        ));
    }

    #[tokio::test]
    async fn hostile_reservation_ids_and_oversized_route_json_are_typed() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(1_000));
        assert!(matches!(
            ledger
                .settle_usage(s.id, ReservationId::new(99_999), 1, 0, 0, 0, None, None)
                .await
                .unwrap_err(),
            BudgetError::UnknownReservation(_)
        ));
        assert!(matches!(
            ledger
                .refund(s.id, ReservationId::new(99_999))
                .await
                .unwrap_err(),
            BudgetError::UnknownReservation(_)
        ));
        assert!(matches!(
            ledger
                .mark_dispatched(s.id, ReservationId::new(99_999))
                .await
                .unwrap_err(),
            BudgetError::UnknownReservation(_)
        ));
        let big = "x".repeat(MAX_ROUTE_DECISION_JSON_BYTES + 1);
        let r = ledger
            .reserve(s.id, task, OpId::new(60), 5, None)
            .await
            .unwrap();
        assert!(matches!(
            ledger
                .settle_usage(s.id, r, 1, 0, 0, 0, None, Some(big))
                .await
                .unwrap_err(),
            BudgetError::Malformed(_)
        ));
        ledger.refund(s.id, r).await.unwrap();
        // Missing task row: typed, and reserve refuses before any write.
        let err = ledger
            .reserve(s.id, TaskId::new(777), OpId::new(61), 1, None)
            .await
            .unwrap_err();
        assert!(matches!(err, BudgetError::MissingTask { .. }));
    }

    #[test]
    fn noop_budget_never_refuses_and_views_are_unlimited() {
        let b = NoopBudget;
        let v = b.session_budget_view(SessionId::new(1), TaskId::new(1));
        assert_eq!(v.max_cost_micro, None);
        assert_eq!(v.free(), u64::MAX);
        b.recover_after_restart();
    }
}
