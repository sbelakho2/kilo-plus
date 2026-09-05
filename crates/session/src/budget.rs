//! The durable monetary budget ledger (P0-6/12): the single authority over
//! how much a task's MODEL CALLS may cost.
//!
//! The agent runtime used to keep a private in-memory spend map keyed by
//! session (`AgentDeps.budget_micro` + a `spent` HashMap): a second budget
//! authority that vanished on restart and never recorded WHY money moved.
//! This module replaces it with a durable ledger over the session store
//! (schema v15): one `cost_reservation` row per paid model call attempt,
//! settled against the task row's monetary columns in the same transaction.
//!
//! # Invariants (locked by adversarial tests in this file)
//!
//! - `reserve` is atomic: spent + predicted > cap refuses with a typed
//!   `BudgetError::BudgetExceeded` and writes NOTHING (no row, no spend).
//!   `max_cost_micro` NULL/0 = unlimited.
//! - `settle` closes an OPEN reservation exactly once (OPEN -> SETTLED) and
//!   adds the CHOSEN actual to `task.spent_cost_micro` in ONE transaction.
//!   The chosen actual is the provider-reported cost when the usage frame
//!   carried one, else the locally calculated actual; BOTH are recorded on
//!   the row (`provider_cost_micro` = local, `provider_reported_micro` =
//!   reported) next to the routing decision's JSON. A settlement that would
//!   push spent past the cap is recorded honestly — the money WAS spent;
//!   the NEXT reservation is what refuses.
//! - A second settle / a settle after refund is a typed `NotOpen` error
//!   (exactly-once semantics per reservation).
//! - `refund` releases an OPEN reservation without spending (exactly-once).
//! - Crash recovery ([`DurableBudgetLedger::recover_after_restart`]) marks
//!   every surviving OPEN reservation ABANDONED and never counts it as
//!   spent: an op that never settled never spent its prediction (ops that
//!   DID run carry settled rows). Idempotent.
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
use tokio::task::JoinError;

use crate::manager::SessionManager;
use crate::SessionError;

/// Bound on the routing decision JSON stored on a reservation row
/// (bounded everything: an audit string is never stored unbounded).
pub const MAX_ROUTE_DECISION_JSON_BYTES: usize = 8192;

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
            BudgetError::UnknownReservation(_) | BudgetError::NotOpen { .. } => {
                SessionError::Conflict(e.to_string())
            }
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
    /// Sum of the predicted micro of every OPEN reservation.
    pub open_reserved_micro: u64,
    /// Count of OPEN reservations.
    pub open_reservations: usize,
    /// Count of SETTLED reservations.
    pub settled_count: usize,
}

impl BudgetView {
    /// Free balance: cap - spent - in-flight predictions (saturating; the
    /// ledger refuses reservations that would cross it).
    pub fn free(&self) -> u64 {
        match self.max_cost_micro {
            None => u64::MAX,
            Some(cap) => cap
                .saturating_sub(self.spent_cost_micro)
                .saturating_sub(self.open_reserved_micro),
        }
    }
}

pub(crate) type BoxFut<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The budget authority the agent runtime consumes before every paid model
/// call. Async operations carry explicit futures (no hidden spawns); every
/// implementation is `Send + Sync` and must make its own durability story.
pub trait BudgetAuthority: Send + Sync {
    /// Reserve `predicted_micro` of the task's budget for one operation.
    /// Fails with typed `BudgetError::BudgetExceeded` when
    /// spent + predicted > cap (`None`/0 cap = unlimited); a refusal writes
    /// NOTHING.
    fn reserve(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        op_id: OpId,
        predicted_micro: u64,
    ) -> BoxFut<'_, Result<ReservationId, BudgetError>>;

    /// Settle one reservation at the chosen actual cost: the provider-
    /// reported cost when `provider_reported_micro` is `Some`, else
    /// `actual_micro` (the locally calculated cost). Both amounts and the
    /// routing decision's JSON are recorded on the row; the chosen actual is
    /// folded into the task's spent total in the same transaction.
    fn settle(
        &self,
        session_id: SessionId,
        reservation: ReservationId,
        actual_micro: u64,
        provider_reported_micro: Option<u64>,
        route_decision_json: Option<String>,
    ) -> BoxFut<'_, Result<(), BudgetError>>;

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

    /// Resolve reservations a crashed process left OPEN (durable impl: they
    /// become ABANDONED and never count as spent). Idempotent.
    fn recover_after_restart(&self);
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

    /// Resolve every reservation a crashed process left OPEN (they become
    /// ABANDONED and never count as spent). Inherent alias of the
    /// [`BudgetAuthority`] method so daemon boot code can call it without
    /// importing the trait. Idempotent.
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
    ) -> Result<ReservationId, BudgetError> {
        let store = self.store();
        let created_ms = self.now_ms();
        tokio::task::spawn_blocking(move || {
            store.cost_reserve(session_id, task_id, op_id, predicted_micro, created_ms)
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

    async fn settle_inner(
        &self,
        reservation: ReservationId,
        actual_micro: u64,
        provider_reported_micro: Option<u64>,
        route_decision_json: Option<String>,
    ) -> Result<(), BudgetError> {
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
        // The chosen actual: the provider-reported cost is authoritative
        // when a usage frame reported one; the locally calculated cost
        // stands in otherwise. Both are recorded on the row.
        let chosen = provider_reported_micro.unwrap_or(actual_micro);
        tokio::task::spawn_blocking(move || {
            store.cost_settle(
                reservation.raw(),
                chosen,
                Some(actual_micro),
                provider_reported_micro,
                route_decision_json.as_deref(),
                settled_ms,
            )
        })
        .await
        .map_err(BudgetError::from)?
        .map_err(BudgetError::from)
        .and_then(|out| match out {
            faktor_store::CostReservationState::Applied => Ok(()),
            faktor_store::CostReservationState::Missing => {
                Err(BudgetError::UnknownReservation(reservation.raw()))
            }
            faktor_store::CostReservationState::NotOpen { current } => Err(BudgetError::NotOpen {
                reservation: reservation.raw(),
                status: current,
            }),
        })
    }

    async fn refund_inner(&self, reservation: ReservationId) -> Result<(), BudgetError> {
        let store = self.store();
        let settled_ms = self.now_ms();
        tokio::task::spawn_blocking(move || store.cost_refund(reservation.raw(), settled_ms))
            .await
            .map_err(BudgetError::from)?
            .map_err(BudgetError::from)
            .and_then(|out| match out {
                faktor_store::CostReservationState::Applied => Ok(()),
                faktor_store::CostReservationState::Missing => {
                    Err(BudgetError::UnknownReservation(reservation.raw()))
                }
                faktor_store::CostReservationState::NotOpen { current } => {
                    Err(BudgetError::NotOpen {
                        reservation: reservation.raw(),
                        status: current,
                    })
                }
            })
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
        let (open_micro, open_count, settled_count) = store
            .cost_reservations_of(session_id, task_id, i64::MAX)
            .ok()
            .map(|rows| {
                let open_micro: u64 = rows
                    .iter()
                    .filter(|r| r.status == "open")
                    .map(|r| r.predicted_micro)
                    .sum();
                let open_count = rows.iter().filter(|r| r.status == "open").count();
                let settled_count = rows.iter().filter(|r| r.status == "settled").count();
                (open_micro, open_count, settled_count)
            })
            .unwrap_or((0, 0, 0));
        BudgetView {
            max_cost_micro: max,
            spent_cost_micro: spent,
            open_reserved_micro: open_micro,
            open_reservations: open_count,
            settled_count,
        }
    }

    fn recover_inner(&self) {
        let store = self.store();
        let at = self.now_ms();
        if let Err(e) = store.cost_abandon_open_reservations(at) {
            tracing::warn!("budget ledger crash recovery failed: {e}");
        }
    }
}

impl BudgetAuthority for DurableBudgetLedger {
    fn reserve(
        &self,
        session_id: SessionId,
        task_id: TaskId,
        op_id: OpId,
        predicted_micro: u64,
    ) -> BoxFut<'_, Result<ReservationId, BudgetError>> {
        Box::pin(self.reserve_inner(session_id, task_id, op_id, predicted_micro))
    }

    fn settle(
        &self,
        _session_id: SessionId,
        reservation: ReservationId,
        actual_micro: u64,
        provider_reported_micro: Option<u64>,
        route_decision_json: Option<String>,
    ) -> BoxFut<'_, Result<(), BudgetError>> {
        Box::pin(self.settle_inner(
            reservation,
            actual_micro,
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
    ) -> BoxFut<'_, Result<ReservationId, BudgetError>> {
        Box::pin(async { Ok(ReservationId::NOOP) })
    }

    fn settle(
        &self,
        _session_id: SessionId,
        _reservation: ReservationId,
        _actual_micro: u64,
        _provider_reported_micro: Option<u64>,
        _route_decision_json: Option<String>,
    ) -> BoxFut<'_, Result<(), BudgetError>> {
        Box::pin(async { Ok(()) })
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
            settled_count: 0,
        }
    }

    fn recover_after_restart(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::tests::{session, test_manager};
    use faktor_core::state::TaskState;

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
            .reserve(s.id, task, OpId::new(1), 101)
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
            .reserve(s.id, task, OpId::new(1), u64::MAX / 2)
            .await
            .unwrap();
        assert!(granted.raw() > 0);
        ledger.refund(s.id, granted).await.unwrap();
    }

    #[tokio::test]
    async fn settle_updates_spent_exactly_once_and_refuses_a_second_settle() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(1_000));
        let r1 = ledger
            .reserve(s.id, task, OpId::new(10), 400)
            .await
            .unwrap();
        let r2 = ledger
            .reserve(s.id, task, OpId::new(11), 400)
            .await
            .unwrap();
        ledger.settle(s.id, r1, 350, None, None).await.unwrap();
        let view = ledger.session_budget_view(s.id, task);
        assert_eq!(view.spent_cost_micro, 350);
        assert_eq!(view.open_reservations, 1);
        assert_eq!(view.open_reserved_micro, 400);
        // Double settle: typed NotOpen, nothing changes.
        let err = ledger.settle(s.id, r1, 1, None, None).await.unwrap_err();
        assert!(matches!(err, BudgetError::NotOpen { .. }));
        assert_eq!(ledger.session_budget_view(s.id, task).spent_cost_micro, 350);
        // Reserve over the remaining cap is refused.
        let err = ledger
            .reserve(s.id, task, OpId::new(12), 300)
            .await
            .unwrap_err();
        assert_eq!(
            err,
            BudgetError::BudgetExceeded {
                free: 250,
                predicted: 300
            }
        );
        ledger.settle(s.id, r2, 400, None, None).await.unwrap();
        assert_eq!(ledger.session_budget_view(s.id, task).spent_cost_micro, 750);
    }

    #[tokio::test]
    async fn provider_reported_cost_wins_and_both_columns_are_recorded() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(5_000));
        let r = ledger
            .reserve(s.id, task, OpId::new(20), 100)
            .await
            .unwrap();
        ledger
            .settle(
                s.id,
                r,
                40, // locally calculated
                Some(9_999),
                Some(r#"{"provider":"p","model":"m"}"#.into()),
            )
            .await
            .unwrap();
        let rows = ledger.reservations_of(s.id, task, 10).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status, "settled");
        assert_eq!(rows[0].provider_cost_micro, Some(40));
        assert_eq!(rows[0].provider_reported_micro, Some(9_999));
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
    async fn refund_is_exactly_once_and_releases_without_spending() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(100));
        let r = ledger.reserve(s.id, task, OpId::new(30), 60).await.unwrap();
        ledger.refund(s.id, r).await.unwrap();
        assert_eq!(ledger.session_budget_view(s.id, task).spent_cost_micro, 0);
        // A refunded reservation refuses settle AND a second refund.
        assert!(matches!(
            ledger.settle(s.id, r, 1, None, None).await.unwrap_err(),
            BudgetError::NotOpen { .. }
        ));
        assert!(matches!(
            ledger.refund(s.id, r).await.unwrap_err(),
            BudgetError::NotOpen { .. }
        ));
        // Space is available again.
        let again = ledger.reserve(s.id, task, OpId::new(31), 90).await.unwrap();
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
        let r = ledger.reserve(s.id, task, OpId::new(40), 60).await.unwrap();
        ledger.settle(s.id, r, 60, Some(150), None).await.unwrap();
        let view = ledger.session_budget_view(s.id, task);
        assert_eq!(
            view.spent_cost_micro, 150,
            "overspend is recorded, not hidden"
        );
        let err = ledger
            .reserve(s.id, task, OpId::new(41), 1)
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
    async fn reserve_settle_survive_restart_and_open_rows_abandon_without_spend() {
        // Crash mid-flight: one OPEN reservation + one settled row. After a
        // simulated kill (fresh open over the same store) recovery abandons
        // the OPEN reservation; the settled row and the spent total are
        // intact; ids stay unique across the restart.
        let dir = tempfile::tempdir().unwrap();
        let (sid, task, settled_id);
        {
            let m =
                crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                    .unwrap();
            let s = session(&m);
            let ledger = DurableBudgetLedger::new(m.clone());
            task = seeded_task(&s, Some(1_000));
            sid = s.id;
            let open = ledger.reserve(sid, task, OpId::new(50), 200).await.unwrap();
            settled_id = ledger.reserve(sid, task, OpId::new(51), 300).await.unwrap();
            ledger
                .settle(sid, settled_id, 250, None, None)
                .await
                .unwrap();
            assert!(open.raw() > 0 && settled_id.raw() > 0);
            // The OPEN reservation simulates the crash (never settled).
        }
        let m2 =
            crate::SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
                .unwrap();
        let ledger2 = DurableBudgetLedger::new(m2.clone());
        let view_before = ledger2.session_budget_view(sid, task);
        assert_eq!(view_before.spent_cost_micro, 250, "settled spend survives");
        assert_eq!(view_before.open_reservations, 1);
        ledger2.recover_after_restart();
        let view = ledger2.session_budget_view(sid, task);
        assert_eq!(
            view.spent_cost_micro, 250,
            "abandoned rows never count as spent"
        );
        assert_eq!(view.open_reservations, 0, "no OPEN rows survive recovery");
        assert_eq!(view.open_reserved_micro, 0);
        let rows = ledger2.reservations_of(sid, task, 10).unwrap();
        let open_status = rows.iter().find(|r| r.op_id == OpId::new(50)).unwrap();
        assert_eq!(open_status.status, "abandoned");
        let settled = rows.iter().find(|r| r.op_id == OpId::new(51)).unwrap();
        assert_eq!(settled.status, "settled");
        assert_eq!(settled.provider_cost_micro, Some(250));
        // Ids minted after the restart are still unique and increasing.
        let after = ledger2.reserve(sid, task, OpId::new(52), 1).await.unwrap();
        assert!(after.raw() > settled_id.raw());
        assert_eq!(ledger2.reservations_of(sid, task, 100).unwrap().len(), 3);
        // Idempotent recovery: a second run finds nothing.
        ledger2.recover_after_restart();
        assert_eq!(ledger2.session_budget_view(sid, task).open_reservations, 0);
    }

    #[tokio::test]
    async fn hostile_reservation_ids_and_oversized_route_json_are_typed() {
        let (_d, m, ledger) = fresh_ledger();
        let s = session(&m);
        let task = seeded_task(&s, Some(1_000));
        assert!(matches!(
            ledger
                .settle(s.id, ReservationId::new(99_999), 1, None, None)
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
        let big = "x".repeat(MAX_ROUTE_DECISION_JSON_BYTES + 1);
        let r = ledger.reserve(s.id, task, OpId::new(60), 5).await.unwrap();
        assert!(matches!(
            ledger
                .settle(s.id, r, 1, None, Some(big))
                .await
                .unwrap_err(),
            BudgetError::Malformed(_)
        ));
        ledger.refund(s.id, r).await.unwrap();
        // Missing task row: typed, and reserve refuses before any write.
        let err = ledger
            .reserve(s.id, TaskId::new(777), OpId::new(61), 1)
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
