//! Reservation state-machine model check (audit item 100).
//!
//! The durable `cost_reservation` machine — `Reserved` / `Dispatched` /
//! `Uncertain` / `Settled` / `Refunded` across reserve / dispatch / fail /
//! refund / settle / recover / reconcile / finalize — is driven over seeded
//! LCG traces and compared against the REAL store-backed
//! [`DurableBudgetLedger`] after every operation. The audit's economic
//! invariants are asserted on every step:
//!
//! - available budget NEVER increases after a reservation became chargeable
//!   (`Dispatched`/potentially chargeable) unless an authoritative
//!   settlement/reconciliation closed it at a recorded actual; a refund may
//!   only release a pre-dispatch (`Reserved`) prediction;
//! - `settled + held <= cap` before every admission — unless an
//!   already-dispatched actual exceeded its estimate, in which case spend is
//!   recorded honestly and FUTURE requests refuse (free saturates to zero);
//! - every illegal transition is a typed refusal that writes nothing;
//! - unknown price is never treated as free (a price-less settle under a
//!   hard cap is a typed `UnknownPrice`; without a cap it records a
//!   documented Unknown spend with nothing folded).
//!
//! The full 5000-seed durable cross-check is `#[ignore]`-gated (`[soak]`);
//! the CI smoke runs a deterministic subset. The pure-model property driver
//! runs the full 5000 traces per seed in normal mode.

#![cfg_attr(
    not(test),
    allow(dead_code, unused_imports, unused_variables, unused_mut)
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use faktor_core::id::{OpId, SessionId, TaskId};
use faktor_core::model::{MicroUsdPerMillionTokens, PriceQuote, PricingSnapshot};
use faktor_core::op::ModelCallAttempt;
use faktor_core::state::TaskState;
use faktor_session::{
    BudgetAuthority, BudgetError, DurableBudgetLedger, ReservationId, SessionHandle,
    SessionManager, Task,
};

/// Traces per seed of the documented model check.
pub const TRACES_PER_SEED: u64 = 5_000;

/// Fixed seeds (platform-independent constants).
pub const SEEDS: [u64; 2] = [0xACC7_0000_0000_0001, 0xACC7_0000_0000_0002];

/// One reservation's logical state — the audit's vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RState {
    Reserved,
    Dispatched,
    Uncertain,
    Settled,
    Refunded,
}

impl RState {
    /// The schema status string (mirrors [`faktor_session::ReservationState`]).
    pub fn db(self) -> &'static str {
        match self {
            RState::Reserved => "reserved",
            RState::Dispatched => "dispatched",
            RState::Uncertain => "uncertain",
            RState::Settled => "settled",
            RState::Refunded => "refunded",
        }
    }

    /// A row in this state still holds its reserved budget.
    pub fn holds(self) -> bool {
        matches!(
            self,
            RState::Reserved | RState::Dispatched | RState::Uncertain
        )
    }
}

/// One machine row.
#[derive(Debug, Clone)]
pub struct Row {
    pub state: RState,
    pub predicted: u64,
    pub snapshot: Option<PricingSnapshot>,
    /// The durable dispatch marker (`false` = dispatch never provably began).
    pub marker: bool,
    /// A completed authoritative provider call of this attempt, when durable.
    pub known_usage: Option<(u64, u64)>,
    /// The folded amount when settled (`None` = unknown spend or unsettled).
    pub charged: Option<u64>,
}

/// A typed model-level refusal (ids are model-local; see [`CanonErr`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedErr {
    BudgetExceeded { free: u64, predicted: u64 },
    UnknownReservation,
    NotOpen { status: String },
    CannotRefundDispatched,
    UnknownPrice,
}

/// Diagnostic/refusal vocabulary comparable across the model and the real
/// ledger (raw reservation ids differ by construction, so they are absent).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonErr {
    BudgetExceeded { free: u64, predicted: u64 },
    UnknownReservation,
    NotOpen { status: String },
    CannotRefundDispatched,
    UnknownPrice,
}

impl TypedErr {
    pub fn canon(&self) -> CanonErr {
        match self {
            TypedErr::BudgetExceeded { free, predicted } => CanonErr::BudgetExceeded {
                free: *free,
                predicted: *predicted,
            },
            TypedErr::UnknownReservation => CanonErr::UnknownReservation,
            TypedErr::NotOpen { status } => CanonErr::NotOpen {
                status: status.clone(),
            },
            TypedErr::CannotRefundDispatched => CanonErr::CannotRefundDispatched,
            TypedErr::UnknownPrice => CanonErr::UnknownPrice,
        }
    }
}

impl CanonErr {
    pub fn of_real(e: &BudgetError) -> CanonErr {
        match e {
            BudgetError::BudgetExceeded { free, predicted } => CanonErr::BudgetExceeded {
                free: *free,
                predicted: *predicted,
            },
            BudgetError::UnknownReservation(_) => CanonErr::UnknownReservation,
            BudgetError::NotOpen { status, .. } => CanonErr::NotOpen {
                status: status.clone(),
            },
            BudgetError::CannotRefundDispatched { .. } => CanonErr::CannotRefundDispatched,
            BudgetError::UnknownPrice { .. } => CanonErr::UnknownPrice,
            other => panic!("unexpected ledger refusal outside the reservation machine: {other:?}"),
        }
    }
}

/// The pure machine mirroring the durable ledger's transition rules.
#[derive(Debug, Clone)]
pub struct Machine {
    pub cap: Option<u64>,
    pub spent: u64,
    pub next_id: u64,
    /// At least one settlement recorded an actual above its prediction.
    pub overshoot: bool,
    pub rows: BTreeMap<u64, Row>,
}

impl Machine {
    pub fn new(cap: Option<u64>) -> Self {
        Self {
            cap,
            spent: 0,
            next_id: 0,
            overshoot: false,
            rows: BTreeMap::new(),
        }
    }

    /// The admission cap (`None`/0 = unlimited, exactly the store's rule).
    pub fn active_cap(&self) -> Option<u64> {
        self.cap.filter(|c| *c > 0)
    }

    pub fn held(&self) -> u64 {
        self.rows
            .values()
            .filter(|r| r.state.holds())
            .fold(0u64, |acc, r| acc.saturating_add(r.predicted))
    }

    pub fn free(&self) -> u64 {
        match self.cap {
            None => u64::MAX,
            Some(cap) => cap.saturating_sub(self.spent).saturating_sub(self.held()),
        }
    }

    pub fn reserve(
        &mut self,
        predicted: u64,
        snapshot: Option<PricingSnapshot>,
        known_usage: Option<(u64, u64)>,
    ) -> Result<u64, TypedErr> {
        let free = self.free();
        if self.active_cap().is_some() && predicted > free {
            return Err(TypedErr::BudgetExceeded { free, predicted });
        }
        self.next_id += 1;
        let id = self.next_id;
        self.rows.insert(
            id,
            Row {
                state: RState::Reserved,
                predicted,
                snapshot,
                marker: false,
                known_usage,
                charged: None,
            },
        );
        Ok(id)
    }

    pub fn dispatch(&mut self, id: u64) -> Result<(), TypedErr> {
        let Some(row) = self.rows.get_mut(&id) else {
            return Err(TypedErr::UnknownReservation);
        };
        match row.state {
            RState::Reserved | RState::Dispatched => {
                row.state = RState::Dispatched;
                row.marker = true;
                Ok(())
            }
            other => Err(TypedErr::NotOpen {
                status: other.db().into(),
            }),
        }
    }

    pub fn fail(&mut self, id: u64) -> Result<(), TypedErr> {
        let Some(row) = self.rows.get_mut(&id) else {
            return Err(TypedErr::UnknownReservation);
        };
        match row.state {
            RState::Dispatched => {
                row.state = RState::Uncertain;
                Ok(())
            }
            other => Err(TypedErr::NotOpen {
                status: other.db().into(),
            }),
        }
    }

    pub fn refund(&mut self, id: u64) -> Result<(), TypedErr> {
        let Some(row) = self.rows.get_mut(&id) else {
            return Err(TypedErr::UnknownReservation);
        };
        // The store's guarded UPDATE refuses every row that left the
        // refundable pre-dispatch state; a row carrying the dispatch marker
        // (dispatched, uncertain, or settled after dispatch) surfaces
        // `CannotRefundDispatched`.
        match row.state {
            RState::Reserved if !row.marker => {
                row.state = RState::Refunded;
                Ok(())
            }
            _ if row.marker => Err(TypedErr::CannotRefundDispatched),
            other => Err(TypedErr::NotOpen {
                status: other.db().into(),
            }),
        }
    }

    /// Settle at the provider-reported actual or the tokens x the stored
    /// snapshot — exactly the store's chosen-actual rule.
    pub fn settle_usage(
        &mut self,
        id: u64,
        tokens: (u64, u64),
        reported: Option<u64>,
    ) -> Result<Option<u64>, TypedErr> {
        let Some(row) = self.rows.get(&id) else {
            return Err(TypedErr::UnknownReservation);
        };
        if !matches!(row.state, RState::Reserved | RState::Dispatched) {
            return Err(TypedErr::NotOpen {
                status: row.state.db().into(),
            });
        }
        let local = row
            .snapshot
            .as_ref()
            .and_then(|s| s.settle_cost(tokens.0, 0, 0, tokens.1));
        match reported.or(local) {
            Some(actual) => {
                let row = self.rows.get_mut(&id).expect("checked above");
                if actual > row.predicted {
                    self.overshoot = true;
                }
                row.charged = Some(actual);
                row.state = RState::Settled;
                self.spent = self.spent.saturating_add(actual);
                Ok(Some(actual))
            }
            None => {
                if self.active_cap().is_some() {
                    Err(TypedErr::UnknownPrice)
                } else {
                    let row = self.rows.get_mut(&id).expect("checked above");
                    row.charged = None;
                    row.state = RState::Settled;
                    Ok(None)
                }
            }
        }
    }

    /// Crash recovery: reserved -> refunded, dispatched -> uncertain.
    pub fn recover(&mut self) {
        for row in self.rows.values_mut() {
            match row.state {
                RState::Reserved => row.state = RState::Refunded,
                RState::Dispatched => row.state = RState::Uncertain,
                _ => {}
            }
        }
    }

    /// Settle every uncertain row with a completed call at its recorded
    /// usage; unpriced rows under a cap stay uncertain, unpriced without a
    /// cap close as documented Unknown spends.
    pub fn reconcile(&mut self) -> ReconcileOutcome {
        let mut out = ReconcileOutcome::default();
        let ids: Vec<u64> = self
            .rows
            .iter()
            .filter(|(_, r)| r.state == RState::Uncertain && r.known_usage.is_some())
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let local = {
                let row = &self.rows[&id];
                let tokens = row.known_usage.expect("filtered");
                row.snapshot
                    .as_ref()
                    .and_then(|s| s.settle_cost(tokens.0, 0, 0, tokens.1))
            };
            match local {
                Some(actual) => {
                    let row = self.rows.get_mut(&id).expect("tracked");
                    if actual > row.predicted {
                        self.overshoot = true;
                    }
                    row.charged = Some(actual);
                    row.state = RState::Settled;
                    self.spent = self.spent.saturating_add(actual);
                    out.settled += 1;
                    out.charged = out.charged.saturating_add(actual);
                }
                None => {
                    if self.active_cap().is_some() {
                        out.left_uncertain += 1;
                    } else {
                        let row = self.rows.get_mut(&id).expect("tracked");
                        row.charged = None;
                        row.state = RState::Settled;
                        out.closed_unknown += 1;
                    }
                }
            }
        }
        out
    }

    /// Conservatively settle every uncertain row at its reserved estimate.
    pub fn finalize(&mut self) -> FinalizeOutcome {
        let mut out = FinalizeOutcome::default();
        let ids: Vec<u64> = self
            .rows
            .iter()
            .filter(|(_, r)| r.state == RState::Uncertain)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let predicted = self.rows[&id].predicted;
            let row = self.rows.get_mut(&id).expect("tracked");
            row.charged = Some(predicted);
            row.state = RState::Settled;
            self.spent = self.spent.saturating_add(predicted);
            out.settled += 1;
            out.charged = out.charged.saturating_add(predicted);
        }
        out
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReconcileOutcome {
    pub settled: u64,
    pub closed_unknown: u64,
    pub left_uncertain: u64,
    pub charged: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FinalizeOutcome {
    pub settled: u64,
    pub charged: u64,
}

/// A target of a state transition: a tracked model id or an id guaranteed
/// absent from the real ledger.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Known(u64),
    Unknown(u64),
}

impl Target {
    fn model_id(self) -> u64 {
        match self {
            Target::Known(id) => id,
            Target::Unknown(k) => u64::MAX - k,
        }
    }

    /// Resolve the REAL ledger id: tracked model ids through the per-trace
    /// map (real ids are global AUTOINCREMENT), unknown ids to a value that
    /// cannot exist.
    fn real_id(self, id_map: &BTreeMap<u64, ReservationId>) -> ReservationId {
        match self {
            Target::Known(id) => *id_map
                .get(&id)
                .unwrap_or_else(|| panic!("untracked model reservation {id}")),
            Target::Unknown(k) => ReservationId::new(i64::MAX - k as i64),
        }
    }
}

/// One generated operation.
#[derive(Debug, Clone, Copy)]
pub enum Op {
    Reserve {
        predicted: u64,
        priced: bool,
        known_usage: Option<(u64, u64)>,
    },
    Dispatch {
        target: Target,
    },
    Fail {
        target: Target,
    },
    Refund {
        target: Target,
    },
    SettleReported {
        target: Target,
        actual: u64,
    },
    SettlePriced {
        target: Target,
        tokens: (u64, u64),
    },
    Recover,
    Reconcile,
    Finalize,
}

/// What applying one op to the pure machine produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    Granted(u64),
    Refused(TypedErr),
    Unit(Result<(), TypedErr>),
    Settle(Result<Option<u64>, TypedErr>),
    Recover,
    Reconcile(ReconcileOutcome),
    Finalize(FinalizeOutcome),
}

pub fn apply_model(model: &mut Machine, op: &Op) -> Effect {
    match op {
        Op::Reserve {
            predicted,
            priced,
            known_usage,
        } => {
            let snapshot = priced.then(|| snapshot_for(*known_usage));
            match model.reserve(*predicted, snapshot, *known_usage) {
                Ok(id) => Effect::Granted(id),
                Err(e) => Effect::Refused(e),
            }
        }
        Op::Dispatch { target } => Effect::Unit(model.dispatch(target.model_id())),
        Op::Fail { target } => Effect::Unit(model.fail(target.model_id())),
        Op::Refund { target } => Effect::Unit(model.refund(target.model_id())),
        Op::SettleReported { target, actual } => {
            Effect::Settle(model.settle_usage(target.model_id(), (0, 0), Some(*actual)))
        }
        Op::SettlePriced { target, tokens } => {
            Effect::Settle(model.settle_usage(target.model_id(), *tokens, None))
        }
        Op::Recover => {
            model.recover();
            Effect::Recover
        }
        Op::Reconcile => Effect::Reconcile(model.reconcile()),
        Op::Finalize => Effect::Finalize(model.finalize()),
    }
}

fn snapshot_for(_known: Option<(u64, u64)>) -> PricingSnapshot {
    PricingSnapshot::exact(
        PriceQuote {
            input: MicroUsdPerMillionTokens(10_000_000),
            output: MicroUsdPerMillionTokens(50_000_000),
            cache_read: MicroUsdPerMillionTokens(1_000_000),
            cache_write: MicroUsdPerMillionTokens(5_000_000),
        },
        1,
        "accounting-modelcheck".into(),
    )
}

/// Deterministic seeded LCG (numerical-recipe constants only: identical on
/// every platform).
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1) | 1)
    }
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32 as u64
    }

    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        self.next() % n
    }

    pub fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

fn pick_target(lcg: &mut Lcg, model: &Machine) -> Target {
    let ids: Vec<u64> = model.rows.keys().copied().collect();
    if ids.is_empty() || lcg.chance(12) {
        Target::Unknown(lcg.below(8))
    } else {
        Target::Known(ids[lcg.below(ids.len() as u64) as usize])
    }
}

pub fn gen_op(lcg: &mut Lcg, model: &Machine, ceiling_hint: u64) -> Op {
    match lcg.below(100) {
        0..=44 => {
            let predicted = 1 + lcg.below(ceiling_hint.max(2));
            let known_usage = lcg.chance(45).then(|| (lcg.below(4_000), lcg.below(1_000)));
            Op::Reserve {
                predicted,
                priced: lcg.chance(60),
                known_usage,
            }
        }
        45..=56 => Op::Dispatch {
            target: pick_target(lcg, model),
        },
        57..=64 => Op::Fail {
            target: pick_target(lcg, model),
        },
        65..=72 => Op::Refund {
            target: pick_target(lcg, model),
        },
        73..=81 => Op::SettleReported {
            target: pick_target(lcg, model),
            actual: lcg.below(60_000),
        },
        82..=88 => Op::SettlePriced {
            target: pick_target(lcg, model),
            tokens: (lcg.below(4_000), lcg.below(1_000)),
        },
        89..=92 => Op::Recover,
        93..=96 => Op::Reconcile,
        _ => Op::Finalize,
    }
}

/// The pre-op observable picture used by the invariant checks.
pub struct PreState {
    free: u64,
    spent: u64,
    rows: BTreeMap<u64, RState>,
}

pub fn pre(model: &Machine) -> PreState {
    PreState {
        free: model.free(),
        spent: model.spent,
        rows: model.rows.iter().map(|(id, r)| (*id, r.state)).collect(),
    }
}

/// The audit's economic invariants, checked after EVERY operation.
pub fn check_effect(model: &Machine, op: &Op, before: &PreState, effect: &Effect, ctx: &str) {
    // (1) Available budget never increases after a reservation became
    // chargeable unless an authoritative settlement/reconciliation recorded
    // its actual. A refund/recover may only release pre-dispatch rows.
    if model.free() > before.free {
        match op {
            Op::SettleReported { .. } | Op::SettlePriced { .. } | Op::Reconcile => {}
            Op::Refund { .. } | Op::Recover => {
                for (id, state) in &before.rows {
                    if matches!(state, RState::Dispatched | RState::Uncertain) {
                        // Recovery moves Dispatched -> Uncertain (still
                        // holding the same amount); nothing may release it.
                        assert!(
                            model.rows.get(id).map(|r| r.state.holds()).unwrap_or(false),
                            "{ctx}: a chargeable reservation was released without an \
                             authoritative actual"
                        );
                    }
                }
            }
            _ => panic!("{ctx}: free budget increased without an authoritative release"),
        }
    }
    // (2) Admission: granted iff the prediction fits the free balance; a
    // settled actual above the estimate makes every future positive request
    // refuse.
    if let Op::Reserve { predicted, .. } = op {
        match effect {
            Effect::Granted(_) => {
                assert!(
                    model.active_cap().is_none() || *predicted <= before.free,
                    "{ctx}: admitted a reservation above the free balance"
                );
            }
            Effect::Refused(e) => {
                assert!(
                    model.active_cap().is_some() && *predicted > before.free,
                    "{ctx}: refused a reservation that fits the free balance: {e:?}"
                );
            }
            _ => panic!("{ctx}: reserve produced a non-admission effect"),
        }
        if model.active_cap().is_some() && before.spent > model.active_cap().expect("checked") {
            assert!(
                model.free() == 0,
                "{ctx}: an overshooting actual must leave FUTURE requests refusing"
            );
        }
    }
    // (3) settled + held <= cap before/at every admission, unless an
    // already-dispatched actual exceeded its estimate (recorded honestly).
    if let Some(cap) = model.active_cap() {
        if !model.overshoot {
            assert!(
                model.spent.saturating_add(model.held()) <= cap,
                "{ctx}: settled + held exceeded the cap without an overshoot"
            );
        }
    }
}

/// Drive one trace against the pure machine (the full 5000-trace property
/// driver; no store, no async).
pub fn pure_trace(seed: u64, trace: u64, ceiling_hint: u64) {
    let mut lcg = Lcg::new(seed ^ trace.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let cap = if lcg.chance(75) {
        Some(ceiling_hint)
    } else {
        None
    };
    let mut model = Machine::new(cap);
    let steps = 24 + lcg.below(17);
    for step in 0..steps {
        let ctx = format!("pure seed={seed:#x} trace={trace} step={step}");
        let op = gen_op(&mut lcg, &model, ceiling_hint);
        let before = pre(&model);
        let effect = apply_model(&mut model, &op);
        check_effect(&model, &op, &before, &effect, &ctx);
        assert_eq!(
            model.held(),
            model
                .rows
                .values()
                .filter(|r| r.state.holds())
                .fold(0u64, |a, r| a.saturating_add(r.predicted)),
            "{ctx}: held sum"
        );
    }
}

pub fn pure_driver(seed: u64) {
    for trace in 0..TRACES_PER_SEED {
        let hint = 20_000 + (trace % 13) * 2_500;
        pure_trace(seed, trace, hint);
    }
}

// ---------------------------------------------------------------------------
// Durable cross-check: the same trace against the real store-backed ledger.
// ---------------------------------------------------------------------------

struct Fixture {
    _dir: tempfile::TempDir,
    manager: Arc<SessionManager>,
    session: SessionId,
    handle: SessionHandle,
    ledger: Arc<DurableBudgetLedger>,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("tempdir");
    let manager = SessionManager::open(dir.path().join("store"), dir.path().join("cas"), true)
        .expect("store opens");
    let ws = manager.create_workspace("/w").expect("workspace");
    let session = manager
        .create_session(ws, "accounting-modelcheck", "fake", "m")
        .expect("session")
        .id();
    let handle = manager.get_session(session).unwrap().unwrap();
    let ledger = DurableBudgetLedger::new(manager.clone());
    Fixture {
        _dir: dir,
        manager,
        session,
        handle,
        ledger,
    }
}

/// Create the per-trace task row (task id `trace + 1`, one task per trace so
/// every trace starts with spent = 0 and a global recovery can never touch a
/// prior trace's terminal rows).
fn task_for(fixture: &Fixture, trace: u64, cap: Option<u64>) -> TaskId {
    let task = TaskId::new(trace + 1);
    fixture
        .handle
        .create_task(Task {
            task_id: task,
            session_id: fixture.session,
            goal: format!("modelcheck trace {trace}"),
            acceptance_criteria: Vec::new(),
            plan: Vec::new(),
            budget: faktor_session::TaskBudget::default(),
            state: TaskState::Pending,
            created_ms: 1,
            updated_ms: 1,
        })
        .expect("task");
    fixture
        .ledger
        .set_task_max_cost(fixture.session, task, cap)
        .expect("cap");
    task
}

fn assert_real_state(
    fixture: &Fixture,
    task: TaskId,
    model: &Machine,
    raw_to_model: &BTreeMap<i64, u64>,
    ctx: &str,
) {
    let rows = fixture
        .ledger
        .reservations_of(fixture.session, task, i64::MAX)
        .expect("reservations read");
    let view = fixture.ledger.session_budget_view(fixture.session, task);
    assert_eq!(
        view.spent_cost_micro, model.spent,
        "{ctx}: spent divergence"
    );
    let held: u64 = rows
        .iter()
        .filter(|r| matches!(r.status.as_str(), "reserved" | "dispatched" | "uncertain"))
        .fold(0u64, |a, r| a.saturating_add(r.predicted_micro));
    assert_eq!(
        view.open_reserved_micro + view.uncertain_reserved_micro,
        held,
        "{ctx}: held divergence"
    );
    assert_eq!(held, model.held(), "{ctx}: model held divergence");
    assert_eq!(view.free(), model.free(), "{ctx}: free divergence");
    assert_eq!(rows.len(), model.rows.len(), "{ctx}: row count divergence");
    for row in &rows {
        let id = raw_to_model
            .get(&row.reservation_id)
            .unwrap_or_else(|| panic!("{ctx}: unknown real reservation {}", row.reservation_id));
        let mrow = &model.rows[id];
        assert_eq!(row.status, mrow.state.db(), "{ctx}: status divergence");
        assert_eq!(
            row.predicted_micro, mrow.predicted,
            "{ctx}: predicted divergence"
        );
        assert_eq!(
            row.settled_cost_micro, mrow.charged,
            "{ctx}: settled amount divergence"
        );
    }
}

#[allow(clippy::too_many_arguments)]
async fn apply_durable(
    fixture: &Fixture,
    task: TaskId,
    model: &mut Machine,
    id_map: &mut BTreeMap<u64, ReservationId>,
    raw_to_model: &mut BTreeMap<i64, u64>,
    attempts: &mut BTreeMap<u64, ModelCallAttempt>,
    op: &Op,
    trace: u64,
    ctx: &str,
) -> Effect {
    let effect = match op {
        Op::Reserve { .. } => {
            let next_model_id = model.next_id + 1;
            let base = trace
                .wrapping_mul(1_000_000)
                .wrapping_add(next_model_id.wrapping_mul(2));
            let attempt = ModelCallAttempt::new(OpId::new(base), OpId::new(base + 1), 0)
                .expect("distinct attempt ids");
            let Op::Reserve {
                predicted,
                priced,
                known_usage,
            } = *op
            else {
                unreachable!()
            };
            let snapshot = priced.then(|| snapshot_for(known_usage));
            let real = fixture
                .ledger
                .reserve_attempt(fixture.session, task, attempt, predicted, snapshot)
                .await;
            let effect = apply_model(model, op);
            match (&effect, &real) {
                (Effect::Granted(mid), Ok(rid)) => {
                    id_map.insert(*mid, *rid);
                    raw_to_model.insert(rid.raw(), *mid);
                    attempts.insert(*mid, attempt);
                }
                (Effect::Refused(me), Err(re)) => {
                    assert_eq!(me.canon(), CanonErr::of_real(re), "{ctx}");
                }
                _ => panic!("{ctx}: model/durable reserve divergence: {effect:?} vs {real:?}"),
            }
            effect
        }
        Op::Dispatch { target } => {
            let real = fixture
                .ledger
                .mark_dispatched(fixture.session, target.real_id(id_map))
                .await;
            let effect = apply_model(model, op);
            compare_unit(&effect, &real, ctx);
            if let Target::Known(id) = target {
                let row = &model.rows[id];
                if row.state == RState::Dispatched {
                    if let Some((tin, tout)) = row.known_usage {
                        fixture
                            .handle
                            .record_provider_call_attempt(
                                attempts[id],
                                Some(id_map[id]),
                                "fake",
                                "m",
                                "completed",
                                Some(tin),
                                Some(tout),
                                None,
                            )
                            .expect("completed provider call row");
                    }
                }
            }
            effect
        }
        Op::Fail { target } => {
            let real = fixture
                .ledger
                .mark_uncertain(
                    fixture.session,
                    target.real_id(id_map),
                    "modelcheck_failure".into(),
                    None,
                )
                .await;
            let effect = apply_model(model, op);
            compare_unit(&effect, &real, ctx);
            effect
        }
        Op::Refund { target } => {
            let real = fixture
                .ledger
                .refund(fixture.session, target.real_id(id_map))
                .await;
            let effect = apply_model(model, op);
            compare_unit(&effect, &real, ctx);
            effect
        }
        Op::SettleReported { target, actual } => {
            let real = fixture
                .ledger
                .settle_usage(
                    fixture.session,
                    target.real_id(id_map),
                    0,
                    0,
                    0,
                    0,
                    Some(*actual),
                    None,
                )
                .await;
            let effect = apply_model(model, op);
            compare_settle(&effect, &real, ctx);
            effect
        }
        Op::SettlePriced { target, tokens } => {
            let real = fixture
                .ledger
                .settle_usage(
                    fixture.session,
                    target.real_id(id_map),
                    tokens.0,
                    0,
                    0,
                    tokens.1,
                    None,
                    None,
                )
                .await;
            let effect = apply_model(model, op);
            compare_settle(&effect, &real, ctx);
            effect
        }
        Op::Recover => {
            let effect = apply_model(model, op);
            fixture.ledger.recover_after_restart();
            assert!(matches!(&effect, Effect::Recover), "{ctx}");
            effect
        }
        Op::Reconcile => {
            let effect = apply_model(model, op);
            let real = fixture
                .ledger
                .reconcile_uncertain(fixture.session, task)
                .await
                .expect("reconcile");
            let Effect::Reconcile(mine) = &effect else {
                panic!("{ctx}: reconcile effect");
            };
            assert_eq!(mine.settled, real.settled, "{ctx}: reconcile settled");
            assert_eq!(
                mine.closed_unknown, real.closed_unknown,
                "{ctx}: reconcile closed_unknown"
            );
            assert_eq!(
                mine.left_uncertain, real.left_uncertain,
                "{ctx}: reconcile left_uncertain"
            );
            assert_eq!(mine.charged, real.charged_micro, "{ctx}: reconcile charged");
            effect
        }
        Op::Finalize => {
            let effect = apply_model(model, op);
            let real = fixture
                .ledger
                .finalize_uncertain(fixture.session, task)
                .await
                .expect("finalize");
            let Effect::Finalize(mine) = &effect else {
                panic!("{ctx}: finalize effect");
            };
            assert_eq!(mine.settled, real.settled, "{ctx}: finalize settled");
            assert_eq!(mine.charged, real.charged_micro, "{ctx}: finalize charged");
            effect
        }
    };
    effect
}

fn compare_unit(effect: &Effect, real: &Result<(), BudgetError>, ctx: &str) {
    let Effect::Unit(mine) = effect else {
        panic!("{ctx}: unit effect expected");
    };
    match (mine, real) {
        (Ok(()), Ok(())) => {}
        (Err(me), Err(re)) => assert_eq!(me.canon(), CanonErr::of_real(re), "{ctx}"),
        _ => panic!("{ctx}: model/durable unit divergence: {mine:?} vs {real:?}"),
    }
}

fn compare_settle(effect: &Effect, real: &Result<Option<u64>, BudgetError>, ctx: &str) {
    let Effect::Settle(mine) = effect else {
        panic!("{ctx}: settle effect expected");
    };
    match (mine, real) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "{ctx}: settled amount divergence"),
        (Err(me), Err(re)) => assert_eq!(me.canon(), CanonErr::of_real(re), "{ctx}"),
        _ => panic!("{ctx}: model/durable settle divergence: {mine:?} vs {real:?}"),
    }
}

/// Drive one full trace against the REAL ledger, comparing after every op.
async fn durable_trace(fixture: &Fixture, seed: u64, trace: u64, ceiling_hint: u64) {
    let mut lcg = Lcg::new(seed ^ trace.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let cap = if lcg.chance(75) {
        Some(ceiling_hint)
    } else {
        None
    };
    let task = task_for(fixture, trace, cap);
    let mut model = Machine::new(cap);
    let mut id_map: BTreeMap<u64, ReservationId> = BTreeMap::new();
    let mut raw_to_model: BTreeMap<i64, u64> = BTreeMap::new();
    let mut attempts: BTreeMap<u64, ModelCallAttempt> = BTreeMap::new();
    let steps = 24 + lcg.below(17);
    for step in 0..steps {
        let ctx = format!("durable seed={seed:#x} trace={trace} step={step}");
        let op = gen_op(&mut lcg, &model, ceiling_hint);
        let before = pre(&model);
        let effect = apply_durable(
            fixture,
            task,
            &mut model,
            &mut id_map,
            &mut raw_to_model,
            &mut attempts,
            &op,
            trace,
            &ctx,
        )
        .await;
        assert_real_state(fixture, task, &model, &raw_to_model, &ctx);
        check_effect(&model, &op, &before, &effect, &ctx);
    }
    // Close every still-open row so a later trace's global recover sees no
    // foreign open reservation.
    let model_before = pre(&model);
    let effect = apply_model(&mut model, &Op::Recover);
    fixture.ledger.recover_after_restart();
    check_effect(
        &model,
        &Op::Recover,
        &model_before,
        &effect,
        "cleanup recover",
    );
    let real = fixture
        .ledger
        .reconcile_uncertain(fixture.session, task)
        .await
        .expect("cleanup reconcile");
    let effect = apply_model(&mut model, &Op::Reconcile);
    let Effect::Reconcile(mine) = effect else {
        unreachable!()
    };
    assert_eq!(mine.settled, real.settled, "cleanup reconcile settled");
    assert_eq!(
        mine.charged, real.charged_micro,
        "cleanup reconcile charged"
    );
    let real = fixture
        .ledger
        .finalize_uncertain(fixture.session, task)
        .await
        .expect("cleanup finalize");
    let effect = apply_model(&mut model, &Op::Finalize);
    let Effect::Finalize(mine) = effect else {
        unreachable!()
    };
    assert_eq!(mine.settled, real.settled, "cleanup finalize settled");
    let ctx = format!("durable seed={seed:#x} trace={trace} cleanup");
    assert_real_state(fixture, task, &model, &raw_to_model, &ctx);
    let scan = fixture
        .manager
        .store()
        .cost_reservation_invariants()
        .expect("invariant scan");
    assert_eq!(scan.open, 0, "{ctx}: open rows left behind");
    assert!(scan.dangling.is_empty(), "{ctx}: dangling reservations");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn pure_property_driver_runs_5000_seeded_traces_per_seed() {
    for seed in SEEDS {
        pure_driver(seed);
    }
}

#[tokio::test]
async fn durable_reference_comparison_smoke() {
    for seed in SEEDS {
        let fixture = fixture();
        for trace in 0..100u64 {
            durable_trace(&fixture, seed, trace, 60_000).await;
        }
    }
}

#[tokio::test]
#[ignore = "[soak] accounting reservation state machine vs the durable ledger, 5000 seeded traces per seed"]
async fn durable_reference_comparison_full() {
    for seed in SEEDS {
        let fixture = fixture();
        for trace in 0..TRACES_PER_SEED {
            durable_trace(&fixture, seed, trace, 60_000).await;
        }
    }
}
