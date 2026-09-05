//! Journaled task-budget ledger in integer micro-units (audit 79-80).
//!
//! The router's cost model works in MICRO-units with exact integer math
//! (`estimated_call_cost`, units proof from the economics suite) and the
//! route-time hard budget denies any candidate whose estimate exceeds
//! `task_budget_remaining_micro`. This module makes that one-shot deny
//! filter STATE-FUL and crash-safe: a per-task ledger with atomic
//! reservations, settlements, refunds/timeouts and an append-only journal
//! (grants AND denials) from which the full in-flight state reconstructs
//! after a crash. No float anywhere: every invariant below is
//! integer-exact.
//!
//! Model-checked invariants (see `tests`, 5000 seeded deterministic traces
//! per seed, inline LCG — workspace deps are frozen):
//!   (a) balance + committed + sum(in-flight reservations) == ceiling,
//!       integer-exact;
//!   (b) a reservation that would push committed above the ceiling is never
//!       granted (typed `Denied`), and the same denial semantics hold in
//!       the real `Router::route` hard-budget filter (cross-checked on
//!       every trace action);
//!   (c) settle-after-deny is impossible: a denied op was never reserved,
//!       so settling it is a typed `UnknownOp` error;
//!   (d) double-settle is rejected (a settled reservation is consumed);
//!   (e) refund/timeout are exactly-once per reservation;
//!   (f) crash mid-flight: replaying the journal reconstructs the budget
//!       identically (driven every 50 trace steps), and a tampered journal
//!       is a typed `Corrupt`, never a silent state change.
//!
//! The journal grows unboundedly with activity; the ledger is per-task and
//! tasks are bounded by the runtime, but a future compaction pass must
//! fold settled reservations, never truncate denials of live ones (see the
//! residual-risk note in the audit report).

use std::collections::HashMap;

use faktor_core::id::OpId;

/// Why a reservation attempt was refused. Denials are journaled (the
/// journal is the crash-reconstruction source: denials changed nothing).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenyReason {
    /// `est` exceeds the free balance (ceiling - committed - in-flight).
    Ceiling { free: u64 },
    /// The op id already holds a live reservation.
    Duplicate,
    /// A settlement was refused because `actual` exceeded the reservation
    /// (an overshoot would push committed past the ceiling).
    Overshoot { reserved: u64 },
}

/// One append-only journal record. Position in the journal is the record's
/// sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JournalEntry {
    /// A reservation was granted for `est`.
    Reserve { op: OpId, est: u64 },
    /// A reservation attempt was refused (state unchanged).
    Deny {
        op: OpId,
        est: u64,
        reason: DenyReason,
    },
    /// A reservation was settled at `actual`; `refunded` (= reserved -
    /// actual) returned to the free balance.
    Settle {
        op: OpId,
        actual: u64,
        refunded: u64,
    },
    /// A reservation was released without spending.
    Refund { op: OpId },
    /// A reservation expired (like a refund, distinct terminal record).
    Timeout { op: OpId },
}

/// Typed rejection reasons; every refusal is loud and precise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    /// No reservation exists for this op: settle/refund/timeout after a
    /// denial, after a settle, or twice — all impossible operations.
    UnknownOp(OpId),
    /// The op already holds a live reservation.
    AlreadyReserved(OpId),
    /// The reservation attempt was refused.
    Denied {
        op: OpId,
        est: u64,
        reason: DenyReason,
    },
    /// A settle whose actual exceeds the reservation was refused.
    Overshoot {
        op: OpId,
        reserved: u64,
        actual: u64,
    },
    /// Journal replay hit a structural inconsistency (tamper / corruption):
    /// duplicate grant for a live reservation, settle of an unknown op,
    /// reserved-total arithmetic overflow, ...
    Corrupt(String),
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetError::UnknownOp(op) => write!(f, "op {op} has no live reservation"),
            BudgetError::AlreadyReserved(op) => write!(f, "op {op} is already reserved"),
            BudgetError::Denied { op, est, reason } => {
                write!(
                    f,
                    "reservation of {est} micro for op {op} denied: {reason:?}"
                )
            }
            BudgetError::Overshoot {
                op,
                reserved,
                actual,
            } => write!(
                f,
                "settle of {actual} micro for op {op} exceeds its {reserved}-micro reservation"
            ),
            BudgetError::Corrupt(what) => write!(f, "corrupt budget journal: {what}"),
        }
    }
}

impl std::error::Error for BudgetError {}

/// The journaled budget ledger. Integer micro-units throughout; all
/// arithmetic is checked (a valid journal can never overflow — overflow
/// means corruption).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskBudget {
    ceiling: u64,
    /// Settled actuals (spent; durable once journaled).
    committed: u64,
    /// Live (in-flight) reservations: op → reserved micro.
    reservations: HashMap<OpId, u64>,
    /// Append-only journal: the crash-reconstruction source.
    journal: Vec<JournalEntry>,
}

impl TaskBudget {
    pub fn new(ceiling: u64) -> Self {
        Self {
            ceiling,
            committed: 0,
            reservations: HashMap::new(),
            journal: Vec::new(),
        }
    }

    pub fn ceiling(&self) -> u64 {
        self.ceiling
    }

    /// Settled (spent) micro.
    pub fn committed(&self) -> u64 {
        self.committed
    }

    /// Sum of live in-flight reservations.
    pub fn reserved_total(&self) -> u64 {
        self.reservations.values().sum()
    }

    /// Free balance: ceiling - committed - in-flight. Integer-exact under
    /// the (a)/(b) invariants, which the model-check enforces after every
    /// step.
    pub fn free(&self) -> u64 {
        self.ceiling
            .saturating_sub(self.committed)
            .saturating_sub(self.reserved_total())
    }

    pub fn reservation(&self, op: OpId) -> Option<u64> {
        self.reservations.get(&op).copied()
    }

    pub fn journal(&self) -> &[JournalEntry] {
        &self.journal
    }

    /// Reserve `est` micro for `op`. Granted iff `est <= free()` and the op
    /// has no live reservation; every refusal is journaled as a `Deny` and
    /// returns a typed error (never a silent state change).
    pub fn reserve(&mut self, op: OpId, est: u64) -> Result<(), BudgetError> {
        if self.reservations.contains_key(&op) {
            let reason = DenyReason::Duplicate;
            self.journal.push(JournalEntry::Deny { op, est, reason });
            return Err(BudgetError::AlreadyReserved(op));
        }
        let free = self.free();
        if est > free {
            let reason = DenyReason::Ceiling { free };
            self.journal.push(JournalEntry::Deny { op, est, reason });
            return Err(BudgetError::Denied { op, est, reason });
        }
        self.reservations.insert(op, est);
        self.journal.push(JournalEntry::Reserve { op, est });
        Ok(())
    }

    /// Settle `op` at `actual` micro. The reservation must be live and
    /// `actual <= reserved`; the difference is refunded to the free
    /// balance. A settlement consumes the reservation, so a second settle
    /// is a typed `UnknownOp` (invariant d), exactly like settling an op
    /// that was denied (invariant c). An overshooting actual is refused
    /// with the reservation intact.
    pub fn settle(&mut self, op: OpId, actual: u64) -> Result<u64, BudgetError> {
        let Some(reserved) = self.reservations.get(&op).copied() else {
            return Err(BudgetError::UnknownOp(op));
        };
        if actual > reserved {
            let reason = DenyReason::Overshoot { reserved };
            self.journal.push(JournalEntry::Deny {
                op,
                est: actual,
                reason,
            });
            return Err(BudgetError::Overshoot {
                op,
                reserved,
                actual,
            });
        }
        self.reservations.remove(&op);
        // actual <= reserved and reserved <= free-of-the-moment implies
        // committed + actual <= ceiling; checked_add is belt-and-braces for
        // corrupted input paths.
        let new_committed = self
            .committed
            .checked_add(actual)
            .ok_or_else(|| BudgetError::Corrupt("committed overflow on settle".into()))?;
        self.committed = new_committed;
        let refunded = reserved - actual;
        self.journal.push(JournalEntry::Settle {
            op,
            actual,
            refunded,
        });
        Ok(refunded)
    }

    /// Release `op`'s reservation without spending (exactly-once per
    /// reservation: a second refund — or a refund after settle — hits a
    /// consumed reservation and is a typed `UnknownOp`).
    pub fn refund(&mut self, op: OpId) -> Result<u64, BudgetError> {
        let Some(reserved) = self.reservations.get(&op).copied() else {
            return Err(BudgetError::UnknownOp(op));
        };
        self.reservations.remove(&op);
        self.journal.push(JournalEntry::Refund { op });
        Ok(reserved)
    }

    /// A reservation that expired (timeout): released like a refund with a
    /// distinct journal record; exactly-once per reservation.
    pub fn timeout(&mut self, op: OpId) -> Result<u64, BudgetError> {
        let Some(reserved) = self.reservations.get(&op).copied() else {
            return Err(BudgetError::UnknownOp(op));
        };
        self.reservations.remove(&op);
        self.journal.push(JournalEntry::Timeout { op });
        Ok(reserved)
    }

    /// Crash reconstruction: fold the journal back into a fresh budget.
    /// Denial records changed nothing and are skipped; every other record
    /// is applied with structural validation, so a truncated or tampered
    /// journal is a typed `Corrupt` — never a silently wrong balance.
    pub fn replay(ceiling: u64, journal: &[JournalEntry]) -> Result<Self, BudgetError> {
        let mut b = TaskBudget::new(ceiling);
        for entry in journal {
            match *entry {
                JournalEntry::Deny { .. } => {}
                JournalEntry::Reserve { op, est } => {
                    if b.reservations.contains_key(&op) {
                        return Err(BudgetError::Corrupt(format!(
                            "duplicate grant for live reservation of op {op}"
                        )));
                    }
                    // A genuine journal only ever granted within the ceiling.
                    if est > b.free() {
                        return Err(BudgetError::Corrupt(format!(
                            "grant of {est} micro for op {op} exceeds the free balance"
                        )));
                    }
                    b.reservations.insert(op, est);
                }
                JournalEntry::Settle {
                    op,
                    actual,
                    refunded,
                } => {
                    let Some(reserved) = b.reservations.remove(&op) else {
                        return Err(BudgetError::Corrupt(format!(
                            "settle of unreserved op {op}"
                        )));
                    };
                    if actual > reserved {
                        return Err(BudgetError::Corrupt(format!(
                            "settle of {actual} micro exceeds the {reserved}-micro reservation of op {op}"
                        )));
                    }
                    if reserved - actual != refunded {
                        return Err(BudgetError::Corrupt(format!(
                            "refunded amount mismatch for op {op}"
                        )));
                    }
                    b.committed = b.committed.checked_add(actual).ok_or_else(|| {
                        BudgetError::Corrupt("committed overflow during replay".into())
                    })?;
                }
                JournalEntry::Refund { op } | JournalEntry::Timeout { op } => {
                    if b.reservations.remove(&op).is_none() {
                        return Err(BudgetError::Corrupt(format!(
                            "release of unreserved op {op}"
                        )));
                    }
                }
            }
        }
        b.journal = journal.to_vec();
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RouteRequest, Router};
    use faktor_core::model::{
        ModelDescriptor, ModelEconomics, ModelSource, RateLimitState, RouterPhase,
    };

    // ------------------------------------------------------------------
    // Tiny inline LCG (no rand/proptest: workspace deps are frozen).
    // ------------------------------------------------------------------

    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1) | 1)
        }

        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) as u32 as u64
        }

        fn below(&mut self, n: u64) -> u64 {
            debug_assert!(n > 0);
            self.next() % n
        }

        fn chance(&mut self, percent: u64) -> bool {
            self.below(100) < percent
        }
    }

    const BUDGET_SEEDS: &[u64] = &[0xB0D6_E7B0_D6E7_0001, 0xB0D6_E7B0_D6E7_0002];
    const BUDGET_TRACES: u64 = 5000;

    // ------------------------------------------------------------------
    // The model-check driver (invariants a-f, see the module docs).
    // ------------------------------------------------------------------

    fn budget_driver(seed: u64) {
        for trace in 0..BUDGET_TRACES {
            let mut lcg = Lcg::new(seed ^ trace.wrapping_mul(0xD1B5_4A32_D192_ED03));
            // Ceiling + op-id universe per trace (ids mimic the durable
            // per-session op_id_seq allocation).
            let ceiling = 1_000 + lcg.below(200_000);
            let mut budget = TaskBudget::new(ceiling);
            let mut op_seq = 1u64;
            let mut live_ops: Vec<OpId> = Vec::new();
            let mut all_ops: Vec<OpId> = Vec::new();

            let steps = 40 + lcg.below(20);
            for step in 0..steps {
                let ctx = || format!("budget trace seed={seed:#x} trace={trace} step={step}");
                // ---- (f) crash reconstruction every 50 steps ------------
                if step > 0 && step % 50 == 0 {
                    let rebuilt = TaskBudget::replay(ceiling, budget.journal())
                        .unwrap_or_else(|e| panic!("replay failed at {}: {e}", ctx()));
                    assert_eq!(budget, rebuilt, "reconstruction diverges at {}", ctx());
                }
                // Every step: the invariants over the live machine.
                assert_invariants(&budget, &ctx());

                let pick = lcg.below(100);
                match pick {
                    // ---- reserve -----------------------------------------
                    0..=59 => {
                        let op = next_op(&mut lcg, &mut op_seq, &mut live_ops, &mut all_ops);
                        let est = 1 + lcg.below(ceiling.saturating_mul(2).max(2));
                        let free = budget.free();
                        let expect_grant = budget.reservation(op).is_none() && est <= free;
                        let result = budget.reserve(op, est);
                        match result {
                            Ok(()) => {
                                assert!(
                                    expect_grant,
                                    "{}: grant against a duplicate or over-ceiling reservation",
                                    ctx()
                                );
                                assert_eq!(
                                    budget.reservation(op),
                                    Some(est),
                                    "{}: reservation not recorded",
                                    ctx()
                                );
                                if !live_ops.contains(&op) {
                                    live_ops.push(op);
                                }
                                // Cross-check against the REAL router's
                                // hard-budget filter: the same est at the
                                // same free balance must be granted there.
                                assert!(
                                    route_grants(est, free),
                                    "{}: router route() denied a granted reservation (est={est}, free={free})",
                                    ctx()
                                );
                            }
                            Err(BudgetError::Denied {
                                reason: DenyReason::Ceiling { .. },
                                ..
                            }) => {
                                assert!(!expect_grant, "{}: spurious denial", ctx());
                            }
                            Err(BudgetError::AlreadyReserved(_)) => {
                                assert!(
                                    budget.reservation(op).is_some(),
                                    "{}: duplicate error without a live reservation",
                                    ctx()
                                );
                            }
                            Err(e) => panic!("{}: unexpected reserve error: {e}", ctx()),
                        }
                    }
                    // ---- settle ------------------------------------------
                    60..=79 => {
                        let op = random_op(&mut lcg, &live_ops);
                        let Some(reserved) = budget.reservation(op) else {
                            // Settle of an op without a live reservation:
                            // covers settle-after-deny and double-settle.
                            let err = budget.settle(op, 0).unwrap_err();
                            assert!(
                                matches!(err, BudgetError::UnknownOp(_)),
                                "{}: settle of an unreserved op must be UnknownOp",
                                ctx()
                            );
                            continue;
                        };
                        // Overshoot ~1/3 of the time: refused, reservation
                        // intact.
                        let overshoot = lcg.chance(33);
                        let actual = if overshoot {
                            reserved.saturating_add(1 + lcg.below(50))
                        } else {
                            lcg.below(reserved + 1)
                        };
                        let result = budget.settle(op, actual);
                        if overshoot {
                            let Err(BudgetError::Overshoot {
                                op: err_op,
                                reserved: err_reserved,
                                actual: err_actual,
                            }) = result
                            else {
                                panic!("{}: overshooting settle was not refused", ctx());
                            };
                            assert_eq!((err_op, err_reserved, err_actual), (op, reserved, actual));
                            assert_eq!(
                                budget.reservation(op),
                                Some(reserved),
                                "{}: refused settle must keep the reservation",
                                ctx()
                            );
                        } else {
                            let refunded =
                                result.unwrap_or_else(|e| panic!("{}: settle failed: {e}", ctx()));
                            assert_eq!(refunded, reserved - actual);
                            assert_eq!(
                                budget.reservation(op),
                                None,
                                "{}: settle must consume the reservation",
                                ctx()
                            );
                            live_ops.retain(|o| *o != op);
                        }
                    }
                    // ---- refund ------------------------------------------
                    80..=89 => {
                        let op = random_op(&mut lcg, &live_ops);
                        match budget.reservation(op) {
                            None => {
                                let err = budget.refund(op).unwrap_err();
                                assert!(
                                    matches!(err, BudgetError::UnknownOp(_)),
                                    "{}: double refund must be UnknownOp",
                                    ctx()
                                );
                            }
                            Some(_) => {
                                let returned = budget
                                    .refund(op)
                                    .unwrap_or_else(|e| panic!("{}: refund failed: {e}", ctx()));
                                assert!(returned > 0);
                                live_ops.retain(|o| *o != op);
                            }
                        }
                    }
                    // ---- timeout ------------------------------------------
                    90..=94 => {
                        let op = random_op(&mut lcg, &live_ops);
                        if budget.reservation(op).is_none() {
                            // timeout after settle/refund/deny: typed.
                            let err = budget.timeout(op).unwrap_err();
                            assert!(
                                matches!(err, BudgetError::UnknownOp(_)),
                                "{}: timeout of an unreserved op must be UnknownOp",
                                ctx()
                            );
                        } else {
                            budget.timeout(op).unwrap();
                            live_ops.retain(|o| *o != op);
                        }
                    }
                    // ---- forced deny --------------------------------------
                    _ => {
                        let op = random_op(&mut lcg, &live_ops);
                        let free = budget.free();
                        let est = free.saturating_add(1 + lcg.below(100));
                        let result = budget.reserve(op, est);
                        match result {
                            Err(BudgetError::Denied {
                                reason: DenyReason::Ceiling { .. },
                                ..
                            }) => {
                                // The real router's filter must also deny —
                                // but ONLY when the remaining budget is >= 1:
                                // RouteRequest documents 0 as "unlimited",
                                // so the router API itself cannot express a
                                // fully-spent (0-free) budget. The ledger
                                // denies at 0; the route filter is
                                // incomparable there (callers must clamp
                                // remaining to >= 1 micro when a task
                                // budget is active).
                                if free >= 1 {
                                    assert!(
                                        !route_grants(est, free),
                                        "{}: router() granted what the ledger denied (est={est}, free={free})",
                                        ctx()
                                    );
                                }
                            }
                            Err(BudgetError::AlreadyReserved(_)) => {
                                // duplicate of a live op — also journaled.
                                assert!(budget.reservation(op).is_some());
                            }
                            Ok(()) => panic!("{}: over-ceiling reservation granted", ctx()),
                            Err(e) => panic!("{}: unexpected reserve error: {e}", ctx()),
                        }
                    }
                }
            }
            assert_invariants(&budget, "end of trace");
            // (f) final reconstruction.
            let rebuilt = TaskBudget::replay(ceiling, budget.journal())
                .unwrap_or_else(|e| panic!("final replay failed: {e}"));
            assert_eq!(budget, rebuilt, "final reconstruction diverges");
        }
    }

    /// Invariant (a): committed + in-flight + free == ceiling, exact.
    /// Invariant (b): committed + in-flight never exceeds the ceiling.
    fn assert_invariants(b: &TaskBudget, ctx: &str) {
        let inflight = b.reserved_total();
        assert_eq!(
            b.committed()
                .saturating_add(inflight)
                .saturating_add(b.free()),
            b.ceiling(),
            "{ctx}: balance + committed + in-flight != ceiling"
        );
        assert!(
            b.committed().checked_add(inflight).is_some()
                && b.committed() + inflight <= b.ceiling(),
            "{ctx}: committed + in-flight exceeded the ceiling"
        );
    }

    fn random_op(lcg: &mut Lcg, live: &[OpId]) -> OpId {
        if live.is_empty() {
            return OpId::new(1 + lcg.below(8));
        }
        // Half the time target a live op, half the time a stale/unknown id
        // (double settle / settle-after-denied coverage is state-driven
        // anyway).
        if lcg.chance(50) {
            live[lcg.below(live.len() as u64) as usize]
        } else {
            OpId::new(1 + lcg.below(8))
        }
    }

    fn next_op(
        lcg: &mut Lcg,
        op_seq: &mut u64,
        live_ops: &mut Vec<OpId>,
        all_ops: &mut Vec<OpId>,
    ) -> OpId {
        // Mostly fresh ids (allocator semantics); occasionally reuse one of
        // the trace's own ids so duplicates occur.
        let _ = all_ops;
        if !live_ops.is_empty() && lcg.chance(15) {
            live_ops[lcg.below(live_ops.len() as u64) as usize]
        } else {
            *op_seq += 1;
            let op = OpId::new(*op_seq);
            live_ops.push(op);
            op
        }
    }

    // ------------------------------------------------------------------
    // Cross-check against the REAL router hard-budget filter.
    // ------------------------------------------------------------------

    /// A one-candidate router whose estimated cost for 1 input token is
    /// exactly `est` micro (input_price_per_mtok == est; zero output
    /// tokens, zero cache). `route()` must grant iff est <= remaining.
    fn route_grants(est: u64, remaining: u64) -> bool {
        let economics = ModelEconomics {
            input_price_per_mtok: est,
            output_price_per_mtok: 1,
            cache_read_price_per_mtok: 0,
            cache_write_price_per_mtok: 0,
            estimated_latency_ms: 1,
            tool_reliability: 100,
            reasoning_reliability: 100,
            coding_reliability: 100,
            context_reliability: 100,
            availability: 100,
            rate_limit_state: RateLimitState::Healthy,
        };
        let d = ModelDescriptor {
            provider: "mc".into(),
            model: "one".into(),
            context: 1,
            max_output: 1,
            tools: true,
            parallel_tools: true,
            reasoning: true,
            thinking: true,
            vision: false,
            structured_output: true,
            embeddings: false,
            streaming: true,
            economics,
            source: ModelSource::ProviderCatalog,
        };
        let r = Router::new(vec![d]);
        let req = RouteRequest {
            phase: RouterPhase::Implement,
            required_capabilities: vec!["tools".into()],
            context_tokens: 1,
            estimated_output_tokens: 0,
            quality_floor: 0,
            task_budget_remaining_micro: remaining,
            latency_preference_ms: None,
        };
        match r.route(&req, &[]) {
            Ok(d) => {
                // Exactness: the real filter must have costed this at `est`.
                assert_eq!(d.estimated_cost_micro, est, "router cost != ledger est");
                true
            }
            Err(e) => {
                assert!(
                    e.contains("budget"),
                    "router denial must name the budget: {e}"
                );
                false
            }
        }
    }

    // ------------------------------------------------------------------
    // Deterministic unit matrix + drivers
    // ------------------------------------------------------------------

    #[test]
    fn typed_refusal_matrix() {
        let mut b = TaskBudget::new(100);
        let a = OpId::new(1);
        let x = OpId::new(2);
        // Grant within the ceiling.
        b.reserve(a, 40).unwrap();
        assert_eq!(b.free(), 60);
        // Over-ceiling: typed deny, nothing changes.
        let err = b.reserve(x, 61).unwrap_err();
        assert!(matches!(
            err,
            BudgetError::Denied {
                reason: DenyReason::Ceiling { free: 60 },
                ..
            }
        ));
        assert_eq!(b.free(), 60);
        // Settle-after-deny is impossible (typed UnknownOp).
        assert!(matches!(
            b.settle(x, 5).unwrap_err(),
            BudgetError::UnknownOp(_)
        ));
        // Duplicate reserve: typed, journaled, first reservation intact.
        assert!(matches!(
            b.reserve(a, 1).unwrap_err(),
            BudgetError::AlreadyReserved(_)
        ));
        assert_eq!(b.reservation(a), Some(40));
        // Settle with actual > reserved is refused; reservation intact.
        assert!(matches!(
            b.settle(a, 41).unwrap_err(),
            BudgetError::Overshoot { .. }
        ));
        assert_eq!(b.reservation(a), Some(40));
        // Settle exactly: double-settle rejected afterwards.
        let refunded = b.settle(a, 35).unwrap();
        assert_eq!(refunded, 5);
        assert!(matches!(
            b.settle(a, 1).unwrap_err(),
            BudgetError::UnknownOp(_)
        ));
        // Refund once per reservation.
        b.reserve(x, 10).unwrap();
        assert_eq!(b.refund(x).unwrap(), 10);
        assert!(matches!(
            b.refund(x).unwrap_err(),
            BudgetError::UnknownOp(_)
        ));
        // Timeout after refund is also UnknownOp.
        assert!(matches!(
            b.timeout(x).unwrap_err(),
            BudgetError::UnknownOp(_)
        ));
        assert_invariants(&b, "typed matrix");
        let rebuilt = TaskBudget::replay(100, b.journal()).unwrap();
        assert_eq!(b, rebuilt);
    }

    #[test]
    fn tampered_journal_is_corrupt_not_silent() {
        let mut b = TaskBudget::new(50);
        let a = OpId::new(1);
        b.reserve(a, 20).unwrap();
        let mut journal = b.journal().to_vec();
        // A forged second grant for the same live op must be Corrupt.
        journal.push(JournalEntry::Reserve { op: a, est: 5 });
        assert!(matches!(
            TaskBudget::replay(50, &journal).unwrap_err(),
            BudgetError::Corrupt(_)
        ));
        // A forged settle of an unreserved op must be Corrupt.
        let journal = vec![JournalEntry::Settle {
            op: OpId::new(99),
            actual: 1,
            refunded: 0,
        }];
        assert!(matches!(
            TaskBudget::replay(50, &journal).unwrap_err(),
            BudgetError::Corrupt(_)
        ));
        // A truncated journal replays to a consistent-but-earlier state
        // (never a panic, never an over-commit): drop the last genuine
        // record and replay the prefix.
        b.settle(a, 20).unwrap();
        let full = b.journal().to_vec();
        let prefix = &full[..full.len() - 1];
        let rebuilt = TaskBudget::replay(50, prefix).unwrap();
        assert_invariants(&rebuilt, "truncated replay");
        assert_eq!(
            rebuilt.committed(),
            0,
            "truncated replay must not over-commit"
        );
    }

    #[test]
    fn budget_driver_seed_0xb0d6e7_a() {
        budget_driver(BUDGET_SEEDS[0]);
    }

    #[test]
    fn budget_driver_seed_0xb0d6e7_b() {
        budget_driver(BUDGET_SEEDS[1]);
    }
}
