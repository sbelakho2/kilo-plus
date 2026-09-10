//! Verified Work per Token — the audited efficiency headline KPI.
//!
//! ```text
//! verified_work        = sum(weight_t for tasks t with verified_t)
//! total_tokens         = sum(tokens_t)
//! verified_work/token  = verified_work / total_tokens     (None when 0 tokens)
//! ```
//!
//! CRITERION-SPLITTING RESISTANCE. The numerator is weighted by TASK, never
//! by criterion count: a task contributes its benchmark `weight` when it is
//! verified, and nothing when it is not. Criterion totals/passes are carried
//! for coverage reporting only and deliberately never enter the numerator.
//! Splitting one acceptance criterion into three, or adding ten more, can
//! therefore not inflate the score — only a task actually reaching its
//! durable `VerifiedComplete` proof can move the numerator, and it moves it
//! by exactly its fixed benchmark weight. The unit tests lock this property
//! against a per-criterion "naive" scorer that does inflate.

use serde::{Deserialize, Serialize};

/// One benchmark task's scoring row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskWorkRow {
    pub task_id: String,
    /// The task's benchmark weight: fixed by the corpus, independent of how
    /// many acceptance criteria it carries.
    pub weight: u64,
    /// Acceptance criteria the task carries (coverage reporting only; NOT
    /// part of the work numerator — this is the splitting-resistance rule).
    pub criteria_total: u32,
    /// Criteria the verification record certified as passed (coverage
    /// reporting only).
    pub criteria_passed: u32,
    /// Durable verified-success signal (`VerifiedComplete`).
    pub verified: bool,
    /// Durable token total attributable to the task.
    pub tokens: u64,
}

/// The weighted Verified Work per Token KPI.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VerifiedWorkScore {
    /// Sum of benchmark weights over verified tasks.
    pub verified_work: f64,
    /// Sum of durable token totals over every task row.
    pub tokens: u64,
    /// `verified_work / tokens`; `None` when no tokens were spent (the KPI
    /// is undefined, never silently zero).
    pub work_per_token: Option<f64>,
}

/// Compute the weighted KPI over the task rows. Deterministic; criterion
/// counts are read but never weighted (see module docs).
pub fn verified_work_score(rows: &[TaskWorkRow]) -> VerifiedWorkScore {
    let mut verified_work = 0.0f64;
    let mut tokens = 0u64;
    for row in rows {
        tokens = tokens.saturating_add(row.tokens);
        if row.verified {
            verified_work += row.weight as f64;
        }
    }
    let work_per_token = if tokens == 0 {
        None
    } else {
        Some(verified_work / tokens as f64)
    };
    VerifiedWorkScore {
        verified_work,
        tokens,
        work_per_token,
    }
}
