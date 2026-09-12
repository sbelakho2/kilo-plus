//! The PR/CI-fix completion contract (P2): pure, provider-agnostic types
//! shared by the native task-start DTO, the executor's run request, the
//! durable session ledger and the completion gate.
//!
//! [`CompletionContract`] declares which conditional steps a task run must
//! durably evidence before `VerifiedComplete` may land. All-false is the
//! default contract: the completion path behaves byte-identically to a tree
//! that never carried the field, and no durable row is written. The gate
//! itself lives in `faktor-session`/`faktor-agent`; this module only owns
//! the frozen wire/durability vocabulary.
//!
//! Follow-up boundary: commit/push/PR step EXECUTION does not exist in this
//! tree. Callers (and tests) record the per-step outcomes through
//! `SessionHandle::set_completion_step_status`; the completion gate is the
//! contract this change lands. A future step executor must call the same
//! durable setter, nothing else.

use std::fmt;

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

/// One conditional completion step of a PR/CI-fix task. The declaration
/// order is the gate evaluation order (commit, push, pr): the first unmet
/// requested step is the one a refusal names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStep {
    Commit,
    Push,
    Pr,
}

impl CompletionStep {
    /// Every step in gate evaluation order.
    pub const ALL: [CompletionStep; 3] = [
        CompletionStep::Commit,
        CompletionStep::Push,
        CompletionStep::Pr,
    ];

    /// The stable wire/durable tag of this step (also the JSON tag).
    pub const fn tag(self) -> &'static str {
        match self {
            CompletionStep::Commit => "commit",
            CompletionStep::Push => "push",
            CompletionStep::Pr => "pr",
        }
    }
}

/// The durable outcome of one requested step. `Succeeded` is the only
/// status that satisfies the gate; a `Failed` row is a TERMINAL refusal
/// (the run can never certify under its contract revision), `Skipped` is a
/// non-succeeded refusal that stays retryable like a missing row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStepOutcome {
    Succeeded,
    Failed,
    Skipped,
}

impl CompletionStepOutcome {
    /// The stable wire/durable tag of this outcome (also the JSON tag).
    pub const fn tag(self) -> &'static str {
        match self {
            CompletionStepOutcome::Succeeded => "succeeded",
            CompletionStepOutcome::Failed => "failed",
            CompletionStepOutcome::Skipped => "skipped",
        }
    }
}

/// The completion contract of one task run. All three members are required
/// on the wire (a missing member is a typed parse error, never a silent
/// false) and unknown members are refused, so a hostile DTO can never
/// half-declare a contract. Deserialization accepts ONLY the map form: a
/// positional JSON array is rejected too (serde's derived struct decoder
/// would otherwise accept sequences positionally). The struct is parsed both
/// from the native start DTO and from durable ledger rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CompletionContract {
    pub include_commit: bool,
    pub include_push: bool,
    pub include_pr: bool,
}

const COMPLETION_CONTRACT_FIELDS: &[&str] = &["include_commit", "include_push", "include_pr"];

impl<'de> Deserialize<'de> for CompletionContract {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ContractVisitor;

        impl<'de> Visitor<'de> for ContractVisitor {
            type Value = CompletionContract;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(
                    "a completion_contract object with boolean include_commit, include_push and include_pr members",
                )
            }

            fn visit_map<A>(self, mut map: A) -> Result<CompletionContract, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut include_commit: Option<bool> = None;
                let mut include_push: Option<bool> = None;
                let mut include_pr: Option<bool> = None;
                while let Some(field) = map.next_key::<String>()? {
                    match field.as_str() {
                        "include_commit" => {
                            if include_commit.is_some() {
                                return Err(de::Error::duplicate_field("include_commit"));
                            }
                            include_commit = Some(map.next_value()?);
                        }
                        "include_push" => {
                            if include_push.is_some() {
                                return Err(de::Error::duplicate_field("include_push"));
                            }
                            include_push = Some(map.next_value()?);
                        }
                        "include_pr" => {
                            if include_pr.is_some() {
                                return Err(de::Error::duplicate_field("include_pr"));
                            }
                            include_pr = Some(map.next_value()?);
                        }
                        other => {
                            return Err(de::Error::unknown_field(other, COMPLETION_CONTRACT_FIELDS))
                        }
                    }
                }
                Ok(CompletionContract {
                    include_commit: include_commit
                        .ok_or_else(|| de::Error::missing_field("include_commit"))?,
                    include_push: include_push
                        .ok_or_else(|| de::Error::missing_field("include_push"))?,
                    include_pr: include_pr.ok_or_else(|| de::Error::missing_field("include_pr"))?,
                })
            }
        }

        // `deserialize_map` (not `deserialize_any`): only the JSON object /
        // map form is accepted, never a positional sequence.
        deserializer.deserialize_map(ContractVisitor)
    }
}

impl CompletionContract {
    /// The explicit all-false contract: today's behavior (no gate, no
    /// durable rows).
    pub const NONE: CompletionContract = CompletionContract {
        include_commit: false,
        include_push: false,
        include_pr: false,
    };

    /// True when no step is requested — the default behavior.
    pub const fn is_default(&self) -> bool {
        !self.include_commit && !self.include_push && !self.include_pr
    }

    /// The requested steps in gate evaluation order (commit, push, pr).
    pub fn requested_steps(&self) -> Vec<CompletionStep> {
        CompletionStep::ALL
            .into_iter()
            .filter(|step| match step {
                CompletionStep::Commit => self.include_commit,
                CompletionStep::Push => self.include_push,
                CompletionStep::Pr => self.include_pr,
            })
            .collect()
    }
}

impl Default for CompletionContract {
    fn default() -> Self {
        Self::NONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_false_is_the_default_and_requests_nothing() {
        assert!(CompletionContract::default().is_default());
        assert!(CompletionContract::NONE.requested_steps().is_empty());
    }

    #[test]
    fn requested_steps_follow_gate_order() {
        let c = CompletionContract {
            include_commit: true,
            include_push: false,
            include_pr: true,
        };
        assert_eq!(
            c.requested_steps(),
            vec![CompletionStep::Commit, CompletionStep::Pr]
        );
    }

    #[test]
    fn contract_dto_is_strict() {
        // A missing member is a parse error, never a silent false.
        let err =
            serde_json::from_str::<CompletionContract>(r#"{"include_commit": true}"#).unwrap_err();
        assert!(err.to_string().contains("missing field"), "{err}");
        // Unknown members are refused.
        let err = serde_json::from_str::<CompletionContract>(
            r#"{"include_commit": true, "include_push": true, "include_pr": true, "extra": 1}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown field"), "{err}");
        // Non-boolean members are refused.
        assert!(serde_json::from_str::<CompletionContract>(
            r#"{"include_commit": "yes", "include_push": false, "include_pr": false}"#
        )
        .is_err());
        // A positional sequence is refused too: only the object form is a
        // contract.
        assert!(serde_json::from_str::<CompletionContract>("[true, true, true]").is_err());
        // Duplicate members are refused.
        assert!(serde_json::from_str::<CompletionContract>(
            r#"{"include_commit": true, "include_commit": true, "include_push": false, "include_pr": false}"#
        )
        .is_err());
    }
}
