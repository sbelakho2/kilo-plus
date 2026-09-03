# Completion proof (normative)

Faktor optimizes for correct VERIFIED task completion per dollar and per
user intervention — not for autonomous activity volume. A task that says
"done" without evidence is not done.

## VerificationRecord

A durable record produced before any logical turn may complete with
"verified":

```text
VerificationRecord {
    acceptance_criteria,
    criteria_results,
    tests_run,
    tests_passed,
    lint_results,
    typecheck_results,
    build_results,
    visual_evidence,       // screenshots/artifacts for UI criteria
    changed_files,
    unrelated_changes,     // reviewer-flagged extra edits
    unresolved_warnings,
    verification_timestamp,
    verifier,              // which engine/check set produced the record
}
```

## Rules

1. For mutating coding tasks, Completed-without-a-record is ILLEGAL
   whenever an objective verification mechanism exists (project type
   detected, checks derived, runner available).
2. A required check that FAILED can never yield Pass acceptance.
3. Test weakening/deletion without justification fails review.
4. "Done" claims must be supported by repository state (files actually
   changed, build green) — an evidence check runs before acceptance.

## Implementation status

- `crates/verify` (faktor-verify): project-type detection, derived checks
  (Rust/Node/Python/Go/Java/DotNet), acceptance semantics
  (Pass/Fail/Pending over REQUIRED checks), adversarial filter
  sanitization.
- Agent hook (`faktor-agent`): end-of-turn automatic verification for
  turns with changed files when a verifier is wired; results ride
  `TurnOutcome.verification` + `acceptance`; failures write durable
  `verification` memory facts; infra absence never fails the turn.
- Daemon wiring (`faktor-cli`): supervisor-backed runner (`sh -c`,
  30s/check, <=3 checks, wall-capped).
