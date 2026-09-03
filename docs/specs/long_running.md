# Long-running operation (normative)

Conversations are disposable; tasks are not. Three lifetimes:

| Lifetime | Bound |
|---|---|
| Session | effectively unlimited / configurable |
| Task | unlimited or explicit user budget |
| Operation | ALWAYS bounded (provider request, MCP call, shell, tool deadlines) |

## Durable task state (extends TaskLedger — no competing memory)

goal, acceptance criteria, current plan/DAG, completed milestones,
decisions, constraints, files examined/modified, symbol knowledge,
commands run, test results, known failures, current hypothesis,
unresolved blockers, next intended action, subagent tree, cost spent/
remaining, provider/model state, external effects, verification state.

## Semantics

- Closing the editor must not stop the task.
- Daemon restart must not lose the task.
- Provider disconnect must not lose the task.
- Machine restart reconstructs the durable task.
- Compaction must not lose acceptance criteria (hard invariant test).
- Continuation from repository + journal + ledger + artifacts with ZERO
  prior transcript tokens must reproduce what remains (explicit test).

## Status

- Durable turns, receipts, queue, recovery, and ledger exist and are
  test-covered. Multi-day plan/milestone rows and the 24-hour turn
  deadline re-examination land with the plan-DAG work.
