# Acceptance criteria (machine-checked)

# The benchmark matches the `crit-NN` keys below against the run's final
# summary and its durable verification record. Keep the keys verbatim and
# report each criterion as PASS or FAIL with one line of evidence.

- crit-01: the repository test suite passes (`go test ./...`, no new dependencies)
- crit-02: range endpoints are inclusive on BOTH sides ("1-3" sums to 6)
- crit-03: single-number tokens still sum correctly and mixed tokens parse together
- crit-04: empty input, malformed tokens, negative numbers, and reversed ranges are reported as errors, never panics
- crit-05: surrounding spaces around tokens and range parts are tolerated
