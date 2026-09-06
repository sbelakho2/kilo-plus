# Acceptance criteria (machine-checked)

# The benchmark matches the `crit-NN` keys below against the run's final
# summary and its durable verification record. Keep the keys verbatim and
# report each criterion as PASS or FAIL with one line of evidence.

- crit-01: the repository test suite passes end to end (`cargo test`, no new dependencies)
- crit-02: word order is reversed while the letters within each word are unchanged
- crit-03: whitespace runs collapse to a single space separator in the output
- crit-04: empty and whitespace-only inputs return an empty string
