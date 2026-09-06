# Acceptance criteria (machine-checked)

# The benchmark matches the `crit-NN` keys below against the run's final
# summary and its durable verification record. Keep the keys verbatim and
# report each criterion as PASS or FAIL with one line of evidence.

- crit-01: the repository test suite passes (pytest when available, else the stdlib unittest runner over the same tests)
- crit-02: duplicates are removed no matter how far apart they appear
- crit-03: the order of first occurrences is preserved exactly
- crit-04: a new list is returned and the input list is never mutated
