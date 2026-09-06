# Acceptance criteria (machine-checked)

# The benchmark matches the `crit-NN` keys below against the run's final
# summary and its durable verification record. Keep the keys verbatim and
# report each criterion as PASS or FAIL with one line of evidence.

- crit-01: the repository test suite passes (`node --test`, node:test, no npm install and no build step)
- crit-02: values below lo are clamped up to lo (never to hi) and values above hi are clamped down to hi
- crit-03: in-range values pass through unchanged before summing
- crit-04: an empty array sums to zero
- crit-05: lo greater than hi throws an Error
