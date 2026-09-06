# Acceptance criteria (machine-checked)

# The benchmark matches the `crit-NN` keys below against the run's final
# summary and its durable verification record. Keep the keys verbatim and
# report each criterion as PASS or FAIL with one line of evidence.

- crit-01: the repository test suite passes end to end (`make test`)
- crit-02: basic FIFO order holds while the buffer is not full
- crit-03: a push on a full buffer evicts the OLDEST element (never the newest) and keeps the most recent 4 values in order
- crit-04: popping an empty buffer returns -1 and leaves the output untouched
- crit-05: the implementation never writes outside the fixed data array
