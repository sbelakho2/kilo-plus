# Acceptance criteria (machine-checked)

# The benchmark matches the `crit-NN` keys below against the run's final
# summary and its durable verification record. Keep the keys verbatim and
# report each criterion as PASS or FAIL with one line of evidence.

- crit-01: the repository test runner passes end to end (javac + java main-based runner, no JUnit and no build system)
- crit-02: century years not divisible by 400 (1900, 2100) are not leap years
- crit-03: years divisible by 400 (2000) are leap years
- crit-04: ordinary leap years (1996, 2004) are recognized and non-leap years (1997) are rejected
- crit-05: non-positive years throw IllegalArgumentException
