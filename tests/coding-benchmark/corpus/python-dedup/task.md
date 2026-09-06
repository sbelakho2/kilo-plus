# Fix: dedup must remove duplicates wherever they appear

`dedup(items)` must return a NEW list containing every element of
`items` exactly once, keeping the order of FIRST occurrence.

Reported symptoms (a recent refactor broke it):

- `dedup(["a", "b", "a", "c"])` currently returns
  `["a", "b", "a", "c"]`; the expected result is `["a", "b", "c"]`.
- `dedup([1, 2, 1, 3, 2, 1, 4])` currently keeps the repeated `1`s and
  `2`s; the expected result is `[1, 2, 3, 4]`.

Contract:

- Duplicates are removed no matter how far apart they appear; only the
  first occurrence survives.
- The order of first occurrences is preserved exactly.
- The input list is never mutated (a fresh list is returned).
- The existing suite in `tests/test_dedup.py` documents the contract;
  make it pass without changing the tests.
