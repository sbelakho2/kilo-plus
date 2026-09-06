# Fix: values below `lo` are clamped to `hi`, not `lo`

`clampSum(nums, lo, hi)` must sum `nums` after clamping every element
into the inclusive range `[lo, hi]`: values below `lo` become `lo`,
values above `hi` become `hi`.

Reported symptoms (a recent change broke the lower bound):

- `clampSum([1, 2, 3], 0, 5)` currently returns `15`; the expected
  result is `6` (1+2+3 — the values are already inside the range and
  must pass through unchanged).
- `clampSum([-5, 0, 5], 0, 10)` currently returns `30`; the expected
  result is `5` (0 + 0 + 5). In particular `-5` is currently clamped UP
  to `10` instead of down-toward `0`.

Contract:

- Every element is clamped into `[lo, hi]` (inclusive on both sides)
  before summing.
- An in-range element passes through unchanged.
- An empty array sums to zero.
- `lo > hi` throws an Error.
- The existing suite in `clamp.test.ts` (node:test) documents the
  contract; make it pass without changing the tests. No dependencies may
  be added and no build step may be introduced.
