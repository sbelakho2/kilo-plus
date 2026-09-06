# Fix: inclusive ranges lose their last number

`SumRanges(s)` parses comma-separated integer tokens and sums the
numbers they denote. A token is a single integer or an inclusive range
`a-b` (both endpoints count: `1-3` denotes {1,2,3}).

Reported symptoms (a recent change broke ranges):

- `SumRanges("1-3")` currently returns `3` (it sums 1+2); the expected
  result is `6` (1+2+3).
- `SumRanges("1-3,5")` currently returns `8`; the expected result is
  `11`.
- Single numbers still work: `SumRanges("5")` returns `5`.

Contract:

- Every range token `a-b` with `a <= b` contributes the sum of ALL
  integers from `a` to `b` inclusive, including the endpoints.
- A token that is just a number contributes that number.
- Errors (returned, not panicked): empty input, an empty token, a
  malformed token, a negative number, and a reversed range `a-b` with
  `a > b`.
- Spaces around tokens and range parts are tolerated.
- The existing suite in `sumranges_test.go` documents the contract; make
  it pass without changing the tests.
