# Fix: century years are misclassified as leap years

`LeapYears.isLeap(year)` implements the Gregorian rule: a year is leap
when divisible by 4, EXCEPT century years, which are leap only when
divisible by 400. Non-positive years must be rejected.

Reported symptoms (a recent change dropped the century exception):

- `isLeap(1900)` currently returns `true`; the expected result is
  `false` (1900 is a century year not divisible by 400).
- `isLeap(2100)` currently returns `true`; the expected result is
  `false`.
- `isLeap(2000)` must stay `true` (divisible by 400) and ordinary years
  like 1996/2004 must stay `true`, 1997 `false`.

Contract:

- `year % 4 == 0`, except century years (`year % 100 == 0`) which need
  `year % 400 == 0` as well.
- `year <= 0` throws `IllegalArgumentException`.
- The existing `TestLeapYears` runner (plain javac + java, no JUnit)
  documents the contract; make it pass without changing the tests.
