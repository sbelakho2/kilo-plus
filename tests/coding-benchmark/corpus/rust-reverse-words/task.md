# Fix: reverse the order of words, not the letters

`reverse_words(s)` must return the words of `s` in REVERSE ORDER with
exactly one space between them. The letters inside each word must be
untouched.

Reported symptoms (a teammate's recent change broke this):

- `reverse_words("one two three")` currently returns `"eerht owt eno"`;
  the expected result is `"three two one"`.
- `reverse_words("hello world")` currently returns `"olleh dlrow"`;
  the expected result is `"world hello"`.

Contract:

- The output joins the words of the input (split on any whitespace run:
  spaces, tabs, newlines) in reverse order with single spaces.
- Punctuation stays attached to the word it touches (`"a, b"` →
  `"b a,"` is wrong; the comma belongs to `"a"`).
- An empty or whitespace-only input returns an empty string.
- The existing test suite documents the contract; make it pass without
  changing the tests.
