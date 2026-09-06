# Fix: a bounded FIFO that never forgets its oldest element

`ringbuf` is a small fixed-capacity FIFO (capacity 4). The contract of
`rb_push`: when the buffer is full, the OLDEST element is dropped and the
new value is appended — the buffer keeps the most recent 4 pushed
elements in FIFO order.

Reported symptoms (a recent change to `rb_push` broke eviction):

- Push 1,2,3,4, then 5 and 6. Popping now yields `1, 2, 3, 6`;
  the expected sequence is `3, 4, 5, 6`.
- Push 10,20,30,40, then 50. The first pop now returns `10` (the oldest
  element that should have been evicted); the expected first pop is `20`.

Contract:

- `rb_push` appends at the tail; on a full buffer the oldest element is
  evicted first.
- `rb_pop` removes the oldest element; popping an empty buffer returns
  -1 and leaves the output untouched.
- No memory errors: the implementation must stay within the fixed
  `data[RB_CAP]` array for every input sequence (run with sanitizers if
  you like — the tests above must pass first).
- The existing `test_ringbuf.c` suite documents the contract; make it
  pass without changing the tests.
