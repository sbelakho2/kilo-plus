#include <stdio.h>

#include "ringbuf.h"

static int failures = 0;

static void expect(int cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", what);
        failures++;
    }
}

int main(void) {
    RingBuf rb;
    int v;

    rb_init(&rb);
    expect(rb_pop(&rb, &v) == -1, "pop on an empty buffer fails");

    rb_init(&rb);
    rb_push(&rb, 1);
    rb_push(&rb, 2);
    rb_push(&rb, 3);
    expect(rb_pop(&rb, &v) == 0 && v == 1, "basic FIFO order (first out)");
    expect(rb_pop(&rb, &v) == 0 && v == 2, "basic FIFO order (second out)");
    expect(rb_pop(&rb, &v) == 0 && v == 3, "basic FIFO order (third out)");
    expect(rb_pop(&rb, &v) == -1, "pop after drain fails");

    rb_init(&rb);
    for (int i = 1; i <= 6; i++) {
        rb_push(&rb, i);
    }
    expect(rb_pop(&rb, &v) == 0 && v == 3, "full buffer drops the oldest (first out is 3)");
    expect(rb_pop(&rb, &v) == 0 && v == 4, "full buffer keeps order (4)");
    expect(rb_pop(&rb, &v) == 0 && v == 5, "full buffer keeps order (5)");
    expect(rb_pop(&rb, &v) == 0 && v == 6, "full buffer keeps order (6)");
    expect(rb_pop(&rb, &v) == -1, "drained after wrap-around");

    rb_init(&rb);
    rb_push(&rb, 10);
    rb_push(&rb, 20);
    rb_push(&rb, 30);
    rb_push(&rb, 40);
    rb_push(&rb, 50);
    expect(
        rb_pop(&rb, &v) == 0 && v == 20,
        "push on a full buffer evicts the oldest element, not the newest"
    );

    if (failures) {
        fprintf(stderr, "%d assertion(s) failed\n", failures);
        return 1;
    }
    printf("ring buffer tests passed\n");
    return 0;
}
