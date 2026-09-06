#include "ringbuf.h"

void rb_init(RingBuf *rb) {
    rb->head = 0;
    rb->count = 0;
}

void rb_push(RingBuf *rb, int v) {
    if (rb->count == RB_CAP) {
        /* BUG: on a full buffer the new value overwrites the NEWEST
         * element in place. The oldest element survives, so the buffer
         * never forgets and the FIFO order corrupts. */
        rb->data[(rb->head + rb->count - 1) % RB_CAP] = v;
        return;
    }
    rb->data[(rb->head + rb->count) % RB_CAP] = v;
    rb->count++;
}

int rb_pop(RingBuf *rb, int *out) {
    if (rb->count == 0) {
        return -1;
    }
    *out = rb->data[rb->head];
    rb->head = (rb->head + 1) % RB_CAP;
    rb->count--;
    return 0;
}
