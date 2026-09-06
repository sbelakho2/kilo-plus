#ifndef RINGBUF_H
#define RINGBUF_H

#define RB_CAP 4

typedef struct {
    int data[RB_CAP];
    int head;
    int count;
} RingBuf;

void rb_init(RingBuf *rb);

/* Append v. When the buffer is full the OLDEST element is dropped first:
 * the buffer always keeps the most recent RB_CAP pushed elements. */
void rb_push(RingBuf *rb, int v);

/* Remove and return the OLDEST element. Returns 0 on success, -1 when
 * the buffer is empty (out is untouched then). */
int rb_pop(RingBuf *rb, int *out);

#endif
