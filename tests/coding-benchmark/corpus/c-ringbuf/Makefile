CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Werror

test: test_ringbuf.c ringbuf.c ringbuf.h
	$(CC) $(CFLAGS) -o test_ringbuf test_ringbuf.c ringbuf.c
	./test_ringbuf

clean:
	rm -f test_ringbuf
