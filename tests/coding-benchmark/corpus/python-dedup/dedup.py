"""dedup(items) returns a NEW list with every duplicate removed, keeping the
order of FIRST occurrence of each element."""


def dedup(items):
    """Remove duplicates while preserving first-occurrence order.

    BUG: only CONSECUTIVE duplicates are removed. A value that repeats
    after other elements in between is kept a second time, so the result
    is not duplicate-free.
    """
    out = []
    for item in items:
        if not out or out[-1] != item:
            out.append(item)
    return out
