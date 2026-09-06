#!/bin/sh
# Repository-native verification. The suite is unittest-based, so both
# drivers collect it: pytest when the runner has it, else the stdlib
# unittest runner. Fails (non-zero exit) while the planted bug is present.
set -eu
cd "$(dirname "$0")"
if command -v pytest >/dev/null 2>&1; then
    python3 -m pytest -q tests/test_dedup.py
else
    python3 -m unittest discover -s tests -t . -v
fi
