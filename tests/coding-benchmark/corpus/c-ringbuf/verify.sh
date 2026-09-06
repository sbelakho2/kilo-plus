#!/bin/sh
# Repository-native verification: build with the project's own Makefile
# and run the assertion-based test runner.
# Fails (non-zero exit) while the planted bug is present.
set -eu
make -s test
