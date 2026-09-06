#!/bin/sh
# Repository-native verification: the module's own Go test suite.
# Fails (non-zero exit) while the planted bug is present.
set -eu
go test ./...
