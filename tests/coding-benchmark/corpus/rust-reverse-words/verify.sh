#!/bin/sh
# Repository-native verification: the crate's own test suite.
# Fails (non-zero exit) while the planted bug is present.
set -eu
cargo test --offline --quiet
