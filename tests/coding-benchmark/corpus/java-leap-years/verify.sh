#!/bin/sh
# Repository-native verification: compile with javac and run the
# main-based test runner (dependency-free; no JUnit).
# Fails (non-zero exit) while the planted bug is present.
set -eu
cd "$(dirname "$0")"
rm -rf .bench-out
mkdir .bench-out
javac -d .bench-out LeapYears.java TestLeapYears.java
java -cp .bench-out TestLeapYears
rm -rf .bench-out
