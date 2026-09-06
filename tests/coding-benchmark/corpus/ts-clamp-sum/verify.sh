#!/bin/sh
# Repository-native verification. The suite is node:test over the real
# .ts sources, run by node's built-in TypeScript type stripping: no npm
# install, no build step. Requires node >= 22.6 (>= 23.6 without the
# experimental flag).
# Fails (non-zero exit) while the planted bug is present.
set -eu
cd "$(dirname "$0")"
NODE_MAJOR=$(node -p 'process.versions.node.split(".")[0]')
NODE_MINOR=$(node -p 'process.versions.node.split(".")[1]')
if [ "$NODE_MAJOR" -ge 24 ] || { [ "$NODE_MAJOR" -eq 23 ] && [ "$NODE_MINOR" -ge 6 ]; }; then
    exec node --test
fi
if { [ "$NODE_MAJOR" -eq 23 ] && [ "$NODE_MINOR" -lt 6 ]; } || { [ "$NODE_MAJOR" -eq 22 ] && [ "$NODE_MINOR" -ge 6 ]; }; then
    exec node --experimental-strip-types --test
fi
echo "verify.sh: node >= 22.6 required (found $(node --version 2>/dev/null || echo none))" >&2
exit 1
