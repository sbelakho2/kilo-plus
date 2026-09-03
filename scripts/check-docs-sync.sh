#!/usr/bin/env bash
# Docs drift guard (audit round 2, P1).
#
# Fails CI when docs/architecture.md mentions identifiers that no longer
# exist in the code, or omits the current scheduler/auth/stdout-contract
# names. Stale docs are a review-rejected artifact: the architecture doc is
# the normative spec, so drift must be loud, never silent.
#
# No external dependencies beyond grep/echo. Run from the repository root.
set -u

DOC=docs/architecture.md
fail=0

if [ ! -f "$DOC" ]; then
    echo "FATAL: $DOC not found (run from the repository root)" >&2
    exit 1
fi

# Identifiers that must NEVER appear in the doc. Verified against the code
# with grep at review time:
#   - TaskSpec            -> renamed to ScheduledOp (crates/scheduler)
#   - CancelFlag          -> removed (no such type anywhere in crates/)
#   - depends_on          -> renamed to dependencies: Vec<(OpId, DependencyPolicy)>
#   - FAKTOR_PLUS_HANDSHAKE -> legacy JSON handshake; the frozen stdout
#                            contract is the startup line (server never
#                            prints the handshake)
STALE_TOKENS=(
    TaskSpec
    CancelFlag
    depends_on
    FAKTOR_PLUS_HANDSHAKE
)

for token in "${STALE_TOKENS[@]}"; do
    if grep -q "$token" "$DOC"; then
        echo "STALE: '$token' is still mentioned in $DOC but no longer exists in the code" >&2
        fail=1
    fi
done

# Auth drift: 'Bearer <password>' is NOT the only accepted claim. The frozen
# v7.5.6 extension authenticates every request (including /global/health)
# with `Authorization: Basic base64("kilo:" + password)`; the Faktor-native
# x-faktor-server-password header and legacy per-start token also remain.
if ! grep -q "x-faktor-server-password" "$DOC"; then
    echo "STALE: 'Bearer <password>' implied as the only auth form — x-faktor-server-password must be documented" >&2
    fail=1
fi

# Current scheduler API names the doc MUST mention (each verified to exist
# in crates/scheduler/src/lib.rs: ScheduledOp, tokio::task::JoinSet,
# DependencyPolicy).
for token in ScheduledOp JoinSet DependencyPolicy; do
    if ! grep -q "$token" "$DOC"; then
        echo "MISSING: '$token' is part of the current scheduler API but absent from $DOC" >&2
        fail=1
    fi
done

# The frozen stdout contract: the startup line, not a JSON handshake.
if ! grep -q "kilo server listening on" "$DOC"; then
    echo "MISSING: the frozen startup line ('faktor server listening on http://127.0.0.1:<port>') is absent from $DOC" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "$DOC is out of sync with the code — fix the tokens listed above before merging." >&2
    exit 1
fi

echo "docs/architecture.md is in sync: no stale identifiers, current API names present."
