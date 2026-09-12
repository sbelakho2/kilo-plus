#!/usr/bin/env bash
# Docs drift guard (audit round 2, P1 + truth-refresh wave).
#
# Fails CI when:
#   - docs/architecture.md mentions identifiers that no longer exist in the
#     code, or omits the current scheduler/auth/stdout-contract names;
#   - README.md / docs/certification.md contradict the tree semantically:
#       * crates/orchestrator/src/completion_steps.rs exists  => completion
#         step EXECUTION may not be described as a follow-up / absent;
#       * crates/winjob uses AssignProcessToJobObject      => Job Objects may
#         not be called a future/target mechanism;
#       * ui/upstream.json exists (VS Code webview vendored) => the docs may
#         not claim the upstream UI is not vendored, except where the claim
#         is explicitly JetBrains-7.1.2-scoped.
# Stale/contradicting docs are a review-rejected artifact: drift must be
# loud, never silent.
#
# No external dependencies beyond grep/echo. Run from the repository root.
set -u

DOC=docs/architecture.md
TRUTH_DOCS=(README.md docs/certification.md)
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
if ! grep -q "faktor server listening on" "$DOC"; then
    echo "MISSING: the frozen startup line ('faktor server listening on http://127.0.0.1:<port>') is absent from $DOC" >&2
    fail=1
fi

# ---------------------------------------------------------- semantic truth
#
# Whitespace-normalized view of one doc (line wrapping must not hide a
# contradiction).
normalized() {
    tr '\n' ' ' < "$1" | tr -s ' ' | sed 's/7\.1\.2/712/g'
}

# 1. Completion-step execution: the runner is in the tree, so no truth doc
#    may still call execution a follow-up or claim no runner exists.
if [ -f crates/orchestrator/src/completion_steps.rs ]; then
    for doc in "${TRUTH_DOCS[@]}"; do
        [ -f "$doc" ] || continue
        if normalized "$doc" | grep -Eqi 'execution[^.]{0,60}follow[- ]?up'; then
            echo "CONTRADICTION: $doc still calls completion-step execution a follow-up, but crates/orchestrator/src/completion_steps.rs exists" >&2
            fail=1
        fi
        if normalized "$doc" | grep -Eqi '(no|without)[^.]{0,40}automatic[^.]{0,80}(commit|push|pr)[^.]{0,60}runner'; then
            echo "CONTRADICTION: $doc claims there is no automatic commit/push/PR runner, but crates/orchestrator/src/completion_steps.rs exists" >&2
            fail=1
        fi
    done
    if ! grep -q 'completion_steps\.rs' README.md docs/certification.md 2>/dev/null; then
        echo "MISSING: the completion-step executor (crates/orchestrator/src/completion_steps.rs) is not referenced by README.md or docs/certification.md" >&2
        fail=1
    fi
fi

# 2. Windows Job Objects: AssignProcessToJobObject is real, so the docs may
#    not describe Job Objects as a future/target mechanism.
if grep -Rq 'AssignProcessToJobObject' crates/winjob/src 2>/dev/null; then
    for doc in "${TRUTH_DOCS[@]}"; do
        [ -f "$doc" ] || continue
        if normalized "$doc" | grep -Eqi 'job objects?[^.]{0,80}(future|target|planned|not yet|eventually|follow[- ]?up|todo)'; then
            echo "CONTRADICTION: $doc calls Windows Job Objects a future/target mechanism, but crates/winjob uses AssignProcessToJobObject" >&2
            fail=1
        fi
        if normalized "$doc" | grep -Eqi '(future|target|planned|not yet)[^.]{0,60}job objects?'; then
            echo "CONTRADICTION: $doc calls Windows Job Objects a future/target mechanism, but crates/winjob uses AssignProcessToJobObject" >&2
            fail=1
        fi
    done
    if ! grep -qi 'job object' README.md; then
        echo "MISSING: README.md does not document the Windows Job Object containment (crates/winjob)" >&2
        fail=1
    fi
fi

# 3. Vendored VS Code webview: ui/upstream.json means the upstream tree IS
#    vendored. Generic "not vendored" claims contradict it; only explicitly
#    JetBrains-7.1.2-scoped statements are allowed.
if [ -f ui/upstream.json ]; then
    for doc in "${TRUTH_DOCS[@]}"; do
        [ -f "$doc" ] || continue
        if normalized "$doc" | grep -q 'not vendored in this repo'; then
            echo "CONTRADICTION: $doc says the UI is 'not vendored in this repo', but ui/upstream.json exists (the v7.5.6 webview is vendored)" >&2
            fail=1
        fi
        while IFS= read -r fragment; do
            [ -z "$fragment" ] && continue
            case "$fragment" in
                *JetBrains*|*jetbrains*|*712*) ;;
                *)
                    echo "CONTRADICTION: $doc says the UI is not vendored without scoping the claim to JetBrains 7.1.2 while ui/upstream.json exists:${fragment}" >&2
                    fail=1
                    ;;
            esac
        done < <(normalized "$doc" | grep -oE '[^.]*not vendored[^.]*' || true)
    done
    if ! grep -q 'ui/upstream\.json' README.md; then
        echo "MISSING: README.md does not reference the vendored-UI manifest (ui/upstream.json)" >&2
        fail=1
    fi
fi

if [ "$fail" -ne 0 ]; then
    echo "$DOC / ${TRUTH_DOCS[*]} are out of sync with the code — fix the items listed above before merging." >&2
    exit 1
fi

echo "docs/architecture.md is in sync: no stale identifiers, current API names present."
echo "${TRUTH_DOCS[*]} pass the semantic truth assertions (completion execution, Windows Job Objects, vendored UI)."
