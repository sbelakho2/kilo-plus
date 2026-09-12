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
# Every semantic assertion below is derived from a source artifact. A
# contradiction fails with the EXACT original line (line number + text) so
# the fix is unambiguous; when a wrapped phrase defeats line matching, the
# whitespace-normalized matching fragment is printed instead.
#
# Whitespace-normalized view of one doc (line wrapping must not hide a
# contradiction).
normalized() {
    tr '\n' ' ' < "$1" | tr -s ' ' | sed 's/7\.1\.2/712/g'
}

# Prints the first original lines matching an ERE (line numbers included).
raw_lines() {
    local doc="$1" re="$2"
    grep -n -i -E "$re" "$doc" 2>/dev/null | head -n 3 || true
}

# Prints the first normalized fragment matching an ERE (for wrapped claims).
normalized_fragment() {
    local doc="$1" re="$2"
    normalized "$doc" | grep -oE -i "[^.]{0,120}(${re})[^.]{0,120}" | head -n 1 || true
}

# Fails when a truth doc matches a contradiction regex; prints the evidence.
assert_no_contradiction() {
    local doc="$1" re="$2" message="$3" evidence
    [ -f "$doc" ] || return 0
    if normalized "$doc" | grep -Eqi "$re"; then
        echo "CONTRADICTION: $doc: $message" >&2
        evidence="$(raw_lines "$doc" "$re")"
        if [ -n "$evidence" ]; then
            echo "  exact line(s):" >&2
            printf '%s\n' "$evidence" >&2
        else
            echo "  wrapped claim: $(normalized_fragment "$doc" "$re")" >&2
        fi
        fail=1
    fi
}

# 1. Completion-step execution: crates/orchestrator/src/completion_steps.rs
#    exists and executes, so no truth doc may call execution a follow-up or
#    claim no automatic runner exists. ("follow-up" assertions only apply in
#    that case; the file IS the CompletionStepRunner.)
if [ -f crates/orchestrator/src/completion_steps.rs ]; then
    for doc in "${TRUTH_DOCS[@]}"; do
        assert_no_contradiction "$doc" \
            'execution[^.]{0,60}follow[- ]?up' \
            "still calls completion-step execution a follow-up, but crates/orchestrator/src/completion_steps.rs exists and executes"
        assert_no_contradiction "$doc" \
            '(no|without)[^.]{0,40}automatic[^.]{0,80}(commit|push|pr)[^.]{0,60}runner' \
            "claims there is no automatic commit/push/PR runner, but crates/orchestrator/src/completion_steps.rs exists and executes"
    done
    if ! grep -q 'completion_steps\.rs' README.md docs/certification.md 2>/dev/null; then
        echo "MISSING: the completion-step executor (crates/orchestrator/src/completion_steps.rs) is not referenced by README.md or docs/certification.md" >&2
        fail=1
    fi
fi

# 2. Windows Job Objects: the sources call CreateJobObjectW AND
#    AssignProcessToJobObject, so no truth doc may describe Job Objects as a
#    future/target mechanism.
if grep -Rq 'AssignProcessToJobObject' crates/winjob/src 2>/dev/null; then
    for doc in "${TRUTH_DOCS[@]}"; do
        assert_no_contradiction "$doc" \
            'job objects?[^.]{0,80}(future|target|planned|not yet|eventually|follow[- ]?up|todo)' \
            "calls Windows Job Objects a future/target mechanism, but crates/winjob uses AssignProcessToJobObject (and CreateJobObjectW)"
        assert_no_contradiction "$doc" \
            '(future|target|planned|not yet)[^.]{0,60}job objects?' \
            "calls Windows Job Objects a future/target mechanism, but crates/winjob uses AssignProcessToJobObject (and CreateJobObjectW)"
    done
    if ! grep -q 'CreateJobObjectW' README.md; then
        echo "MISSING: README.md does not name CreateJobObjectW, the constructor crates/winjob actually uses" >&2
        fail=1
    fi
    if ! grep -qi 'job object' README.md; then
        echo "MISSING: README.md does not document the Windows Job Object containment (crates/winjob)" >&2
        fail=1
    fi
fi

# 3. Vendored VS Code webview: ui/upstream.json means the upstream tree IS
#    vendored, and the VSIX staging step (prepackage:vsix) ships it under
#    media/. Generic "not vendored" claims contradict BOTH; only explicitly
#    JetBrains-7.1.2-scoped statements are allowed.
VSIX_STAGING=apps/vscode/scripts/prepare-vendored-webview.mjs
if [ -f ui/upstream.json ] && [ -f "$VSIX_STAGING" ]; then
    for doc in "${TRUTH_DOCS[@]}"; do
        [ -f "$doc" ] || continue
        doc_fail=0
        raw_count=0
        while IFS= read -r entry; do
            [ -z "$entry" ] && continue
            raw_count=$((raw_count + 1))
            lineno="${entry%%:*}"
            line="${entry#*:}"
            case "$line" in
                *JetBrains*|*jetbrains*|*7.1.2*|*712*) continue ;;
            esac
            echo "CONTRADICTION: $doc:$lineno says the UI is not vendored without a JetBrains-7.1.2 scope, but ui/upstream.json exists and prepackage:vsix stages the pinned closure + overlay into media/" >&2
            echo "  exact line: $line" >&2
            doc_fail=1
        done < <(grep -n -i 'not vendored' "$doc" || true)
        # A wrapped claim ("not\nvendored") evades line matching entirely;
        # catch it via the normalized view and print the matching fragment.
        if [ "$raw_count" -eq 0 ] && normalized "$doc" | grep -Eqi 'not[[:space:]]+vendored'; then
            echo "CONTRADICTION: $doc contains a wrapped 'not vendored' claim that no single line captured; normalized evidence follows" >&2
            echo "  wrapped claim: $(normalized_fragment "$doc" 'not[[:space:]]+vendored')" >&2
            doc_fail=1
        fi
        if [ "$doc_fail" -ne 0 ]; then
            fail=1
        fi
    done
    if ! grep -q 'ui/upstream\.json' README.md; then
        echo "MISSING: README.md does not reference the vendored-UI manifest (ui/upstream.json)" >&2
        fail=1
    fi
fi

# 4. apps/ real panels: when the rich Faktor panels exist (task tree,
#    tournament, board, blockers, evidence), no truth doc may describe
#    apps/ as scaffolding/stubs/placeholders.
if [ -f apps/jetbrains/frontend/src/main/kotlin/dev/faktor/frontend/BoardPanel.kt ] &&
    [ -f apps/jetbrains/frontend/src/main/kotlin/dev/faktor/frontend/TaskTreePanel.kt ] &&
    [ -f apps/jetbrains/frontend/src/main/kotlin/dev/faktor/frontend/TournamentPanel.kt ]; then
    for doc in "${TRUTH_DOCS[@]}"; do
        assert_no_contradiction "$doc" \
            'apps?/[^.]{0,100}(scaffold|scaffolding|stub|placeholder)' \
            "calls apps/ scaffolding/stubs, but the real Faktor panels exist (BoardPanel/TaskTreePanel/TournamentPanel)"
        assert_no_contradiction "$doc" \
            '(scaffold|scaffolding|stub|placeholder)[^.]{0,100}apps?/' \
            "calls apps/ scaffolding/stubs, but the real Faktor panels exist (BoardPanel/TaskTreePanel/TournamentPanel)"
    done
fi

if [ "$fail" -ne 0 ]; then
    echo "$DOC / ${TRUTH_DOCS[*]} are out of sync with the code — fix the items listed above before merging." >&2
    exit 1
fi

echo "docs/architecture.md is in sync: no stale identifiers, current API names present."
echo "${TRUTH_DOCS[*]} pass the semantic truth assertions (completion execution, Windows Job Objects, vendored UI + VSIX staging, real apps/ panels)."
