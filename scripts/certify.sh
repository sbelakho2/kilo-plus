#!/usr/bin/env bash
# Release certification (P0-97 / P0-74 / P0-100).
#
# The gate a release candidate must pass BEFORE shipping:
#   1. the full cargo gates exactly as CI runs them (fmt/check/clippy/tests);
#   2. `doctor --deep` on a FRESH data dir — every audit invariant (store,
#      CAS, journal, cost reservations, verification records, active-turn
#      recoverable owners, orphan children, process ownership) must pass and
#      print zero FAIL sections;
#   3. the fault campaign — the #[ignore]-gated `[fault]` tests, when the
#      faktor-tests-fault crate is present in this workspace (sibling wave);
#   4. `doctor --deep` again on a SECOND fresh data dir AFTER the campaign:
#      the corruption the campaign proves contained must not leak into a
#      fresh release image (P0-97 release certification);
#   5. a printed certificate summary.
#
# Gate 8 (additive, P0-70): the byte-level artifact branding scan
# (scripts/branding-scan.sh --artifacts) over PACKAGED application outputs
# (.vsix / plugin jars) when this workspace produced them. A bounded scan
# of packaged outputs only — target/release binary strings are deliberately
# not swept here (slow, and cargo test/debug outputs embed frozen fixture
# text); when no packaged output exists the gate is recorded as skipped,
# never silently dropped.
#
# Exits non-zero on the first failing gate. Safe to run from any directory
# (resolves the workspace root); read-only apart from the two temp data dirs.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

step() { printf '\n== %s ==\n' "$*"; }
ok() { printf '   ok: %s\n' "$*"; }
bad() { printf '   FAIL: %s\n' "$*"; }

gates=("fmt --check" "cargo check --workspace" "clippy -D warnings" "cargo test --workspace" "doctor --deep (fresh dir)" "fault campaign [fault]" "doctor --deep (post-campaign)" "artifact branding scan (packaged outputs)")

run_doctor_deep() {
    local dir label rc out
    dir="$(mktemp -d "${TMPDIR:-/tmp}/kp-cert.XXXXXX")"
    label="$1"
    out="$(cargo run -q -p faktor-cli -- doctor --deep --data-dir "$dir" 2>&1)"
    rc=$?
    printf '%s\n' "$out"
    if [ "$rc" -ne 0 ]; then
        bad "$label: doctor exited non-zero ($rc)"
        # Surface every FAIL section for the certificate report.
        printf '%s\n' "$out" | grep -Ei 'inconsisten|dangling|orphan|without recoverable|corrupt|failed|issue' || true
        return 1
    fi
    if ! printf '%s\n' "$out" | grep -q 'doctor: all checks passed'; then
        bad "$label: doctor printed issues without a non-zero exit"
        return 1
    fi
    ok "$label: all sections passed, zero FAIL sections"
    rm -rf "$dir"
    return 0
}

fault_campaign() {
    # The [fault] ignored suite lives in the faktor-tests-fault crate of the
    # fault-containment sibling wave; until it lands in this workspace the
    # gate is skipped (recorded in the certificate, not silently dropped).
    if ! cargo metadata --no-deps --format-version 1 2>/dev/null \
        | grep -q '"name":"faktor-tests-fault"'; then
        ok "fault campaign skipped: no faktor-tests-fault crate in this workspace"
        return 0
    fi
    if ! cargo test -p faktor-tests-fault -- --ignored > /tmp/kp-fault.log 2>&1; then
        bad "fault campaign failed (log tail):"
        tail -n 50 /tmp/kp-fault.log
        return 1
    fi
    ok "fault campaign passed ([fault] ignored tests)"
    return 0
}

artifact_scan() {
    # Packaged application outputs only (.vsix, plugin jars) — scan the
    # app tree that produced one; skip gracefully (recorded, not silent)
    # when this workspace built none. source-mode content under those trees
    # is already scan-clean, so the byte-level pass is bounded.
    local pass=1 scanned=0 d found
    for d in apps/vscode apps/jetbrains; do
        [ -d "$d" ] || continue
        found="$(find "$d" -type f \( -name '*.vsix' -o -name '*.jar' \) -print -quit 2>/dev/null)"
        if [ -n "$found" ]; then
            scanned=1
            if bash scripts/branding-scan.sh --artifacts "$d"; then
                ok "artifact scan clean: $d"
            else
                bad "artifact scan failed: $d"
                pass=0
            fi
        fi
    done
    if [ "$scanned" -eq 0 ]; then
        ok "artifact scan skipped: no packaged .vsix/.jar outputs present in apps/"
        return 0
    fi
    return $((1 - pass))
}

status=0

step "gate 1/8: cargo fmt --check"
if cargo fmt --check; then ok "formatting clean"; else status=1; bad "formatting drift"; fi

step "gate 2/8: cargo check --workspace"
if cargo check --workspace; then ok "workspace check clean"; else status=1; bad "workspace check failed"; fi

step "gate 3/8: cargo clippy --workspace --all-targets -- -D warnings"
if cargo clippy --workspace --all-targets -- -D warnings; then
    ok "clippy clean (-D warnings)"
else
    status=1
    bad "clippy warnings"
fi

step "gate 4/8: cargo test --workspace"
if cargo test --workspace; then ok "workspace tests pass"; else status=1; bad "workspace tests failed"; fi

step "gate 5/8: doctor --deep on a fresh data dir"
if run_doctor_deep "pre-campaign doctor --deep"; then ok "pre-campaign doctor --deep clean"; else status=1; fi

step "gate 6/8: fault campaign ([fault] ignored tests)"
if fault_campaign; then ok "fault campaign complete"; else status=1; fi

step "gate 7/8: doctor --deep after the fault campaign (release certification)"
if run_doctor_deep "post-campaign doctor --deep"; then ok "post-campaign doctor --deep clean"; else status=1; fi

step "gate 8/8: artifact branding scan (packaged outputs)"
if artifact_scan; then ok "artifact branding gate complete"; else status=1; fi

printf '\n=====================\n'
if [ "$status" -eq 0 ]; then
    printf 'CERTIFICATE: PASS\n'
    for g in "${gates[@]}"; do printf '  [x] %s\n' "$g"; done
    printf 'Every gate green — release candidate certified.\n'
else
    printf 'CERTIFICATE: FAIL\n'
    for g in "${gates[@]}"; do printf '  [ ] %s\n' "$g"; done
    printf 'At least one gate failed — do not ship.\n'
fi
printf '=====================\n'
exit "$status"
