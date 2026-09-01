#!/usr/bin/env bash
# JetBrains split-mode smoke (no Gradle, no network):
#   1. build kilop-cli if missing
#   2. compile shared + backend + test + frontend with plain kotlinc
#   3. run BackendSmoke <binary> against the real daemon; exit with its code
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
JETBRAINS="$ROOT/apps/jetbrains"

SHARED_SRC="$JETBRAINS/shared/src/main/kotlin/dev/kilop/shared/Protocol.kt"
BACKEND_SRC="$JETBRAINS/backend/src/main/kotlin/dev/kilop/backend/BackendProcessManager.kt"
TEST_SRC="$JETBRAINS/backend/src/test/kotlin/dev/kilop/backend/BackendProcessManagerTest.kt"
FRONTEND_SRC="$JETBRAINS/frontend/src/main/kotlin/dev/kilop/frontend/PlaceholderFrontend.kt"

BIN="${KILOP_CLI_BIN:-$ROOT/target/debug/kilop-cli}"

echo "[compile-and-smoke] repo root: $ROOT"

# ---- 1. the CLI binary -----------------------------------------------------
if [ ! -x "$BIN" ]; then
  echo "[compile-and-smoke] building kilop-cli (missing: $BIN)"
  (cd "$ROOT" && cargo build -p kilop-cli) || {
    echo "FAIL: cargo build -p kilop-cli" >&2
    exit 1
  }
fi
if [ ! -x "$BIN" ]; then
  echo "FAIL: $BIN missing or not executable after build" >&2
  exit 1
fi

# ---- 2. kotlinc ------------------------------------------------------------
KOTLINC="${KOTLINC:-}"
if [ -z "$KOTLINC" ]; then
  KOTLINC="$(command -v kotlinc 2>/dev/null || true)"
fi
if [ -z "$KOTLINC" ]; then
  echo "FAIL: kotlinc not found (set KOTLINC or install kotlin)" >&2
  exit 1
fi

# ---- 3. kotlin-stdlib.jar (bundled with the compiler distribution) ---------
# The test file is dependency-free (plain check/require, no kotlin.test),
# so only the stdlib is needed on the compile classpath. Both homebrew
# (opt/kotlin/libexec/lib) and the Ubuntu apt package
# (/usr/share/java, /usr/share/kotlin/kotlinc/lib) ship it next to kotlinc.
resolve_symlink() {
  local p="$1"
  while [ -L "$p" ]; do
    local dir
    dir="$(cd "$(dirname "$p")" && pwd -P)"
    local target
    target="$(ls -ld "$p" | sed 's/.* -> //')"
    case "$target" in
      /*) p="$target" ;;
      *) p="$dir/$target" ;;
    esac
  done
  echo "$p"
}

find_kotlin_stdlib_jar() {
  local j
  for j in \
    "${KOTLIN_STDLIB_JAR:-}" \
    "$(command -v brew >/dev/null 2>&1 && brew --prefix kotlin 2>/dev/null)/libexec/lib/kotlin-stdlib.jar" \
    "$(dirname "$(resolve_symlink "$KOTLINC")")/../lib/kotlin-stdlib.jar" \
    /usr/share/kotlin/kotlinc/lib/kotlin-stdlib.jar \
    /usr/lib/kotlin/kotlinc/lib/kotlin-stdlib.jar \
    /opt/kotlin/kotlinc/lib/kotlin-stdlib.jar; do
    if [ -n "$j" ] && [ -f "$j" ]; then
      echo "$j"
      return 0
    fi
  done
  return 1
}

STDLIB_JAR="$(find_kotlin_stdlib_jar || true)"
if [ -z "$STDLIB_JAR" ]; then
  echo "FAIL: kotlin-stdlib.jar not found next to kotlinc" >&2
  exit 1
fi
echo "[compile-and-smoke] kotlin-stdlib.jar: $STDLIB_JAR"

# ---- 4. JDK fallback for old apt kotlinc -----------------------------------
# kotlinc 1.3.31 (Ubuntu apt) cannot read class files from JDK >= 16. When
# the plain compile fails and an older JDK (<= 12) is installed, retry the
# compiler on it. The compiled jar targets 1.8 and still runs on any JVM.
jdk_major() {
  local v
  v="$("$1/bin/java" -version 2>&1 | head -1)"
  v="${v#*\"}"
  v="${v%%\"*}"
  case "$v" in
    1.*) v="${v#1.}" ;;
  esac
  v="${v%%.*}"
  echo "$v"
}

newest_jdk_at_most() {
  local limit="$1" best="" d v
  for d in /usr/lib/jvm/*/; do
    [ -x "$d/bin/java" ] || continue
    v="$(jdk_major "$d")"
    case "$v" in
      '' | *[!0-9]*) continue ;;
    esac
    if [ "$v" -le "$limit" ] && { [ -z "$best" ] || [ "$v" -gt "$(jdk_major "$best")" ]; }; then
      best="$d"
    fi
  done
  echo "$best"
}

OLD_JDK="$(newest_jdk_at_most 12)"

kotlinc_cmd() {
  local jdk="$1"
  shift
  if [ -n "$jdk" ]; then
    JAVA_HOME="$jdk" PATH="$jdk/bin:$PATH" "$KOTLINC" "$@"
  else
    "$KOTLINC" "$@"
  fi
}

# ---- 5. compile -------------------------------------------------------------
WORK="$(mktemp -d "${TMPDIR:-/tmp}/kilop-jb-smoke.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
SMOKE_JAR="$WORK/smoke.jar"

compile_kotlin() {
  local out rc
  out="$(kotlinc_cmd "" -classpath "$STDLIB_JAR" -include-runtime -d "$SMOKE_JAR" \
    "$SHARED_SRC" "$BACKEND_SRC" "$TEST_SRC" "$FRONTEND_SRC" 2>&1)"
  rc=$?
  if [ $rc -ne 0 ] && [ -n "$OLD_JDK" ]; then
    echo "[compile-and-smoke] plain kotlinc failed; retrying with $OLD_JDK" >&2
    out="$(kotlinc_cmd "$OLD_JDK" -classpath "$STDLIB_JAR" -include-runtime -d "$SMOKE_JAR" \
      "$SHARED_SRC" "$BACKEND_SRC" "$TEST_SRC" "$FRONTEND_SRC" 2>&1)"
    rc=$?
  fi
  if [ $rc -ne 0 ]; then
    echo "$out" >&2
  fi
  return $rc
}

echo "[compile-and-smoke] kotlinc: $KOTLINC"
compile_kotlin || {
  echo "FAIL: kotlinc compilation" >&2
  exit 1
}

# ---- 6. smoke against the real daemon ---------------------------------------
echo "[compile-and-smoke] running BackendSmoke against $BIN"
java -cp "$SMOKE_JAR" dev.kilop.backend.BackendSmoke "$BIN"
exit $?
