#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
K=1000
NLIST=200

# ── Parse flags ───────────────────────────────────────────────────────────────
JAVA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k) K="$2"; JAVA_ARGS+=("-k" "$2"); shift 2 ;;
    -Nlist) NLIST="$2"; JAVA_ARGS+=("-Nlist" "$2"); shift 2 ;;
    -N) NLIST="$2"; JAVA_ARGS+=("-N" "$2"); shift 2 ;;
    *) JAVA_ARGS+=("$1"); shift ;;
  esac
done

echo "=== Compiling ==="
mvn -q package -DskipTests

echo "=== Running System 2 (k=${K}) ==="
java -cp target/tp4-molecular-dynamics-1.0-SNAPSHOT.jar \
    ar.edu.itba.sds.system2.System2Main \
    -dt 0.001 -dt2 1.0 -tf 500 \
    -out output/system2 \
    "${JAVA_ARGS[@]}"

echo "Done. output/system2/timing updated."