#!/usr/bin/env bash
# Usage:
#   ./run_system2.sh                          # single run N=200, k=1000
#   ./run_system2.sh -N 300 -seeds 42,43,44  # 3 realizations
#   ./run_system2.sh -Nlist 100,200,300,500   # timing sweep
#   ./run_system2.sh -Nlist 100,200,500,1000 -k 100 -seeds 42
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
K=1000
NLIST=200

# ── Parse known flags we need to forward / inspect ────────────────────────────
JAVA_ARGS=()
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -k)
      K="$2"; JAVA_ARGS+=("-k" "$2"); shift 2 ;;
    -Nlist)
      NLIST="$2"; JAVA_ARGS+=("-Nlist" "$2"); shift 2 ;;
    -N)
      NLIST="$2"; JAVA_ARGS+=("-N" "$2"); shift 2 ;;
    *)
      JAVA_ARGS+=("$1"); EXTRA_ARGS+=("$1"); shift ;;
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

#echo "=== Analysing ==="
#python3 visualizer/analyze_system2.py \
#    --base   output/system2 \
#    --k      "$K" \
#    --Nlist  "$NLIST" \
#    --out    output/system2/plots

echo "Done. See output/system2/"