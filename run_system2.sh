#!/usr/bin/env bash
# Usage:
#   ./run_system2.sh                          # single run N=200, k=1000
#   ./run_system2.sh -N 300 -seeds 42,43,44  # 3 realizations
#   ./run_system2.sh -Nlist 100,200,300,500   # timing sweep
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== Compiling ==="
mvn -q package -DskipTests

echo "=== Running System 2 ==="
java -cp target/tp4-molecular-dynamics-1.0-SNAPSHOT.jar \
    ar.edu.itba.sds.system2.System2Main \
    -dt 0.001 -dt2 1.0 -tf 500 -k 1000 \
    -out output/system2 \
    "$@"

echo "=== Analysing ==="
python3 visualizer/analyze_system2.py \
    --base output/system2 \
    --k    1000 \
    --Nlist 200 \
    --out  output/system2/plots

echo "Done. See output/system2/"
