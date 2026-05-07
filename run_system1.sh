#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "=== Compiling ==="
mvn -q package -DskipTests

echo "=== Running System 1 ==="
java -cp target/tp4-molecular-dynamics-1.0-SNAPSHOT.jar \
    ar.edu.itba.sds.system1.OscillatorSimulation \
    output/system1

echo "=== Plotting System 1 ==="
python3 visualizer/plot_oscillator.py \
    --traj output/system1/trajectories.csv \
    --mse  output/system1/mse_vs_dt.csv \
    --out  output/system1

echo "Done. See output/system1/"
