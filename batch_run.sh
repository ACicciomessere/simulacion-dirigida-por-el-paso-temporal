#!/usr/bin/env bash
set -e

BASE_DIR="output/system2"
SEEDS=${1:-42}

# 1. Buscar el próximo número de run disponible
RUN_NUM=1
while [[ -d "$BASE_DIR/run_$RUN_NUM" ]]; do
  ((RUN_NUM++))
done

TARGET_DIR="$BASE_DIR/run_$RUN_NUM"
mkdir -p "$TARGET_DIR"

echo "=== Iniciando tanda de simulación: $TARGET_DIR (Seeds: $SEEDS) ==="

K_VALUES=(100 1000 10000 100000)
N_LIST="100,200,300,500,600,800,1000"

for K in "${K_VALUES[@]}"; do
  echo "--- Ejecutando para k=$K ---"
  ./run_system2.sh -Nlist "$N_LIST" -k "$K" -seeds "$SEEDS"

  # Mover archivos sueltos
  find "$BASE_DIR" -maxdepth 1 -type f ! -name ".*" -exec mv {} "$TARGET_DIR/" \;

  # Mover subcarpetas generadas, excluyendo las run_N
  for subdir in "$BASE_DIR"/*/; do
    dirname=$(basename "$subdir")
    if [[ "$dirname" =~ ^run_[0-9]+$ ]] || [[ "$subdir" == "$TARGET_DIR/" ]]; then
      continue
    fi
    echo "  Moviendo contenido de $dirname -> $TARGET_DIR/"
    mv "$subdir"/* "$TARGET_DIR/" 2>/dev/null || true
    rmdir "$subdir"  # Elimina la carpeta si quedó vacía
  done
done

echo "=== Tanda finalizada. Todos los resultados en $TARGET_DIR ==="