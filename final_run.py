#!/usr/bin/env python3
import os
import glob
import re
import shutil
import pandas as pd

BASE_DIR = "output/system2"
FINAL_DIR = os.path.join(BASE_DIR, "final_run")
os.makedirs(FINAL_DIR, exist_ok=True)

# 1. Buscar y ordenar numéricamente todas las carpetas run_N
run_folders = sorted(
    glob.glob(os.path.join(BASE_DIR, "run_*")),
    key=lambda x: int(x.split('_')[-1])
)

if not run_folders:
    print(f"No se encontraron carpetas 'run_N' en {BASE_DIR}.")
    exit(1)

print(f"Encontradas {len(run_folders)} corridas para procesar.")

# 2. Mapear archivos por (subcarpeta_NXXX_kXXX, nombre_archivo)
files_map = {}
for run_folder in run_folders:
    for subfolder in glob.glob(os.path.join(run_folder, "*")):
        if not os.path.isdir(subfolder):
            continue
        subfolder_name = os.path.basename(subfolder)
        for filepath in glob.glob(os.path.join(subfolder, "*")):
            filename = os.path.basename(filepath)
            key = (subfolder_name, filename)
            if key not in files_map:
                files_map[key] = []
            files_map[key].append(filepath)

# 3. Función de ordenamiento estricto: N (Descendente), K (Ascendente), Filename
def get_sort_key(item):
    (subfolder_name, filename), _ = item
    match = re.match(r"N(\d+)_[kK](\d+)", subfolder_name)
    if match:
        n_val = int(match.group(1))
        k_val = int(match.group(2))
        # -n_val logra el orden descendente (1000, 800, 600...)
        return (-n_val, k_val, filename)
    return (0, 0, filename)

# 4. Procesar cada combinación de forma ordenada
for (subfolder_name, filename), paths in sorted(files_map.items(), key=get_sort_key):
    out_subfolder = os.path.join(FINAL_DIR, subfolder_name)
    os.makedirs(out_subfolder, exist_ok=True)
    output_path = os.path.join(out_subfolder, filename)

    # Si es info o states, clonamos el de la corrida 1 para el visualizador
    if filename in ["info.txt", "states.txt"]:
        print(f"Copiando estructura: {subfolder_name}/{filename}")
        shutil.copy2(paths[0], output_path)
        continue

    print(f"Promediando: {subfolder_name}/{filename} ({len(paths)} corridas)")
    all_dfs = []

    for path in paths:
        try:
            df = pd.read_csv(path)
            all_dfs.append(df)
        except Exception as e:
            print(f"  [Aviso] No se pudo leer {path}: {e}")

    if not all_dfs:
        continue

    df_concat = pd.concat(all_dfs, ignore_index=True)

    # Definir columna de agrupación fija según el archivo analizado
    if filename == "cfc.txt":
        group_col = "cfc"
    elif filename == "energy.txt":
        group_col = "time"
    else:
        group_col = df_concat.columns[0]

    # Asegurar que los datos métricos sean numéricos
    metric_cols = [c for c in df_concat.columns if c != group_col]
    for col in metric_cols:
        df_concat[col] = pd.to_numeric(df_concat[col], errors='coerce')

    grouped = df_concat.groupby(group_col)
    
    summary_data = {group_col: []}
    for col in metric_cols:
        summary_data[col] = []
        summary_data[f"{col}_err"] = []

    for name, group in grouped:
        summary_data[group_col].append(name)
        for col in metric_cols:
            summary_data[col].append(group[col].mean())
            std_val = group[col].std()
            summary_data[f"{col}_err"].append(std_val if not pd.isna(std_val) else 0.0)

    df_final = pd.DataFrame(summary_data)
    df_final.to_csv(output_path, index=False)
    print(f"  -> Guardado en: {output_path}")

print("\nProcesamiento completo de forma ordenada.")