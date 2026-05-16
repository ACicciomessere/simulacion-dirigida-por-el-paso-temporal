#!/usr/bin/env python3
"""
analyze_system2.py  –  System 2 analysis
Covers TP4 tasks 1.1 through 1.4:
  1.1  Execution time vs N
  1.2  Scanning rate J(N) from linear fit to Cfc(t)
  1.3  Radial profiles <rho_fin>(S), <v_fin>(S), Jin(S)
  1.4  Comparison across k values

Usage:
    python analyze_system2.py --base output/system2 --k 100,1000,10000 --Nlist 100,200,300,500
"""

import argparse, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

R_DOMAIN   = 40.0
R_OBSTACLE = 1.0
R_PARTICLE = 1.0
DS         = 0.2   # radial shell width


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_runs(base_dir, N, k_str):
    """Return list of run directories for given N and k."""
    patterns = [
        os.path.join(base_dir, "final_run", f"N{N}_K{k_str}"),   
        os.path.join(base_dir, "final_run", f"N{N}_k{k_str}"),    
    ]
    for pattern in patterns:
        dirs = sorted(glob.glob(pattern))
        if dirs:
            return dirs
    return []

def find_runs2(base_dir="system2"):
    base_path = Path(base_dir)
    config_groups = {}
    
    for file_path in base_path.glob("run_*/N*_[kK]*/states.txt"):
        config_name = file_path.parent.name.lower()
        
        if config_name not in config_groups:
            config_groups[config_name] = []
        
        config_groups[config_name].append(file_path)
        
    return config_groups

def load_cfc(path):
    df = pd.read_csv(path)
    return df["time"].values, df["cfc"].values


def load_info(path):
    info = {}
    with open(path) as f:
        for line in f:
            k, v = line.strip().split("=")
            info[k] = v
    return info


def parse_frames(states_path):
    """Generator: yields (time, ndarray[N,6]) for each frame in states.txt."""
    with open(states_path) as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            n     = int(header)
            t     = float(f.readline().strip().split("=")[1])
            rows  = [list(map(float, f.readline().split())) for _ in range(n)]
            yield t, np.array(rows)


def savefig(fig, path, tight=True):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.1  Execution time vs N
# ═══════════════════════════════════════════════════════════════════════════════

def plot_timing(timing_csv, out_dir, k_str):
    df = pd.read_csv(timing_csv)
    N  = df["N"].values
    t  = df["elapsed_ms"].values / 1e3

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(N, t, "o-", color="#3498db", label="TP4 (Tiempo discreto)")

    # Fit power law
    if len(np.unique(N)) >= 2:
        log_n, log_t = np.log(N), np.log(np.maximum(t, 1e-12)) 
        slope, intercept, *_ = stats.linregress(log_n, log_t)
        ax.plot(N, np.exp(intercept) * N**slope, "--", color="#7f8c8d",
                alpha=0.7, label=f"Ajuste: $N^{{{slope:.2f}}}$")

    ax.set_xlabel("N (número de partículas)")
    ax.set_ylabel("Tiempo de ejecución (s)")
    ax.set_title(f"Tiempo de ejecución vs N  (k={k_str} N/m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, f"timing_k{k_str}.png"))


# ═══════════════════════════════════════════════════════════════════════════════
# 1.2  Scanning rate J(N)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_J_weighted(cfc_vals, time_vals, time_err):
    """
    Regresión lineal ponderada: cfc en función de time.
    Usa 1/time_err^2 como pesos para propagar el error real entre corridas.
    """
    # Ignorar puntos sin error (time_err == 0 da peso infinito)
    mask = time_err > 0
    if mask.sum() < 2:
        return 0.0, 0.0

    w = 1.0 / time_err[mask]**2
    x = time_vals[mask]
    y = cfc_vals[mask]

    # Fórmulas de mínimos cuadrados ponderados
    sw   = w.sum()
    swx  = (w * x).sum()
    swx2 = (w * x**2).sum()
    swy  = (w * y).sum()
    swxy = (w * x * y).sum()

    denom = sw * swx2 - swx**2
    slope = (sw * swxy - swx * swy) / denom
    slope_err = np.sqrt(sw / denom)   # error del slope propagado

    return slope, slope_err


def scanning_rate_vs_N(base_dir, N_list, k_str, out_dir):
    J_mean, J_std, Ns_ok = [], [], []

    for N in N_list:
        runs = find_runs(base_dir, N, k_str)
        if not runs:
            print(f"  No runs found for N={N}, k={k_str}")
            continue

        cfc_path = os.path.join(runs[0], "cfc.txt")  # solo hay uno en final_run
        if not os.path.exists(cfc_path):
            continue

        df = pd.read_csv(cfc_path)
        if len(df) < 2:
            continue

        slope, slope_err = compute_J_weighted(
            df["cfc"].values,
            df["time"].values,
            df["time_err"].values
        )

        J_mean.append(slope)
        J_std.append(slope_err)   # error propagado desde time_err, no std entre runs
        Ns_ok.append(N)
        if not Ns_ok:
            print("No scanning-rate data found.")
            return

    Ns_ok  = np.array(Ns_ok)
    J_mean = np.array(J_mean)
    J_std  = np.array(J_std)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(Ns_ok, J_mean, yerr=J_std, fmt="o-", capsize=4,
                color="#e67e22", label=f"<J> (k={k_str})")
    ax.set_xlabel("N")
    ax.set_ylabel("Scanning rate J  [1/s]")
    ax.set_title(f"Scanning rate vs N  (k={k_str} N/m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, f"scanning_rate_k{k_str}.png"))

    # Save CSV for task 1.4 comparison
    df_out = pd.DataFrame({"N": Ns_ok, "J_mean": J_mean, "J_std": J_std})
    df_out.to_csv(os.path.join(out_dir, f"J_vs_N_k{k_str}.csv"), index=False)

    return Ns_ok, J_mean, J_std


# ═══════════════════════════════════════════════════════════════════════════════
# Energy validation
# ═══════════════════════════════════════════════════════════════════════════════

def plot_energy(base_dir, N, k_str, out_dir, seed=42):
    runs = find_runs(base_dir, N, k_str)
    if not runs:
        return
    run_dir  = runs[0] 
    ene_path = os.path.join(run_dir, "energy.txt")
    if not os.path.exists(ene_path):
        return
    df = pd.read_csv(ene_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["time"], df["energy"], lw=0.8, color="#2c3e50")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Energía total (J)")
    ax.set_title(f"Energía total vs tiempo  N={N}, k={k_str}")
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, f"energy_N{N}_k{k_str}.png"))

# ═══════════════════════════════════════════════════════════════════════════════
# 1.3  Radial profiles
# ═══════════════════════════════════════════════════════════════════════════════

def shell_area(s_center):
    """Area of annular shell [s_center-DS/2, s_center+DS/2]."""
    r_out = s_center + DS / 2
    r_in  = max(0.0, s_center - DS / 2)
    return np.pi * (r_out**2 - r_in**2)


def build_radial_profile(states_path):
    """Calculates spatial features for a single simulation run."""
    S_max   = R_DOMAIN
    S_bins  = np.arange(R_OBSTACLE + DS / 2, S_max, DS)
    counts  = np.zeros(len(S_bins))
    vrad_sum= np.zeros(len(S_bins))
    n_frames = 0

    with open(states_path) as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            n     = int(header)
            t     = float(f.readline().strip().split("=")[1])
            arr   = np.array([list(map(float, f.readline().split())) for _ in range(n)])

            fresh = arr[arr[:, 5] == 0]
            if len(fresh) == 0:
                n_frames += 1
                continue

            x, y = fresh[:, 0], fresh[:, 1]
            vx, vy = fresh[:, 2], fresh[:, 3]
            dist = np.sqrt(x**2 + y**2)
            xdotv = x * vx + y * vy

            mask  = xdotv < 0
            dist_in = dist[mask]
            xdotv_in = xdotv[mask]

            if len(dist_in) == 0:
                n_frames += 1
                continue

            vrad = xdotv_in / dist_in

            for idx, s in enumerate(S_bins):
                s_lo, s_hi = s - DS / 2, s + DS / 2
                shell_mask = (dist_in >= s_lo) & (dist_in < s_hi)
                counts[idx]   += shell_mask.sum()
                vrad_sum[idx] += np.sum(np.abs(vrad[shell_mask]))

            n_frames += 1

    if n_frames == 0:
        return S_bins, np.zeros_like(S_bins), np.zeros_like(S_bins), np.zeros_like(S_bins)

    areas = np.array([shell_area(s) for s in S_bins])
    rho   = counts / (n_frames * areas)
    with np.errstate(invalid="ignore", divide="ignore"):
        vrad_mean = np.where(counts > 0, vrad_sum / np.maximum(counts, 1), 0.0)
    jin = rho * vrad_mean

    return S_bins, rho, vrad_mean, jin


def plot_radial_profiles(base_dir, N_list, k_str, out_dir, seed=42):
    cmap   = plt.cm.plasma
    colors = [cmap(i / max(len(N_list) - 1, 1)) for i in range(len(N_list))]

    fig_rho, ax_rho = plt.subplots(figsize=(8, 5))
    fig_v,   ax_v   = plt.subplots(figsize=(8, 5))
    fig_jin, ax_jin = plt.subplots(figsize=(8, 5))

    jin_near_obs = []

    # Cargamos el mapa completo de corridas una sola vez para este análisis radial
    config_groups = find_runs2(base_dir)

    for N, color in zip(N_list, colors):
        # Buscamos en el diccionario usando la clave normalizada en minúsculas
        target_key = f"n{N}_k{int(float(k_str))}"
        run_files = config_groups.get(target_key, [])
        
        if not run_files:
            print(f"  No runs found for N={N}, k={k_str} (Key: {target_key})")
            continue

        rhos_run, vmeans_run, jins_run = [], [], []
        S = None

        # Procesamos de manera exacta los estados microscópicos de cada corrida por separado
        for states_path in run_files:
            S, rho, vmean, jin = build_radial_profile(states_path)
            rhos_run.append(rho)
            vmeans_run.append(vmean)
            jins_run.append(jin)

        if not rhos_run:
            print(f"  Missing states.txt data for N={N}")
            continue

        # Promedios y errores estadísticos (Desviación estándar de las muestras por semilla)
        rho_avg, rho_std = np.mean(rhos_run, axis=0), np.std(rhos_run, axis=0)
        v_avg, v_std     = np.mean(vmeans_run, axis=0), np.std(vmeans_run, axis=0)
        jin_avg, jin_std = np.mean(jins_run, axis=0), np.std(jins_run, axis=0)

        # Ploteo de líneas promedio macroscópicas
        ax_rho.plot(S, rho_avg,   color=color, lw=1.2)
        ax_v  .plot(S, v_avg,     color=color, lw=1.2)
        ax_jin.plot(S, jin_avg,   color=color, lw=1.2)

        # Región sombreada con transparencia representando la dispersión real de las semillas
        if len(rhos_run) > 1:
            ax_rho.fill_between(S, rho_avg - rho_std, rho_avg + rho_std, color=color, alpha=0.15)
            ax_v  .fill_between(S, v_avg - v_std, v_avg + v_std, color=color, alpha=0.15)
            ax_jin.fill_between(S, jin_avg - jin_std, jin_avg + jin_std, color=color, alpha=0.15)

        near = (S >= 1.5) & (S <= 5.0)
        jin_near = jin_avg[near].mean() if near.any() else 0.0
        jin_near_obs.append((N, jin_near))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(N_list), vmax=max(N_list)))
    sm.set_array([])

    for ax, fig_, title, ylabel, tag in [
        (ax_rho, fig_rho, f"Densidad $\\langle\\rho_{{fin}}\\rangle$(S)  k={k_str}", "$\\rho$ (m$^{-2}$)",     "rho"),
        (ax_v,   fig_v,   f"Velocidad radial $\\langle v_{{fin}}\\rangle$(S)  k={k_str}", "|v| (m/s)", "vrad"),
        (ax_jin, fig_jin, f"Flujo $J_{{in}}$(S)  k={k_str}", "$J_{in}$ (m$^{-2}$ s$^{-1}$)",       "jin"),
    ]:
        ax.set_xlabel("S (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig_.colorbar(sm, ax=ax, label="N")
        savefig(fig_, os.path.join(out_dir, f"radial_{tag}_k{k_str}.png"))

    # Gráfico de Detalle Jin incorporando bandas de error
    fig_det, ax_det = plt.subplots(figsize=(8, 5))
    for N, color in zip(N_list, colors):
        target_key = f"n{N}_k{int(float(k_str))}"
        run_files = config_groups.get(target_key, [])
        jins_run = []
        
        for r_path in run_files:
            S, _, _, jin = build_radial_profile(r_path)
            jins_run.append(jin)
        
        if not jins_run: continue
        jin_avg, jin_std = np.mean(jins_run, axis=0), np.std(jins_run, axis=0)
        near = (S >= 1.5) & (S <= 5.0)
        
        ax_det.plot(S[near], jin_avg[near], color=color, lw=1.2)
        if len(jins_run) > 1:
            ax_det.fill_between(S[near], jin_avg[near] - jin_std[near], jin_avg[near] + jin_std[near], color=color, alpha=0.15)

    ax_det.set_xlabel("S (m)")
    ax_det.set_ylabel("$J_{in}$ (m$^{-2}$ s$^{-1}$)")
    ax_det.set_title(f"Detalle Jin  S$\\in$[1.5,5] m  k={k_str}")
    ax_det.grid(True, alpha=0.3)
    fig_det.colorbar(sm, ax=ax_det, label="N")
    savefig(fig_det, os.path.join(out_dir, f"jin_detail_k{k_str}.png"))

    return jin_near_obs

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE ESCALADO 
# ═══════════════════════════════════════════════════════════════════════════════

def get_characteristic_value(df_j):
    idx_max = df_j["J_mean"].idxmax()
    n_star = df_j["N"].iloc[idx_max]
    return n_star 


# ═══════════════════════════════════════════════════════════════════════════════
# TAREA 1.4: COMPARACIÓN DE K
# ═══════════════════════════════════════════════════════════════════════════════

def run_task_1_4_analysis(base_dir, k_list, N_list, out_dir):
    # Instanciamos el mapa completo de archivos una sola vez
    config_groups = find_runs2(base_dir)

    max_k = max(k_list)
    max_n = max(N_list)
    target_max_key = f"n{max_n}_k{int(max_k)}"
    
    # Si esa clave existe en nuestro diccionario, significa que hay carpetas para graficar la energía
    if target_max_key in config_groups:
        # Le pasamos la primera ruta de states.txt que encontremos para esa configuración
        # (O si plot_energy necesita el directorio padre, usamos .parent)
        plot_energy(base_dir, max_n, str(int(max_k)), out_dir)

    k_values_sorted = sorted(k_list)
    escalares_j = []
    escalares_j_err = []
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i / max(len(k_list)-1, 1)) for i in range(len(k_list))]

    for i, k_val in enumerate(k_values_sorted):
        k_str = str(int(k_val))
        csv_path = os.path.join(out_dir, f"J_vs_N_k{k_str}.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # --- PANEL 0: CURVA J(N) (Se mantiene leyendo del CSV de la tarea 1.2) ---
            ax[0].errorbar(df["N"], df["J_mean"], yerr=df["J_std"], 
                           fmt="o-", color=colors[i], capsize=4, label=f"k={k_str}")
            
            idx_max = df["J_mean"].idxmax() 
            escalares_j.append(df["J_mean"].iloc[idx_max])   
            escalares_j_err.append(df["J_std"].iloc[idx_max])

            # --- PANEL 1: CURVA Jin(N) en S ~ 2 con barras de error reales de semilla ---
            jin_n_means = []
            jin_n_stds = []
            
            for N in N_list:
                target_key = f"n{N}_k{int(k_val)}"
                run_files = config_groups.get(target_key, [])
                jins_seeds = []
                
                for file_path in run_files:
                    S, _, _, jin = build_radial_profile(file_path)
                    # Encontramos la posición espacial más cercana a S = 2.0 metros
                    idx_s2 = np.abs(S - 2.0).argmin()
                    jins_seeds.append(jin[idx_s2])
                
                if jins_seeds:
                    jin_n_means.append(np.mean(jins_seeds))
                    jin_n_stds.append(np.std(jins_seeds) if len(jins_seeds) > 1 else 0.0)
                else:
                    jin_n_means.append(0.0)
                    jin_n_stds.append(0.0)
            
            if len(jin_n_means) == len(N_list):
                # Aplicamos barras de error (capsize) explícitas representando la desviación estándar real
                ax[1].errorbar(N_list, jin_n_means, yerr=jin_n_stds, fmt="s--", 
                               color=colors[i], capsize=4, label=f"k={k_str}")

    ax[0].set_title("Scanning Rate $\\langle J \\rangle$ vs N")
    ax[0].set_ylabel("$\\langle J \\rangle$ [1/s]")
    ax[1].set_title("Flujo Radial $\\langle J_{in} |_{S \\approx 2} \\rangle$ vs N")
    ax[1].set_ylabel("Flujo en S=2")
    for a in ax:
        a.set_xlabel("N")
        a.legend()
        a.grid(True, alpha=0.3)
    
    savefig(fig, os.path.join(out_dir, "1_4_comparacion_curvas.png"))

    # 3. Gráfico del ESCALAR vs k
    fig_esc, ax_esc = plt.subplots(figsize=(8, 6))
    ax_esc.errorbar(k_values_sorted, escalares_j, yerr=escalares_j_err,
                fmt="ro-", capsize=5, label="$N^*$ característico")
    ax_esc.set_xscale("log")
    ax_esc.set_xlabel("Constante elástica k [N/m]")
    ax_esc.set_ylabel("$J^*$ [1/s]")
    ax_esc.set_title("Evolución del parámetro característico vs k")
    ax_esc.grid(True, which="both", alpha=0.3)
    ax_esc.legend()
    
    savefig(fig_esc, os.path.join(out_dir, "1_4_escalar_vs_k.png"))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",   default="output/system2")
    ap.add_argument("--k",      default="1000",  help="comma-separated k values, e.g. 100,1000,10000")
    ap.add_argument("--Nlist",  default="100,200,300,500", help="comma-separated N values to analyse")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--out",    default="output/system2/plots")
    args = ap.parse_args()

    N_list = [int(x) for x in args.Nlist.split(",")]
    k_list = [float(x) for x in args.k.split(",")]
    os.makedirs(args.out, exist_ok=True)

    for k_val in k_list:
        k_str = str(int(k_val))
        print(f"\n── k = {k_str} N/m ──────────────────────────────")

        # Timing
        timing_csv = os.path.join(args.base, f"timing_k{k_str}.csv")
        if os.path.exists(timing_csv):
            print("Plotting timing…")
            plot_timing(timing_csv, args.out, k_str)

        # Energy validation (one example run)
        plot_energy(args.base, N_list[0], k_str, args.out)

        # Scanning rate
        print("Computing scanning rate…")
        scanning_rate_vs_N(args.base, N_list, k_str, args.out)

        # Radial profiles
        print("Building radial profiles con bandas de error…")
        plot_radial_profiles(args.base, N_list, k_str, args.out, seed=args.seed)

    # Cross-k comparison
    if len(k_list) > 1:
        print("\nComparing k values con barras de error cruzadas…")
        run_task_1_4_analysis(args.base, k_list, N_list, args.out)

    print("\nAll done.")


if __name__ == "__main__":
    main()