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

# Banda de capas cercanas al obstáculo sobre la que se promedia Jin para 1.3/1.4.
# Identificada del detalle Jin(S) en S ∈ [1.5, 5] m.
S_BAND_LO  = 2.0
S_BAND_HI  = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_runs(base_dir, N, k_str):
    """
    Devuelve directorios que contienen al menos energy.txt para (N, k).
    Soporta:
      - base/final_run/N{N}_[kK]{k}/                  (layout legado)
      - base/run_*/N{N}_[kK]{k}/                       (batch_run.sh)
      - base/N{N}_[kK]{k}/seed*/                       (run_system2.sh directo)
      - base/N{N}_[kK]{k}/                             (un único directorio por config)
    """
    k_int = int(float(k_str))
    patterns = [
        os.path.join(base_dir, "final_run", f"N{N}_K{k_str}"),
        os.path.join(base_dir, "final_run", f"N{N}_k{k_str}"),
        os.path.join(base_dir, "run_*",      f"N{N}_k{k_int}"),
        os.path.join(base_dir, "run_*",      f"N{N}_K{k_int}"),
        os.path.join(base_dir,               f"N{N}_k{k_int}", "seed*"),
        os.path.join(base_dir,               f"N{N}_K{k_int}", "seed*"),
        os.path.join(base_dir,               f"N{N}_k{k_int}"),
        os.path.join(base_dir,               f"N{N}_K{k_int}"),
    ]
    for pattern in patterns:
        dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
        if dirs:
            return dirs
    return []

def find_runs2(base_dir="system2"):
    """
    Mapea cada configuración 'n{N}_k{k}' a la lista de archivos states.txt
    asociados, soportando dos layouts:
      1) base/run_*/N{N}_k{k}/states.txt           (batch_run.sh)
      2) base/N{N}_k{k}/seed{S}/states.txt          (run_system2.sh directo)
    """
    base_path = Path(base_dir)
    config_groups = {}

    patterns = [
        "run_*/N*_[kK]*/states.txt",   # layout batch_run.sh (config_name = N{N}_k{k})
        "N*_[kK]*/seed*/states.txt",   # layout run_system2.sh (config_name = N{N}_k{k})
    ]
    for pattern in patterns:
        for file_path in base_path.glob(pattern):
            # En el layout 2, el padre es seed{S}; subimos uno más para obtener N{N}_k{k}.
            cfg_dir = file_path.parent
            if cfg_dir.name.lower().startswith("seed"):
                cfg_dir = cfg_dir.parent
            config_name = cfg_dir.name.lower()
            config_groups.setdefault(config_name, []).append(file_path)

    return config_groups


def find_seed_dirs(base_dir, N, k_str):
    """
    Devuelve todos los directorios de seed para (N, k), soportando dos layouts:
      1) base/run_*/N{N}_k{k}/cfc.txt       → la carpeta es la seed (una por run_X)
      2) base/N{N}_k{k}/seed{S}/cfc.txt     → cada seed es una carpeta
    """
    base_path = Path(base_dir)
    k_int = int(float(k_str))
    out = []
    # Layout 1
    for cfg in base_path.glob(f"run_*/N{N}_[kK]{k_int}"):
        if cfg.is_dir() and (cfg / "cfc.txt").exists():
            out.append(cfg)
    # Layout 2
    for seed_dir in base_path.glob(f"N{N}_[kK]{k_int}/seed*"):
        if seed_dir.is_dir() and (seed_dir / "cfc.txt").exists():
            out.append(seed_dir)
    return sorted(out)

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

def _timing_from_info(base_dir, k_str, N_filter=None):
    """
    Recolecta elapsed_ms por seed leyendo cada output/system2/N{N}_k{k}/seed*/info.txt
    y devuelve un DataFrame agregado (N, t_mean, t_std, n_seeds, t_min, t_max,
    seeds_t [lista]). Si N_filter está dado, se queda sólo con esos N.
    """
    base = Path(base_dir)
    k_int = int(float(k_str))
    allowed = set(N_filter) if N_filter is not None else None
    by_N = {}
    for info_path in base.glob(f"N*_[kK]{k_int}/seed*/info.txt"):
        try:
            info = load_info(str(info_path))
            N = int(info["N"])
            t = float(info["elapsed_ms"]) / 1000.0
        except (KeyError, ValueError):
            continue
        if allowed is not None and N not in allowed:
            continue
        by_N.setdefault(N, []).append(t)
    if not by_N:
        return None
    rows = []
    for N in sorted(by_N):
        ts = np.array(by_N[N])
        rows.append({
            "N": N,
            "t_mean": float(ts.mean()),
            "t_std":  float(ts.std(ddof=1) if len(ts) > 1 else 0.0),
            "t_min":  float(ts.min()),
            "t_max":  float(ts.max()),
            "n_seeds": len(ts),
            "seeds_t": ts.tolist(),
        })
    return pd.DataFrame(rows)


def plot_timing(timing_csv, out_dir, k_str, base_dir=None, N_list=None):
    """
    Si hay datos por seed (info.txt de runRealizations), grafica mean ± std de
    elapsed por N. Si no, cae al timing_csv del barrido -Nlist (1 seed por N).
    """
    df_agg = _timing_from_info(base_dir, k_str, N_filter=N_list) if base_dir else None

    fig, ax = plt.subplots(figsize=(7, 5))

    if df_agg is not None and (df_agg["n_seeds"] > 1).any():
        N      = df_agg["N"].values
        t_mean = df_agg["t_mean"].values
        t_std  = df_agg["t_std"].values

        # Promedio con barras de error (± std sobre seeds).
        ax.errorbar(N, t_mean, yerr=t_std, fmt="o-",
                    color="#3498db", ecolor="#2c3e50",
                    capsize=5, capthick=1.4, lw=1.8,
                    label="Tiempo de ejecución")

        # Tabla mean/std/min/max por N — para reporte y trazabilidad.
        df_agg.drop(columns=["seeds_t"]).to_csv(
            os.path.join(out_dir, f"timing_per_seed_k{k_str}.csv"), index=False)
    else:
        if not os.path.exists(timing_csv) or os.path.getsize(timing_csv) == 0:
            print(f"  Skipping timing plot: {timing_csv} ausente/vacío y no hay info.txt por seed.")
            plt.close(fig)
            return
        df = pd.read_csv(timing_csv)
        N  = df["N"].values
        t  = df["elapsed_ms"].values / 1e3
        ax.plot(N, t, "o-", color="#3498db", label="Tiempo de ejecución")

    ax.set_xlabel("N (número de partículas)")
    ax.set_ylabel("Tiempo de ejecución (s)")
    ax.set_title(f"Tiempo de ejecución vs N  (k={k_str} N/m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, f"timing_k{k_str}.png"))


# ═══════════════════════════════════════════════════════════════════════════════
# 1.2  Scanning rate J(N)
# ═══════════════════════════════════════════════════════════════════════════════

def J_from_cfc(time_vals, cfc_vals):
    """Pendiente J de la interpolación lineal de Cfc(t) para una realización."""
    if len(time_vals) < 2:
        return 0.0
    slope, _intercept, *_ = stats.linregress(time_vals, cfc_vals)
    return slope


def scanning_rate_vs_N(base_dir, N_list, k_str, out_dir):
    """
    Para cada N:
      - Lee cfc.txt de TODAS las seeds (un seed por carpeta run_*/N{N}_k{k}/).
      - Calcula J_s = pendiente(linear_fit(t, cfc)) por seed.
      - <J> = mean(J_s), σ_J = std(J_s) sobre seeds.
    Esto reemplaza la regresión ponderada sobre cfc promediado.
    """
    J_mean, J_std, Ns_ok = [], [], []

    for N in N_list:
        seed_dirs = find_seed_dirs(base_dir, N, k_str)
        if not seed_dirs:
            print(f"  No runs found for N={N}, k={k_str}")
            continue

        Js = []
        for d in seed_dirs:
            df = pd.read_csv(d / "cfc.txt")
            if len(df) < 2:
                continue
            Js.append(J_from_cfc(df["time"].values, df["cfc"].values))

        if not Js:
            continue

        Js = np.array(Js)
        J_mean.append(Js.mean())
        J_std.append(Js.std(ddof=1) if len(Js) > 1 else 0.0)
        Ns_ok.append(N)

    if not Ns_ok:
        print("No scanning-rate data found.")
        return None

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
# 1.2 (extra)  Cfc(t) por N — curvas acumuladas promediadas sobre seeds
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cfc_vs_t(base_dir, N_list, k_str, out_dir, n_grid=400):
    """
    Para cada N grafica el promedio sobre seeds de Cfc(t).
    cfc.txt es una lista de eventos (time, cfc_acumulado): se hace step-interp
    a una grilla uniforme común para poder promediar entre semillas.
    """
    cmap   = plt.cm.plasma
    colors = [cmap(i / max(len(N_list) - 1, 1)) for i in range(len(N_list))]

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False

    for N, color in zip(N_list, colors):
        seed_dirs = find_seed_dirs(base_dir, N, k_str)
        if not seed_dirs:
            continue

        per_seed = []
        t_max = 0.0
        for d in seed_dirs:
            df = pd.read_csv(d / "cfc.txt")
            if len(df) < 2:
                continue
            per_seed.append((df["time"].values, df["cfc"].values))
            t_max = max(t_max, df["time"].values[-1])

        if not per_seed:
            continue

        t_grid = np.linspace(0.0, t_max, n_grid)
        cfc_stack = []
        for tt, cc in per_seed:
            # Cfc es escalonada: en t=0 vale 0, salta a cc[k] a partir de tt[k].
            idx = np.searchsorted(tt, t_grid, side="right") - 1
            vals = np.where(idx >= 0, cc[np.clip(idx, 0, len(cc) - 1)], 0)
            cfc_stack.append(vals)

        cfc_mean = np.mean(cfc_stack, axis=0)
        ax.plot(t_grid, cfc_mean, color=color, lw=1.4, label=f"N={N}")
        plotted = True

    if not plotted:
        print("  Sin datos de Cfc(t) para graficar.")
        plt.close(fig)
        return

    ax.set_xlabel("t [s]")
    ax.set_ylabel("$C_{fc}(t)$")
    ax.set_title(f"$C_{{fc}}(t)$ acumulado (promedio sobre semillas)  k={k_str} N/m")
    ax.grid(True, alpha=0.3)
    ax.legend(title="N", fontsize=9, loc="best")
    savefig(fig, os.path.join(out_dir, f"cfc_vs_t_k{k_str}.png"))


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
            try:
                n = int(header)
            except ValueError:
                break
            time_line = f.readline()
            if not time_line:
                break
            # Una corrida abortada puede dejar la última frame truncada (filas con
                # < 6 columnas o que se cortan de golpe). Si detectamos eso, descartamos
                # el frame y salimos: lo importante es no romper la corrida entera.
            rows = []
            truncated = False
            for _ in range(n):
                parts = f.readline().split()
                if len(parts) != 6:
                    truncated = True
                    break
                rows.append([float(v) for v in parts])
            if truncated or len(rows) != n:
                print(f"    aviso: frame truncado en {states_path}, ignorado.")
                break
            arr = np.array(rows)

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
    # (N, jin_mean, jin_std, rho_mean, rho_std, v_mean, v_std) promediado en la
    # banda cercana al obstáculo S∈[S_BAND_LO, S_BAND_HI]. Usado por plot_band_vs_N
    # para cumplir la última parte de TP4 1.3.
    band_obs = []

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

        # Ploteo de líneas promedio macroscópicas (sin bandas de ±std).
        lbl = f"N={N}"
        ax_rho.plot(S, rho_avg,   color=color, lw=1.2, label=lbl)
        ax_v  .plot(S, v_avg,     color=color, lw=1.2, label=lbl)
        ax_jin.plot(S, jin_avg,   color=color, lw=1.2, label=lbl)

        near = (S >= 1.5) & (S <= 5.0)
        jin_near = jin_avg[near].mean() if near.any() else 0.0
        jin_near_obs.append((N, jin_near))

        # Banda cercana al obstáculo: promediamos por capa primero (sobre seeds),
        # y después promediamos las capas dentro de la banda. Reportamos también
        # el desvío sobre las seeds del promedio por banda para barras de error.
        band = (S >= S_BAND_LO) & (S <= S_BAND_HI)
        if band.any():
            jin_per_seed = [j[band].mean() for j in jins_run]
            rho_per_seed = [r[band].mean() for r in rhos_run]
            v_per_seed   = [v[band].mean() for v in vmeans_run]
            band_obs.append((
                N,
                float(np.mean(jin_per_seed)), float(np.std(jin_per_seed, ddof=1) if len(jin_per_seed) > 1 else 0.0),
                float(np.mean(rho_per_seed)), float(np.std(rho_per_seed, ddof=1) if len(rho_per_seed) > 1 else 0.0),
                float(np.mean(v_per_seed)),   float(np.std(v_per_seed,   ddof=1) if len(v_per_seed)   > 1 else 0.0),
            ))

    for ax, fig_, title, ylabel, tag in [
        (ax_rho, fig_rho, f"Densidad $\\langle\\rho_{{fin}}\\rangle$(S)  k={k_str}", "$\\rho$ (m$^{-2}$)",     "rho"),
        (ax_v,   fig_v,   f"Velocidad radial $\\langle v_{{fin}}\\rangle$(S)  k={k_str}", "|v| (m/s)", "vrad"),
        (ax_jin, fig_jin, f"Flujo $J_{{in}}$(S)  k={k_str}", "$J_{in}$ (m$^{-2}$ s$^{-1}$)",       "jin"),
    ]:
        ax.set_xlabel("S (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(title="N", fontsize=9, loc="best")
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
        
        ax_det.plot(S[near], jin_avg[near], color=color, lw=1.2, label=f"N={N}")

    ax_det.set_xlabel("S (m)")
    ax_det.set_ylabel("$J_{in}$ (m$^{-2}$ s$^{-1}$)")
    ax_det.set_title(f"Detalle Jin  S$\\in$[1.5,5] m  k={k_str}")
    ax_det.grid(True, alpha=0.3)
    ax_det.legend(title="N", fontsize=9, loc="best")
    savefig(fig_det, os.path.join(out_dir, f"jin_detail_k{k_str}.png"))

    # band_obs queda disponible para que main() lo pase a plot_band_vs_N
    # (cierre de TP4 1.3: <Jin>, <rho>, <v> vs N en la banda S∈[S_BAND_LO, S_BAND_HI]).
    return jin_near_obs, band_obs


# ═══════════════════════════════════════════════════════════════════════════════
# TP4 1.3 cierre: <Jin>, <rho>, <v> vs N promediados en la banda cercana al obstáculo
# ═══════════════════════════════════════════════════════════════════════════════

def plot_band_vs_N(band_obs, k_str, out_dir):
    """
    Cierre de TP4 1.3: una vez identificada la banda S∈[S_BAND_LO, S_BAND_HI] como
    la región donde el régimen cambia por la presencia del obstáculo (ver
    jin_detail_k{k}.png), promediamos las capas dentro de esa banda y graficamos
    <Jin>(N), <rho_fin>(N) y <v_fin>(N).

    Salida: dos figuras con doble eje y (Jin & rho, Jin & v) + CSV con los datos
    para que TP3 pueda comparar contra estos valores.
    """
    if not band_obs:
        print("  band_obs vacío: no se genera el plot de banda.")
        return None

    arr = np.array(band_obs)  # columnas: N, Jin, σJin, ρ, σρ, v, σv
    Ns          = arr[:, 0]
    jin, sj     = arr[:, 1], arr[:, 2]
    rho, sr     = arr[:, 3], arr[:, 4]
    v,   sv     = arr[:, 5], arr[:, 6]

    order = np.argsort(Ns)
    Ns, jin, sj, rho, sr, v, sv = Ns[order], jin[order], sj[order], rho[order], sr[order], v[order], sv[order]

    # Figura 1: Jin y rho con doble eje y.
    fig1, ax_j = plt.subplots(figsize=(8, 5))
    ax_r = ax_j.twinx()
    ax_j.errorbar(Ns, jin, yerr=sj, fmt="o-", color="#c0392b", capsize=4,
                  label="$\\langle J_{in}\\rangle$")
    ax_r.errorbar(Ns, rho, yerr=sr, fmt="s--", color="#2980b9", capsize=4,
                  label="$\\langle\\rho_{fin}\\rangle$")
    ax_j.set_xlabel("N")
    ax_j.set_ylabel("$\\langle J_{in}\\rangle$  [m$^{-2}$ s$^{-1}$]", color="#c0392b")
    ax_r.set_ylabel("$\\langle\\rho_{fin}\\rangle$  [m$^{-2}$]",     color="#2980b9")
    ax_j.tick_params(axis="y", colors="#c0392b")
    ax_r.tick_params(axis="y", colors="#2980b9")
    ax_j.set_title(f"TP4 1.3 — $\\langle J_{{in}}\\rangle$, $\\langle\\rho_{{fin}}\\rangle$ vs N "
                   f"(S∈[{S_BAND_LO},{S_BAND_HI}] m, k={k_str})")
    ax_j.grid(True, alpha=0.3)
    lines_j, labels_j = ax_j.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_j.legend(lines_j + lines_r, labels_j + labels_r, loc="best")
    savefig(fig1, os.path.join(out_dir, f"band_Jin_rho_vs_N_k{k_str}.png"))

    # Figura 2: Jin y v con doble eje y.
    fig2, ax_j2 = plt.subplots(figsize=(8, 5))
    ax_v = ax_j2.twinx()
    ax_j2.errorbar(Ns, jin, yerr=sj, fmt="o-", color="#c0392b", capsize=4,
                   label="$\\langle J_{in}\\rangle$")
    ax_v.errorbar(Ns, v,    yerr=sv, fmt="^--", color="#27ae60", capsize=4,
                  label="$|\\langle v_{fin}\\rangle|$")
    ax_j2.set_xlabel("N")
    ax_j2.set_ylabel("$\\langle J_{in}\\rangle$  [m$^{-2}$ s$^{-1}$]", color="#c0392b")
    ax_v .set_ylabel("$|\\langle v_{fin}\\rangle|$  [m/s]",            color="#27ae60")
    ax_j2.tick_params(axis="y", colors="#c0392b")
    ax_v .tick_params(axis="y", colors="#27ae60")
    ax_j2.set_title(f"TP4 1.3 — $\\langle J_{{in}}\\rangle$, $|\\langle v_{{fin}}\\rangle|$ vs N "
                    f"(S∈[{S_BAND_LO},{S_BAND_HI}] m, k={k_str})")
    ax_j2.grid(True, alpha=0.3)
    lines_j2, labels_j2 = ax_j2.get_legend_handles_labels()
    lines_v,  labels_v  = ax_v .get_legend_handles_labels()
    ax_j2.legend(lines_j2 + lines_v, labels_j2 + labels_v, loc="best")
    savefig(fig2, os.path.join(out_dir, f"band_Jin_v_vs_N_k{k_str}.png"))

    # CSV con los datos por N — útil para el informe y para comparar con TP3.
    pd.DataFrame({
        "N": Ns, "Jin": jin, "Jin_std": sj,
        "rho": rho, "rho_std": sr,
        "v": v, "v_std": sv,
    }).to_csv(os.path.join(out_dir, f"band_vs_N_k{k_str}.csv"), index=False)

    return Ns, jin, sj, rho, sr, v, sv


# ═══════════════════════════════════════════════════════════════════════════════
# TP4 1.3 — Comparación con TP3 (Comparar la curva de Jin con el TP3)
# ═══════════════════════════════════════════════════════════════════════════════

def _tp3_parse_output(filepath):
    """Mismo formato que TP3 analyze.py: bloques [t / x y vx vy ...]."""
    times, frames = [], []
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(lines):
        try:
            t = float(lines[i])
        except ValueError:
            i += 1
            continue
        i += 1
        frame = []
        while i < len(lines):
            parts = lines[i].split()
            if len(parts) == 4:
                frame.append([float(v) for v in parts])
                i += 1
            else:
                break
        if frame:
            times.append(t)
            frames.append(frame)
    return np.array(times), np.array(frames)


def _tp3_reconstruct_states(times, states, r_outer, r_inner, particle_radius, tol=0.5):
    """Reconstruye estado fresca(0)/usada(1) a partir de contactos con paredes."""
    T, N, _ = states.shape
    x, y   = states[:, :, 0], states[:, :, 1]
    dist   = np.sqrt(x ** 2 + y ** 2)
    contact_inner = np.abs(dist - (r_inner + particle_radius)) < tol
    contact_outer = np.abs(dist - (r_outer - particle_radius)) < tol
    state   = np.zeros((T, N), dtype=int)
    current = np.zeros(N, dtype=int)
    for t in range(T):
        for j in range(N):
            if contact_inner[t, j] and current[j] == 0:
                current[j] = 1
            elif contact_outer[t, j] and current[j] == 1:
                current[j] = 0
        state[t] = current
    return state


def _tp3_band_average(times, states, particle_state, r_inner=1.0, r_outer=40.0):
    """
    Para una corrida TP3, computa el promedio en la banda S∈[S_BAND_LO, S_BAND_HI]
    de rho, |v_radial| y Jin, usando la misma definición que TP4 (capas dS=DS,
    frescas con R·v < 0).
    """
    S_bins = np.arange(r_inner + DS / 2, r_outer, DS)
    counts   = np.zeros(len(S_bins))
    vrad_sum = np.zeros(len(S_bins))
    n_frames = states.shape[0]

    x, y   = states[:, :, 0], states[:, :, 1]
    vx, vy = states[:, :, 2], states[:, :, 3]
    dist   = np.sqrt(x ** 2 + y ** 2)
    rdotv  = x * vx + y * vy

    for t in range(n_frames):
        fresh = (particle_state[t] == 0)
        mask  = fresh & (rdotv[t] < 0)
        if not mask.any():
            continue
        d_in = dist[t][mask]
        v_in = np.abs(rdotv[t][mask] / np.maximum(d_in, 1e-15))
        for idx, s in enumerate(S_bins):
            lo, hi = s - DS / 2, s + DS / 2
            sh = (d_in >= lo) & (d_in < hi)
            counts[idx]   += sh.sum()
            vrad_sum[idx] += v_in[sh].sum()

    areas = np.array([shell_area(s) for s in S_bins])
    rho   = counts / max(n_frames, 1) / areas
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(counts > 0, vrad_sum / np.maximum(counts, 1), 0.0)
    jin = rho * v

    band = (S_bins >= S_BAND_LO) & (S_bins <= S_BAND_HI)
    if not band.any():
        return np.nan, np.nan, np.nan
    return float(jin[band].mean()), float(rho[band].mean()), float(v[band].mean())


def tp3_jin_vs_N(tp3_runs_dir, r_outer=40.0, r_inner=1.0, radius=1.0):
    """
    Recorre tp3_runs_dir/N{N}/run_*/output.txt, computa <Jin>, <rho>, <v> en la banda
    para cada seed, y agrega por N. Devuelve un DataFrame con columnas:
    N, Jin, Jin_std, rho, rho_std, v, v_std.
    """
    base = Path(tp3_runs_dir)
    if not base.is_dir():
        print(f"  TP3 runs dir no existe: {tp3_runs_dir}")
        return None

    rows = []
    n_dirs = sorted(base.glob("N*"))
    for nd in n_dirs:
        try:
            N = int(nd.name[1:])
        except ValueError:
            continue
        seed_jins, seed_rhos, seed_vs = [], [], []
        for run_dir in sorted(nd.glob("run_*")):
            out_path = run_dir / "output.txt"
            if not out_path.exists():
                continue
            times, states = _tp3_parse_output(str(out_path))
            if states.ndim != 3 or len(times) == 0:
                continue
            ps = _tp3_reconstruct_states(times, states, r_outer, r_inner, radius)
            j, r_, v_ = _tp3_band_average(times, states, ps, r_inner, r_outer)
            if np.isfinite(j):
                seed_jins.append(j); seed_rhos.append(r_); seed_vs.append(v_)
        if not seed_jins:
            continue
        rows.append({
            "N": N,
            "Jin":     np.mean(seed_jins),
            "Jin_std": np.std(seed_jins, ddof=1) if len(seed_jins) > 1 else 0.0,
            "rho":     np.mean(seed_rhos),
            "rho_std": np.std(seed_rhos, ddof=1) if len(seed_rhos) > 1 else 0.0,
            "v":       np.mean(seed_vs),
            "v_std":   np.std(seed_vs,   ddof=1) if len(seed_vs)   > 1 else 0.0,
        })
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def plot_tp3_comparison(band_tuple, tp3_df, k_str, out_dir):
    """
    Compara <Jin>(N) de TP4 (este k) contra <Jin>(N) de TP3 en la misma banda.
    band_tuple es lo que devuelve plot_band_vs_N.
    """
    if band_tuple is None or tp3_df is None or tp3_df.empty:
        print("  Sin datos para comparación TP3 (band_tuple o tp3_df vacío).")
        return
    Ns, jin, sj, *_ = band_tuple

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ns, jin, yerr=sj, fmt="o-", color="#c0392b", capsize=4,
                label=f"TP4 (k={k_str})")
    ax.errorbar(tp3_df["N"].values, tp3_df["Jin"].values, yerr=tp3_df["Jin_std"].values,
                fmt="s--", color="#2c3e50", capsize=4, label="TP3 (event-driven)")
    ax.set_xlabel("N")
    ax.set_ylabel("$\\langle J_{in}\\rangle$  [m$^{-2}$ s$^{-1}$]")
    ax.set_title(f"TP4 1.3 — Comparación $\\langle J_{{in}}\\rangle$(N) vs TP3 "
                 f"(S∈[{S_BAND_LO},{S_BAND_HI}] m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(fig, os.path.join(out_dir, f"band_Jin_vs_N_TP3_compare_k{k_str}.png"))

    # CSV con la tabla TP3 (para referencia y reproducibilidad).
    tp3_df.to_csv(os.path.join(out_dir, "TP3_band_vs_N.csv"), index=False)

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
    # Escalares característicos por k. Capturamos los DOS más relevantes:
    #   max_J[k]    = max_N <J>(N,k)           — magnitud del scanning rate óptimo
    #   N_star[k]   = argmax_N <J>(N,k)        — N que maximiza el scanning rate
    #   max_Jin[k]  = max_N <Jin|S_band>(N,k)
    max_J,    max_J_err    = [], []
    N_star_J                = []
    max_Jin,  max_Jin_err  = [], []
    N_star_Jin              = []

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    cmap = matplotlib.colormaps["viridis"]
    colors = [cmap(i / max(len(k_list)-1, 1)) for i in range(len(k_list))]

    for i, k_val in enumerate(k_values_sorted):
        k_str = str(int(k_val))
        csv_path = os.path.join(out_dir, f"J_vs_N_k{k_str}.csv")

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        # --- PANEL 0: <J>(N) por k ---
        ax[0].errorbar(df["N"], df["J_mean"], yerr=df["J_std"],
                       fmt="o-", color=colors[i], capsize=4, label=f"k={k_str}")
        idx_max_J = df["J_mean"].idxmax()
        max_J.append(df["J_mean"].iloc[idx_max_J])
        max_J_err.append(df["J_std"].iloc[idx_max_J])
        N_star_J.append(df["N"].iloc[idx_max_J])

        # --- PANEL 1: <Jin|S_band>(N) por k, promediado sobre capas cercanas al obstáculo ---
        jin_n_means, jin_n_stds = [], []
        for N in N_list:
            target_key = f"n{N}_k{int(k_val)}"
            run_files = config_groups.get(target_key, [])
            jins_seeds = []
            for file_path in run_files:
                S, _, _, jin = build_radial_profile(file_path)
                band = (S >= S_BAND_LO) & (S <= S_BAND_HI)
                if band.any():
                    jins_seeds.append(np.nanmean(jin[band]))
            if jins_seeds:
                jin_n_means.append(np.mean(jins_seeds))
                jin_n_stds .append(np.std(jins_seeds, ddof=1) if len(jins_seeds) > 1 else 0.0)
            else:
                jin_n_means.append(np.nan)
                jin_n_stds .append(0.0)

        jin_n_means = np.array(jin_n_means)
        jin_n_stds  = np.array(jin_n_stds)

        ax[1].errorbar(N_list, jin_n_means, yerr=jin_n_stds, fmt="s--",
                       color=colors[i], capsize=4, label=f"k={k_str}")

        if np.isfinite(jin_n_means).any():
            idx_max_Jin = int(np.nanargmax(jin_n_means))
            max_Jin    .append(jin_n_means[idx_max_Jin])
            max_Jin_err.append(jin_n_stds [idx_max_Jin])
            N_star_Jin .append(N_list[idx_max_Jin])
        else:
            max_Jin.append(np.nan); max_Jin_err.append(0.0); N_star_Jin.append(np.nan)

    ax[0].set_title("Scanning Rate $\\langle J \\rangle$ vs N")
    ax[0].set_ylabel("$\\langle J \\rangle$ [1/s]")
    ax[1].set_title(f"Flujo Radial $\\langle J_{{in}} \\rangle$ vs N  (S∈[{S_BAND_LO},{S_BAND_HI}] m)")
    ax[1].set_ylabel("$\\langle J_{in} \\rangle$  [m$^{-2}$ s$^{-1}$]")
    for a in ax:
        a.set_xlabel("N")
        a.legend()
        a.grid(True, alpha=0.3)
    savefig(fig, os.path.join(out_dir, "1_4_comparacion_curvas.png"))

    # 3. Escalares vs k — dos paneles consistentes: max(<·>) y N*(k).
    fig_esc, ax_esc = plt.subplots(1, 2, figsize=(14, 5))

    ax_esc[0].errorbar(k_values_sorted, max_J,   yerr=max_J_err,
                       fmt="o-", color="#c0392b", capsize=5, label="max $\\langle J \\rangle$")
    ax_esc[0].errorbar(k_values_sorted, max_Jin, yerr=max_Jin_err,
                       fmt="s-", color="#2980b9", capsize=5,
                       label=f"max $\\langle J_{{in}}|_{{S\\in[{S_BAND_LO},{S_BAND_HI}]}} \\rangle$")
    ax_esc[0].set_xscale("log")
    ax_esc[0].set_xlabel("k [N/m]")
    ax_esc[0].set_ylabel("Máximo de la curva")
    ax_esc[0].set_title("max($\\cdot$) vs k")
    ax_esc[0].grid(True, which="both", alpha=0.3)
    ax_esc[0].legend()

    ax_esc[1].plot(k_values_sorted, N_star_J,   "o-", color="#c0392b", label="$N^*$ de $\\langle J \\rangle$")
    ax_esc[1].plot(k_values_sorted, N_star_Jin, "s-", color="#2980b9", label="$N^*$ de $\\langle J_{in} \\rangle$")
    ax_esc[1].set_xscale("log")
    ax_esc[1].set_xlabel("k [N/m]")
    ax_esc[1].set_ylabel("$N^*$")
    ax_esc[1].set_title("$N^*(k)$")
    ax_esc[1].grid(True, which="both", alpha=0.3)
    ax_esc[1].legend()

    savefig(fig_esc, os.path.join(out_dir, "1_4_escalar_vs_k.png"))

    # Guardar tabla resumen
    pd.DataFrame({
        "k": k_values_sorted,
        "max_J": max_J, "max_J_err": max_J_err, "N_star_J": N_star_J,
        "max_Jin": max_Jin, "max_Jin_err": max_Jin_err, "N_star_Jin": N_star_Jin,
    }).to_csv(os.path.join(out_dir, "1_4_escalares.csv"), index=False)


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
    ap.add_argument("--tp3-runs", default=None,
                    help="Path al directorio runs/ del TP3 (event-driven). Si se "
                         "pasa, se computa <Jin>(N) de TP3 en la misma banda y se "
                         "genera el plot de comparación pedido en TP4 1.3.")
    args = ap.parse_args()

    N_list = [int(x) for x in args.Nlist.split(",")]
    k_list = [float(x) for x in args.k.split(",")]
    
    os.makedirs(args.out, exist_ok=True)

    for k_val in k_list:
        k_str = str(int(k_val))
        print(f"\n── k = {k_str} N/m ──────────────────────────────")

        # Timing — usa info.txt por seed si hay >1 seed por N (mean±std);
        # si no, cae al CSV del barrido -Nlist. Si tampoco existe, se omite.
        timing_csv = os.path.join(args.base, f"timing_k{k_str}.csv")
        print("Plotting timing…")
        plot_timing(timing_csv, args.out, k_str, base_dir=args.base, N_list=N_list)

        # Energy validation (one example run)
        plot_energy(args.base, N_list[0], k_str, args.out)

        # Scanning rate
        print("Computing scanning rate…")
        scanning_rate_vs_N(args.base, N_list, k_str, args.out)

        # Curvas Cfc(t) por N
        print("Plotting Cfc(t) por N…")
        plot_cfc_vs_t(args.base, N_list, k_str, args.out)

        # Radial profiles
        print("Building radial profiles con bandas de error…")
        _, band_obs = plot_radial_profiles(args.base, N_list, k_str, args.out, seed=args.seed)

        # Cierre de TP4 1.3: <Jin>, <rho>, <v> vs N en la banda cercana al obstáculo.
        print("Promedios en banda (S∈[%s,%s]) vs N…" % (S_BAND_LO, S_BAND_HI))
        band_tuple = plot_band_vs_N(band_obs, k_str, args.out)

        # Comparación contra TP3 (si se pasó --tp3-runs).
        if args.tp3_runs:
            print(f"Computando referencia TP3 desde {args.tp3_runs} …")
            tp3_df = tp3_jin_vs_N(args.tp3_runs)
            plot_tp3_comparison(band_tuple, tp3_df, k_str, args.out)

    # Cross-k comparison
    if len(k_list) > 1:
        print("\nComparing k values con barras de error cruzadas…")
        run_task_1_4_analysis(args.base, k_list, N_list, args.out)

    print("\nAll done.")


if __name__ == "__main__":
    main()