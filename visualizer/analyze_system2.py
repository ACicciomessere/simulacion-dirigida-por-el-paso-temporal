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

import argparse, glob, os, re
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
    Devuelve directorios de corridas para (N, k). Soporta varios layouts:
      base/final_run/N{N}_[kK]{k}/, base/run_*/N{N}_[kK]{k}/,
      base/N{N}_[kK]{k}/seed*/, base/N{N}_[kK]{k}/.
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
    Mapea cada configuración 'n{N}_k{k}' a la lista de archivos states.txt.
    Soporta dos layouts:
      1) base/run_*/N{N}_k{k}/states.txt           (batch_run.sh)
      2) base/N{N}_k{k}/seed{S}/states.txt          (run_system2.sh directo)
    """
    base_path = Path(base_dir)
    config_groups = {}
    patterns = [
        "run_*/N*_[kK]*/states.txt",
        "N*_[kK]*/seed*/states.txt",
    ]
    for pattern in patterns:
        for file_path in base_path.glob(pattern):
            cfg_dir = file_path.parent
            if cfg_dir.name.lower().startswith("seed"):
                cfg_dir = cfg_dir.parent
            config_name = cfg_dir.name.lower()
            config_groups.setdefault(config_name, []).append(file_path)
    return config_groups


def find_seed_dirs(base_dir, N, k_str):
    """
    Devuelve los directorios de seed para (N, k). Soporta:
      1) base/run_*/N{N}_k{k}/cfc.txt
      2) base/N{N}_k{k}/seed{S}/cfc.txt
    """
    base_path = Path(base_dir)
    k_int = int(float(k_str))
    out = []
    for cfg in base_path.glob(f"run_*/N{N}_[kK]{k_int}"):
        if cfg.is_dir() and (cfg / "cfc.txt").exists():
            out.append(cfg)
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


def plot_energy_dt_sweep(sweep_dir, out_dir, N=None, k_str=None):
    """
    Overlay de E_total(t) para distintos dt — validación del paso de integración
    (TP4 spec: "verificar que el dt utilizado es el adecuado graficando la
    evolución temporal de la energía total").

    Lee `sweep_dir/dt*/N{N}_k{k}/energy.txt`. El dt se parsea del nombre del
    directorio (ej. `dt0.001` o `dt_0.001`). Si N/k_str son None toma los
    primeros que encuentre. Cada curva queda etiquetada con su dt.
    """
    base = Path(sweep_dir)
    if not base.is_dir():
        print(f"  Skip dt-sweep: no existe {sweep_dir}")
        return

    # Detectar carpetas dt*.
    dt_dirs = sorted(base.glob("dt*"))
    if not dt_dirs:
        print(f"  Skip dt-sweep: no se encontraron subcarpetas dt* en {sweep_dir}")
        return

    series = []  # (dt_value, time_array, energy_array)
    for d in dt_dirs:
        # Parsear el valor de dt del nombre de la carpeta.
        m = re.search(r"dt[_]?([0-9.]+)", d.name)
        if not m:
            continue
        try:
            dt_val = float(m.group(1))
        except ValueError:
            continue

        # Si N o k no se especifican, tomar la primera config que aparezca.
        cfg_glob = f"N{N}_[kK]{int(float(k_str))}" if (N and k_str) else "N*_[kK]*"
        # energy.txt puede estar directamente en N{N}_k{k}/ (layout antiguo)
        # o en N{N}_k{k}/seed*/ (layout run_system2.sh directo).
        candidate_paths = []
        for cfg in d.glob(cfg_glob):
            candidate_paths.append(cfg / "energy.txt")
            candidate_paths.extend(sorted(cfg.glob("seed*/energy.txt")))
        ene_path = next((p for p in candidate_paths if p.exists()), None)
        if ene_path is None:
            continue
        df = pd.read_csv(ene_path)
        series.append((dt_val, df["time"].values, df["energy"].values))

    if not series:
        print(f"  Skip dt-sweep: no se encontraron energy.txt válidos en {sweep_dir}")
        return

    series.sort(key=lambda s: s[0], reverse=True)  # dt más grande primero (más ruidoso al fondo)

    fig, ax = plt.subplots(figsize=(11, 5))
    for dt_val, t, e in series:
        # Format dt para legend: usar la representación más corta sin ceros sobrantes.
        label = f"dt={('%g' % dt_val)}"
        ax.plot(t, e, lw=0.8, label=label)

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Energía total (J)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    savefig(fig, os.path.join(out_dir, "energy_dt_sweep.png"))


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
    # (N, jin_mean, jin_std, rho_mean, rho_std, v_mean, v_std) promediado en la
    # banda cercana al obstáculo S∈[S_BAND_LO, S_BAND_HI]. Lo usa plot_band_vs_N
    # para cerrar TP4 1.3 y la comparación contra TP3.
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

        # Línea promedio sobre semillas.
        ax_rho.plot(S, rho_avg,   color=color, lw=1.2)
        ax_v  .plot(S, v_avg,     color=color, lw=1.2)
        ax_jin.plot(S, jin_avg,   color=color, lw=1.2)

        # Banda ±std (gradient) sobre las semillas.
        if len(rhos_run) > 1:
            ax_rho.fill_between(S, rho_avg - rho_std, rho_avg + rho_std, color=color, alpha=0.15)
            ax_v  .fill_between(S, v_avg   - v_std,   v_avg   + v_std,   color=color, alpha=0.15)
            ax_jin.fill_between(S, jin_avg - jin_std, jin_avg + jin_std, color=color, alpha=0.15)

        near = (S >= 1.5) & (S <= 5.0)
        jin_near = jin_avg[near].mean() if near.any() else 0.0
        jin_near_obs.append((N, jin_near))

        # Banda cercana al obstáculo (consigna): promediar por capa primero (sobre
        # seeds) y después promediar las capas dentro de la banda. Se mantiene
        # también el desvío sobre semillas para barras de error en vs-N.
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
            ax_det.fill_between(S[near], jin_avg[near] - jin_std[near],
                                jin_avg[near] + jin_std[near], color=color, alpha=0.15)

    ax_det.set_xlabel("S (m)")
    ax_det.set_ylabel("$J_{in}$ (m$^{-2}$ s$^{-1}$)")
    ax_det.set_title(f"Detalle Jin  S$\\in$[1.5,5] m  k={k_str}")
    ax_det.grid(True, alpha=0.3)
    fig_det.colorbar(sm, ax=ax_det, label="N")
    savefig(fig_det, os.path.join(out_dir, f"jin_detail_k{k_str}.png"))

    return jin_near_obs, band_obs


# ═══════════════════════════════════════════════════════════════════════════════
# TP4 1.3 cierre: <Jin>, <rho>, <v> vs N en la banda cercana al obstáculo
# ═══════════════════════════════════════════════════════════════════════════════

def plot_band_vs_N(band_obs, k_str, out_dir):
    """
    Cierre del item 1.3: una vez identificada la banda S∈[S_BAND_LO, S_BAND_HI]
    como la región donde el régimen cambia por el obstáculo, promediar las capas
    dentro de esa banda y graficar <Jin>(N), <rho_fin>(N) y <v_fin>(N).
    """
    if not band_obs:
        print("  band_obs vacío: no se genera el plot de banda.")
        return None

    arr = np.array(band_obs)
    Ns          = arr[:, 0]
    rho, sr     = arr[:, 3], arr[:, 4]
    v,   sv     = arr[:, 5], arr[:, 6]
    order = np.argsort(Ns)
    Ns, rho, sr, v, sv = Ns[order], rho[order], sr[order], v[order], sv[order]
    # Jin se reconstruye como el producto de los promedios <rho>·<|v|>, sin
    # volver a promediar el producto. La incertidumbre se propaga linealmente
    # (primer orden): σ_J / J ≈ σ_ρ/ρ + σ_v/v.
    jin = rho * v
    with np.errstate(invalid="ignore", divide="ignore"):
        sj  = np.where((rho > 0) & (v > 0),
                       jin * (sr / np.maximum(rho, 1e-15) + sv / np.maximum(v, 1e-15)),
                       0.0)

    # Figura 1: Jin & rho (doble eje y).
    fig1, ax_j = plt.subplots(figsize=(8, 5))
    ax_r = ax_j.twinx()
    ax_j.errorbar(Ns, jin, yerr=sj, fmt="o-", color="#c0392b", capsize=4,
                  label="$J_{in}$")
    ax_r.errorbar(Ns, rho, yerr=sr, fmt="s--", color="#2980b9", capsize=4,
                  label="$\\langle\\rho_{fin}\\rangle$")
    ax_j.set_xlabel("N")
    ax_j.set_ylabel("$J_{in}$  [m$^{-2}$ s$^{-1}$]", color="#c0392b")
    ax_r.set_ylabel("$\\langle\\rho_{fin}\\rangle$  [m$^{-2}$]",     color="#2980b9")
    ax_j.tick_params(axis="y", colors="#c0392b")
    ax_r.tick_params(axis="y", colors="#2980b9")
    ax_j.set_title(f"TP4 1.3 — $\\langle J_{{in}}\\rangle$, $\\langle\\rho_{{fin}}\\rangle$ vs N "
                   f"(S∈[{S_BAND_LO},{S_BAND_HI}] m, k={k_str})")
    ax_j.grid(True, alpha=0.3)
    lj, ljl = ax_j.get_legend_handles_labels()
    lr, lrl = ax_r.get_legend_handles_labels()
    ax_j.legend(lj + lr, ljl + lrl, loc="best")
    savefig(fig1, os.path.join(out_dir, f"band_Jin_rho_vs_N_k{k_str}.png"))

    # Figura 2: Jin & v (doble eje y).
    fig2, ax_j2 = plt.subplots(figsize=(8, 5))
    ax_v = ax_j2.twinx()
    ax_j2.errorbar(Ns, jin, yerr=sj, fmt="o-", color="#c0392b", capsize=4,
                   label="$J_{in}$")
    ax_v .errorbar(Ns, v,    yerr=sv, fmt="^--", color="#27ae60", capsize=4,
                   label="$|\\langle v_{fin}\\rangle|$")
    ax_j2.set_xlabel("N")
    ax_j2.set_ylabel("$J_{in}$  [m$^{-2}$ s$^{-1}$]", color="#c0392b")
    ax_v .set_ylabel("$|\\langle v_{fin}\\rangle|$  [m/s]",            color="#27ae60")
    ax_j2.tick_params(axis="y", colors="#c0392b")
    ax_v .tick_params(axis="y", colors="#27ae60")
    ax_j2.set_title(f"TP4 1.3 — $\\langle J_{{in}}\\rangle$, $|\\langle v_{{fin}}\\rangle|$ vs N "
                    f"(S∈[{S_BAND_LO},{S_BAND_HI}] m, k={k_str})")
    ax_j2.grid(True, alpha=0.3)
    lj2, lj2l = ax_j2.get_legend_handles_labels()
    lv,  lvl  = ax_v .get_legend_handles_labels()
    ax_j2.legend(lj2 + lv, lj2l + lvl, loc="best")
    savefig(fig2, os.path.join(out_dir, f"band_Jin_v_vs_N_k{k_str}.png"))

    pd.DataFrame({
        "N": Ns, "Jin": jin, "Jin_std": sj,
        "rho": rho, "rho_std": sr,
        "v": v, "v_std": sv,
    }).to_csv(os.path.join(out_dir, f"band_vs_N_k{k_str}.csv"), index=False)

    return Ns, jin, sj, rho, sr, v, sv


# ═══════════════════════════════════════════════════════════════════════════════
# TP4 1.3 — Comparación con TP3
# ═══════════════════════════════════════════════════════════════════════════════

def _tp3_parse_output(filepath):
    """Formato TP3 (4 o 5 columnas): bloques [t / x y vx vy [fresh]]."""
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
            if len(parts) in (4, 5):
                frame.append([float(v) for v in parts])
                i += 1
            else:
                break
        if frame:
            times.append(t)
            frames.append(frame)
    return np.array(times), np.array(frames)


def _tp3_reconstruct_states(times, states, r_outer, r_inner, particle_radius, tol=0.5):
    """Si hay 5 cols usa el flag fresh; si no, reconstruye por proximidad."""
    T, N, ncols = states.shape
    if ncols >= 5:
        # columna 4 del TP3 = 1 si fresca; pasamos a 0=fresca, 1=usada
        return (1 - states[:, :, 4]).astype(int)
    x, y = states[:, :, 0], states[:, :, 1]
    dist = np.sqrt(x ** 2 + y ** 2)
    ci   = np.abs(dist - (r_inner + particle_radius)) < tol
    co   = np.abs(dist - (r_outer - particle_radius)) < tol
    state, cur = np.zeros((T, N), dtype=int), np.zeros(N, dtype=int)
    for t in range(T):
        for j in range(N):
            if ci[t, j] and cur[j] == 0:
                cur[j] = 1
            elif co[t, j] and cur[j] == 1:
                cur[j] = 0
        state[t] = cur
    return state


def _tp3_band_average(times, states, particle_state, r_inner=1.0, r_outer=40.0):
    """Promedio en la banda S∈[S_BAND_LO, S_BAND_HI] para una corrida TP3."""
    S_bins = np.arange(r_inner + DS / 2, r_outer, DS)
    counts   = np.zeros(len(S_bins))
    vrad_sum = np.zeros(len(S_bins))
    n_frames = states.shape[0]
    x, y   = states[:, :, 0], states[:, :, 1]
    vx, vy = states[:, :, 2], states[:, :, 3]
    dist   = np.sqrt(x ** 2 + y ** 2)
    rdotv  = x * vx + y * vy
    for t in range(n_frames):
        mask = (particle_state[t] == 0) & (rdotv[t] < 0)
        if not mask.any():
            continue
        d_in = dist[t][mask]
        v_in = np.abs(rdotv[t][mask] / np.maximum(d_in, 1e-15))
        for idx, s in enumerate(S_bins):
            lo, hi = s - DS / 2, s + DS / 2
            sh = (d_in >= lo) & (d_in < hi)
            counts[idx]   += sh.sum()
            vrad_sum[idx] += v_in[sh].sum()
    areas = np.pi * ((S_bins + DS / 2) ** 2 - (S_bins - DS / 2) ** 2)
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
    Recorre {tp3_runs_dir}/N{N}/run_*/output.txt, computa <Jin>, <rho>, <v> en la
    banda por semilla, y devuelve un DataFrame con mean/std por N.
    """
    base = Path(tp3_runs_dir)
    if not base.is_dir():
        print(f"  TP3 runs dir no existe: {tp3_runs_dir}")
        return None
    rows = []
    for nd in sorted(base.glob("N*")):
        try:
            N = int(nd.name[1:])
        except ValueError:
            continue
        seed_j, seed_r, seed_v = [], [], []
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
                seed_j.append(j); seed_r.append(r_); seed_v.append(v_)
        if not seed_j:
            continue
        rows.append({
            "N": N,
            "Jin":     np.mean(seed_j),
            "Jin_std": np.std(seed_j, ddof=1) if len(seed_j) > 1 else 0.0,
            "rho":     np.mean(seed_r),
            "rho_std": np.std(seed_r, ddof=1) if len(seed_r) > 1 else 0.0,
            "v":       np.mean(seed_v),
            "v_std":   np.std(seed_v,   ddof=1) if len(seed_v)   > 1 else 0.0,
        })
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True) if rows else None


def plot_tp3_comparison(band_tuple, tp3_df, k_str, out_dir):
    """Overlay de <Jin>(N) TP4 (k actual) vs TP3 en la misma banda."""
    if band_tuple is None or tp3_df is None or tp3_df.empty:
        print("  Sin datos para comparación TP3.")
        return
    Ns, jin, sj, *_ = band_tuple
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(Ns, jin, yerr=sj, fmt="o-", color="#c0392b", capsize=4,
                label=f"TP4 (k={k_str})")
    ax.errorbar(tp3_df["N"].values, tp3_df["Jin"].values, yerr=tp3_df["Jin_std"].values,
                fmt="s--", color="#2c3e50", capsize=4, label="TP3 (event-driven)")
    ax.set_xlabel("N")
    ax.set_ylabel("$J_{in}$  [m$^{-2}$ s$^{-1}$]")
    ax.set_title(f"TP4 1.3 — Comparación $\\langle J_{{in}}\\rangle$(N) vs TP3 "
                 f"(S∈[{S_BAND_LO},{S_BAND_HI}] m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(fig, os.path.join(out_dir, f"band_Jin_vs_N_TP3_compare_k{k_str}.png"))
    tp3_df.to_csv(os.path.join(out_dir, "TP3_band_vs_N.csv"), index=False)

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
        exp = int(np.log10(k_val))
        ax[0].errorbar(df["N"], df["J_mean"], yerr=df["J_std"],
                       fmt="o-", color=colors[i], capsize=4, label=f"$k=10^{{{exp}}}$")
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

        ax[1].errorbar(N_list, jin_n_means, yerr=jin_n_stds, fmt="o-",
                       color=colors[i], capsize=4, label=f"$k=10^{{{exp}}}$")

        if np.isfinite(jin_n_means).any():
            idx_max_Jin = int(np.nanargmax(jin_n_means))
            max_Jin    .append(jin_n_means[idx_max_Jin])
            max_Jin_err.append(jin_n_stds [idx_max_Jin])
            N_star_Jin .append(N_list[idx_max_Jin])
        else:
            max_Jin.append(np.nan); max_Jin_err.append(0.0); N_star_Jin.append(np.nan)

    ax[0].set_title("Scanning Rate $\\langle J \\rangle$ vs N")
    ax[0].set_ylabel("$\\langle J \\rangle$ [1/s]")
    ax[1].set_title(f"Flujo Radial $\\langle J_{{in}} \\rangle$ vs N")
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
                    help="Path al directorio runs/ del TP3. Si se pasa, se "
                         "genera la comparación <Jin>(N) entre TP4 y TP3.")
    ap.add_argument("--dt-sweep", default=None,
                    help="Path a un directorio con subcarpetas dt*/N{N}_k{k}/ "
                         "(uno por dt). Si se pasa, se genera energy_dt_sweep.png.")
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
        print("Building radial profiles…")
        _, band_obs = plot_radial_profiles(args.base, N_list, k_str, args.out, seed=args.seed)

        # Cierre de 1.3: <Jin>, <rho>, <v> vs N en la banda cercana al obstáculo.
        print(f"Promedios en banda (S∈[{S_BAND_LO},{S_BAND_HI}]) vs N…")
        band_tuple = plot_band_vs_N(band_obs, k_str, args.out)

        # Comparación contra TP3 (sólo si se pasó --tp3-runs).
        if args.tp3_runs:
            print(f"Computando referencia TP3 desde {args.tp3_runs}…")
            tp3_df = tp3_jin_vs_N(args.tp3_runs)
            plot_tp3_comparison(band_tuple, tp3_df, k_str, args.out)

    # dt-sweep energy validation (independente de N/k del barrido principal).
    if args.dt_sweep:
        print(f"\nGenerando plot de barrido de dt desde {args.dt_sweep}…")
        plot_energy_dt_sweep(args.dt_sweep, args.out,
                             N=N_list[0], k_str=str(int(k_list[0])))

    # Cross-k comparison
    if len(k_list) > 1:
        print("\nComparing k values con barras de error cruzadas…")
        run_task_1_4_analysis(args.base, k_list, N_list, args.out)

    print("\nAll done.")


if __name__ == "__main__":
    main()