"""
analyze_system2.py  –  System 2 analysis
Covers TP4 tasks 1.1 through 1.4:
  1.1  Execution time vs N
  1.2  Scanning rate J(N) from linear fit to Cfc(t)
  1.3  Radial profiles <rho_fin>(S), <v_fin>(S), Jin(S)
  1.4  Comparison across k values

Directory convention (written by System2Main):
  base_dir/
    N{N}_k{k}/
      seed{s}/
        states.txt
        cfc.txt
        energy.txt
        info.txt
    timing_k{k}.csv

Usage:
    python analyze_system2.py --base output/system2 --k 1000 --Nlist 100,200,300,500
"""

import argparse, glob, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

R_DOMAIN   = 40.0
R_OBSTACLE = 1.0
R_PARTICLE = 1.0
DS         = 0.2   # radial shell width


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_runs(base_dir, N, k_str):
    """Return list of run directories for given N and k."""
    pattern = os.path.join(base_dir, f"N{N}_k{k_str}", "seed*")
    dirs = sorted(glob.glob(pattern))
    return dirs


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
    log_n, log_t = np.log(N), np.log(t)
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

def compute_J(cfc_times, cfc_vals):
    """Linear regression slope = scanning rate J [1/s]."""
    if len(cfc_times) < 2:
        return 0.0, 0.0
    slope, intercept, r, p, se = stats.linregress(cfc_times, cfc_vals)
    return slope, se


def scanning_rate_vs_N(base_dir, N_list, k_str, out_dir):
    J_mean, J_std, Ns_ok = [], [], []

    for N in N_list:
        runs = find_runs(base_dir, N, k_str)
        if not runs:
            print(f"  No runs found for N={N}, k={k_str}")
            continue
        Js = []
        for run_dir in runs:
            cfc_path = os.path.join(run_dir, "cfc.txt")
            if not os.path.exists(cfc_path):
                continue
            t_arr, c_arr = load_cfc(cfc_path)
            if len(t_arr) < 2:
                Js.append(0.0)
                continue
            slope, _ = compute_J(t_arr, c_arr)
            Js.append(slope)
        if Js:
            J_mean.append(np.mean(Js))
            J_std .append(np.std(Js, ddof=1) if len(Js) > 1 else 0.0)
            Ns_ok .append(N)

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
    run_dir   = os.path.join(base_dir, f"N{N}_k{k_str}", f"seed{seed}")
    ene_path  = os.path.join(run_dir, "energy.txt")
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
    """
    For each frame accumulate density and radial velocity of fresh particles
    moving toward the center (x·v < 0).
    Returns (S_bins, rho_mean, vrad_mean, jin_mean).
    """
    S_max   = R_DOMAIN
    S_bins  = np.arange(R_OBSTACLE + DS / 2, S_max, DS)
    counts  = np.zeros(len(S_bins))
    vrad_sum= np.zeros(len(S_bins))
    n_frames = 0

    for t, arr in parse_frames(states_path):
        # arr columns: x y vx vy r state
        fresh = arr[arr[:, 5] == 0]          # fresh only
        if len(fresh) == 0:
            n_frames += 1
            continue

        x, y = fresh[:, 0], fresh[:, 1]
        vx, vy = fresh[:, 2], fresh[:, 3]
        dist = np.sqrt(x**2 + y**2)
        xdotv = x * vx + y * vy            # x·v

        # Only particles moving toward center
        mask  = xdotv < 0
        x_in, y_in = x[mask], y[mask]
        vx_in, vy_in = vx[mask], vy[mask]
        dist_in = dist[mask]
        xdotv_in = xdotv[mask]

        if len(dist_in) == 0:
            n_frames += 1
            continue

        # radial velocity component (toward center → negative)
        vrad = xdotv_in / dist_in        # v_fin_j = (x_j · v_j) / |x_j|

        for idx, s in enumerate(S_bins):
            s_lo, s_hi = s - DS / 2, s + DS / 2
            shell_mask = (dist_in >= s_lo) & (dist_in < s_hi)
            counts[idx]   += shell_mask.sum()
            vrad_sum[idx] += np.sum(np.abs(vrad[shell_mask]))

        n_frames += 1

    if n_frames == 0:
        return S_bins, np.zeros_like(S_bins), np.zeros_like(S_bins), np.zeros_like(S_bins)

    areas = np.array([shell_area(s) for s in S_bins])
    rho       = counts / (n_frames * areas)
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

    jin_near_obs = []   # averaged over S in [1.5, 5] m

    for N, color in zip(N_list, colors):
        run_dir    = os.path.join(base_dir, f"N{N}_k{k_str}", f"seed{seed}")
        states_path = os.path.join(run_dir, "states.txt")
        if not os.path.exists(states_path):
            print(f"  Missing {states_path}")
            continue

        S, rho, vmean, jin = build_radial_profile(states_path)

        ax_rho.plot(S, rho,   color=color, lw=1.2)
        ax_v  .plot(S, vmean, color=color, lw=1.2)
        ax_jin.plot(S, jin,   color=color, lw=1.2)

        # Near-obstacle average Jin  (S in [1.5, 5] m as suggested by TP)
        near = (S >= 1.5) & (S <= 5.0)
        jin_near = jin[near].mean() if near.any() else 0.0
        jin_near_obs.append((N, jin_near))

    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(N_list), vmax=max(N_list)))
    sm.set_array([])

    for ax, fig_, title, ylabel, tag in [
        (ax_rho, fig_rho, f"Densidad <ρ_fin>(S)  k={k_str}", "ρ (m⁻²)",     "rho"),
        (ax_v,   fig_v,   f"Velocidad radial <v_fin>(S)  k={k_str}", "|v| (m/s)", "vrad"),
        (ax_jin, fig_jin, f"Flujo Jin(S)  k={k_str}", "Jin (m⁻² s⁻¹)",       "jin"),
    ]:
        ax.set_xlabel("S (m)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig_.colorbar(sm, ax=ax, label="N")
        savefig(fig_, os.path.join(out_dir, f"radial_{tag}_k{k_str}.png"))

    # Detail of Jin near obstacle
    fig_det, ax_det = plt.subplots(figsize=(8, 5))
    for N, color in zip(N_list, colors):
        run_dir    = os.path.join(base_dir, f"N{N}_k{k_str}", f"seed{seed}")
        states_path = os.path.join(run_dir, "states.txt")
        if not os.path.exists(states_path):
            continue
        S, rho, vmean, jin = build_radial_profile(states_path)
        near = (S >= 1.5) & (S <= 5.0)
        ax_det.plot(S[near], jin[near], color=color, lw=1.2)

    ax_det.set_xlabel("S (m)")
    ax_det.set_ylabel("Jin (m⁻² s⁻¹)")
    ax_det.set_title(f"Detalle Jin  S∈[1.5,5] m  k={k_str}")
    ax_det.grid(True, alpha=0.3)
    fig_det.colorbar(sm, ax=ax_det, label="N")
    savefig(fig_det, os.path.join(out_dir, f"jin_detail_k{k_str}.png"))

    return jin_near_obs


# ═══════════════════════════════════════════════════════════════════════════════
# 1.4  Comparison across k values
# ═══════════════════════════════════════════════════════════════════════════════

def compare_k_values(base_dir, N_list, k_list, out_dir):
    cmap   = plt.cm.viridis
    colors = [cmap(i / max(len(k_list) - 1, 1)) for i in range(len(k_list))]

    fig_J, ax_J = plt.subplots(figsize=(8, 5))

    for k_val, color in zip(k_list, colors):
        k_str = str(int(k_val))
        csv   = os.path.join(base_dir, f"J_vs_N_k{k_str}.csv")
        if not os.path.exists(csv):
            print(f"  Missing {csv} – run scanning_rate_vs_N first for k={k_str}")
            continue
        df = pd.read_csv(csv)
        ax_J.errorbar(df["N"], df["J_mean"], yerr=df["J_std"],
                      fmt="o-", capsize=4, color=color, label=f"k={k_str}")

    ax_J.set_xlabel("N")
    ax_J.set_ylabel("Scanning rate J  [1/s]")
    ax_J.set_title("Scanning rate <J>(N) para distintos k")
    ax_J.legend()
    ax_J.grid(True, alpha=0.3)
    savefig(fig_J, os.path.join(out_dir, "scanning_rate_compare_k.png"))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",   default="output/system2")
    ap.add_argument("--k",      default="1000",  help="comma-separated k values, e.g. 100,1000,10000")
    ap.add_argument("--Nlist",  default="100,200,300,500",
                    help="comma-separated N values to analyse")
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
        plot_energy(args.base, N_list[0], k_str, args.out, seed=args.seed)

        # Scanning rate
        print("Computing scanning rate…")
        scanning_rate_vs_N(args.base, N_list, k_str, args.out)

        # Radial profiles
        print("Building radial profiles…")
        plot_radial_profiles(args.base, N_list, k_str, args.out, seed=args.seed)

    # Cross-k comparison
    if len(k_list) > 1:
        print("\nComparing k values…")
        compare_k_values(args.base, N_list, k_list, args.out)

    print("\nAll done.")


if __name__ == "__main__":
    main()
