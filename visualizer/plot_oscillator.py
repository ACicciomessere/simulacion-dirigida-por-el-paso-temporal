"""
plot_oscillator.py  –  System 1 visualizer
Reads output/system1/trajectories.csv and output/system1/mse_vs_dt.csv
and produces:
  - trajectories.png   (analytical vs each integrator)
  - mse_vs_dt.png      (log-log convergence study)
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── defaults ─────────────────────────────────────────────────────────────────

COLORS = {
    "r_euler":      "#f9e90c",
    "r_verlet":     "#2ecc71",
    "r_beeman":     "#3498db",
    "r_gear":       "#9b59b6",
    "r_analytical": "#e74c3c",
}
LABELS = {
    "r_euler":      "Euler",
    "r_verlet":     "Verlet",
    "r_beeman":     "Beeman",
    "r_gear":       "Gear PC-5",
    "r_analytical": "Analítica",
}
INTEGRATORS = ["r_euler", "r_verlet", "r_beeman", "r_gear"]

# ── helpers ───────────────────────────────────────────────────────────────────

def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  → {path}")
    plt.close(fig)


# ── 1. Trajectories ───────────────────────────────────────────────────────────

def plot_trajectories(traj_path, out_dir):
    df = pd.read_csv(traj_path)
    t  = df["time"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, col in zip(axes, INTEGRATORS):
        ana_col = "r_analytical" if "r_analytical" in df.columns else "analytical"
        ax.plot(t, df[col], color=COLORS[col],
                lw=1.5, ls="--", label=LABELS[col], zorder=3)   # numérica adelante
        ax.plot(t, df[ana_col], color=COLORS["r_analytical"],
                lw=1.2, label="Analítica", zorder=2)             # analítica atrás
        ax.set_title(LABELS[col], fontsize=11)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Posición (m)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Oscilador amortiguado – trayectorias", fontsize=13)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "trajectories.png"))

    # combined in one axes
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(t, df["r_analytical"], color=COLORS["r_analytical"],
             lw=2, label="Analítica", zorder=2)                  # analítica atrás
    for col in INTEGRATORS:
        ax2.plot(t, df[col], color=COLORS[col],
                 lw=1.5, ls="--", label=LABELS[col], zorder=3)   # numéricas adelante
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("Posición (m)")
    ax2.set_title("Oscilador amortiguado – comparación de integradores")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    savefig(fig2, os.path.join(out_dir, "trajectories_combined.png"))

    # error vs analítica en un solo gráfico
    ana_col = "r_analytical" if "r_analytical" in df.columns else "analytical"
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for col in INTEGRATORS:
        error = np.abs(df[col].values - df[ana_col].values)
        ax3.plot(t, error, color=COLORS[col], lw=1.2, label=LABELS[col])
    ax3.set_xlabel("Tiempo (s)")
    ax3.set_ylabel("Error absoluto |x_num − x_ana| (m)")
    ax3.set_title("Error vs. Solución analítica") #más residuo q error, pero bueno
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")          # log por si los errores difieren en órdenes de magnitud
    savefig(fig3, os.path.join(out_dir, "trajectories_error.png"))

# ── 2. MSE vs dt (log-log) ────────────────────────────────────────────────────

def plot_mse(mse_path, out_dir):
    df  = pd.read_csv(mse_path)
    dts = df["dt"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    # column names can be "mse_euler" or "mse_r_euler" depending on Java output
    col_map = {}
    for col in INTEGRATORS:
        short = col.replace("r_", "")          # "euler", "verlet", etc.
        for candidate in [f"mse_{col}", f"mse_{short}"]:
            if candidate in df.columns:
                col_map[col] = candidate
                break

    for col in INTEGRATORS:
        mse_col = col_map.get(col)
        if mse_col:
            ax.loglog(dts, df[mse_col], "o-", color=COLORS[col], label=LABELS[col])

    euler_col = col_map.get("r_euler", "mse_euler")
    # Reference lines
    xref = np.array([dts.min(), dts.max()])
    for order, ls, lbl in [(1, ":", "O(dt)"), (2, "--", "O(dt²)"), (4, "-.", "O(dt⁴)")]:
        scale = df[euler_col].iloc[0] / dts[0]**order
        ax.loglog(xref, scale * xref**order, color="gray", ls=ls, lw=0.8,
                  alpha=0.6, label=lbl)

    ax.set_xlabel("Paso temporal dt (s)")
    ax.set_ylabel("Error cuadrático medio (m²)")
    ax.set_title("ECM vs dt  –  oscilador amortiguado")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    savefig(fig, os.path.join(out_dir, "mse_vs_dt.png"))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="output/system1/trajectories.csv")
    ap.add_argument("--mse",  default="output/system1/mse_vs_dt.csv")
    ap.add_argument("--out",  default="output/system1")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if os.path.exists(args.traj):
        print("Plotting trajectories…")
        plot_trajectories(args.traj, args.out)
    else:
        print(f"Missing {args.traj} – run System 1 first.")

    if os.path.exists(args.mse):
        print("Plotting MSE vs dt…")
        plot_mse(args.mse, args.out)
    else:
        print(f"Missing {args.mse} – run System 1 first.")


if __name__ == "__main__":
    main()
