"""
animate_particles.py  –  System 2 animator
Reads states.txt from a simulation run and produces an MP4 animation.

Usage:
    python animate_particles.py --states output/system2/N200_k1000/seed42/states.txt \
                                 --out    output/system2/N200_k1000/seed42/anim.mp4 \
                                 --fps 20 --frames 300
"""

import argparse, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter, FuncAnimation

R_DOMAIN   = 40.0
R_OBSTACLE = 1.0
COLOR_FRESH = "#27ae60"
COLOR_USED  = "#e74c3c"
COLOR_OBS   = "#7f8c8d"


# ── parser ────────────────────────────────────────────────────────────────────

def parse_states(path):
    """Yield (time, array[N,6]) frames from the states.txt file."""
    with open(path) as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            n   = int(header)
            tline = f.readline().strip()          # "time=X.XXXXXX"
            t = float(tline.split("=")[1])
            rows = []
            for _ in range(n):
                vals = list(map(float, f.readline().split()))
                rows.append(vals)                 # x y vx vy r state
            yield t, np.array(rows)


def load_frames(path, max_frames=None):
    frames = []
    for t, arr in parse_states(path):
        frames.append((t, arr))
        if max_frames and len(frames) >= max_frames:
            break
    return frames


# ── animation ─────────────────────────────────────────────────────────────────

def make_animation(states_path, out_path, fps=15, max_frames=300):
    print(f"Loading frames from {states_path}…")
    frames = load_frames(states_path, max_frames)
    if not frames:
        print("No frames found.")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-R_DOMAIN - 1, R_DOMAIN + 1)
    ax.set_ylim(-R_DOMAIN - 1, R_DOMAIN + 1)
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    ax.axis("off")

    # Domain circle (white outline)
    domain_circle = plt.Circle((0, 0), R_DOMAIN, fill=False,
                                edgecolor="white", lw=1.5, alpha=0.8)
    ax.add_patch(domain_circle)

    # Obstacle
    obs = plt.Circle((0, 0), R_OBSTACLE, color=COLOR_OBS, zorder=3)
    ax.add_patch(obs)

    title = ax.set_title("", color="white", fontsize=10)
    fresh_label = ax.text(-R_DOMAIN + 1, -R_DOMAIN + 2, "", color=COLOR_FRESH, fontsize=9)
    used_label  = ax.text(-R_DOMAIN + 1, -R_DOMAIN + 4, "", color=COLOR_USED,  fontsize=9)

    # Particle artists
    t0, arr0 = frames[0]
    n = len(arr0)
    circles = []
    for row in arr0:
        x, y, vx, vy, r, state = row
        c = plt.Circle((x, y), r,
                        color=COLOR_USED if state else COLOR_FRESH,
                        alpha=0.85, zorder=4)
        ax.add_patch(c)
        circles.append(c)

    def update(frame_idx):
        t, arr = frames[frame_idx]
        n_fresh = int((arr[:, 5] == 0).sum())
        n_used  = len(arr) - n_fresh
        title.set_text(f"t = {t:.1f} s   N = {len(arr)}")
        fresh_label.set_text(f"● Frescas: {n_fresh}")
        used_label .set_text(f"● Usadas:  {n_used}")
        for i, row in enumerate(arr):
            x, y, vx, vy, r, state = row
            circles[i].center = (x, y)
            circles[i].set_color(COLOR_USED if state else COLOR_FRESH)
        return circles + [title, fresh_label, used_label]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    print(f"Saving animation → {out_path}")
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print("Done.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--fps",    type=int, default=15)
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    make_animation(args.states, args.out, fps=args.fps, max_frames=args.frames)


if __name__ == "__main__":
    main()
