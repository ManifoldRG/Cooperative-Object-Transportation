"""Figures for a pipeline run (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _equal_3d(ax, pts):
    pts = np.vstack(pts)
    c = pts.mean(axis=0)
    r = np.max(np.ptp(pts, axis=0)) / 2 or 1.0
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def trajectories(sc, out, path):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")
    allpts = []
    for j in range(sc.M):
        c = COLORS[j % len(COLORS)]
        r0 = out["refs0"][j]["r"]
        rf = out["verify"]["paths"][j]
        allpts += [r0, rf]
        ax.plot(*r0.T, ls=":", color=c, alpha=0.6, lw=1)
        ax.plot(*rf.T, color=c, lw=2.2, label=f"payload {j}")
        ax.scatter(*rf[0], color=c, marker="o", s=40)
        ax.scatter(*rf[-1], color=c, marker="X", s=60)
        for w in out["waypoints"][j].values():
            ax.scatter(*w, color="gold", edgecolor="k", marker="*", s=120)
    _equal_3d(ax, allpts)
    ax.set_xlabel("radial x [m]")
    ax.set_ylabel("along-track y [m]")
    ax.set_zlabel("normal z [m]")
    ax.set_title("Hill-frame trajectories: dotted = Step-1 references, solid = verified\n"
                 "rollout, stars = repair waypoints, o = start, X = goal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def separation(sc, out, path):
    paths = out["verify"]["paths"]
    t = np.linspace(0, sc.tf, sc.N + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j in range(sc.M):
        for l in range(j + 1, sc.M):
            d0 = np.linalg.norm(out["refs0"][j]["r"] - out["refs0"][l]["r"], axis=1)
            d = np.linalg.norm(paths[j] - paths[l], axis=1)
            ln, = ax.plot(t, d, lw=2, label=f"{j}-{l}")
            ax.plot(t, d0, ls=":", color=ln.get_color(), alpha=0.7)
    ax.axhline(sc.delta, color="k", ls="--", lw=1.2, label=r"$\delta$")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("center separation [m]")
    ax.set_title("Pairwise separation: dotted = unrepaired references, solid = verified")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def allocation(sc, out, path):
    occ = out["allocation"]["occupied"]
    fig = plt.figure(figsize=(4 * sc.M, 4))
    for j, P in enumerate(sc.payloads):
        ax = fig.add_subplot(1, sc.M, j + 1, projection="3d")
        pts = P.points
        for i, p in enumerate(pts):
            on = i in occ[j]
            ax.scatter(*p, s=60, color=COLORS[j % len(COLORS)] if on else "white",
                       edgecolor="k")
            if on:
                d = -p / np.linalg.norm(p) * 0.35 * np.linalg.norm(p)
                ax.quiver(*p, *d, color="#d62728", lw=1.5)
        ax.scatter(0, 0, 0, color="k", marker="+", s=80)
        _equal_3d(ax, [pts, np.zeros((1, 3))])
        ax.set_title(f"payload {j}: {len(occ[j])} agents\n(filled = occupied, arrow = cone axis)",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def all_figures(sc, out, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    trajectories(sc, out, outdir / "trajectories.png")
    separation(sc, out, outdir / "separation.png")
    allocation(sc, out, outdir / "allocation.png")
