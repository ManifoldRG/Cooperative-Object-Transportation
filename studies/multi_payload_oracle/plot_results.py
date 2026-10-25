"""Regenerate the two headline charts from results/ next to this script.

  py plot_results.py            both charts (needs the fair gauss + full tent runs)
  py plot_results.py tubes      inflation+tubes scaling chart alone
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).with_name("results")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SEC, MUT, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"


def load(name):
    return [r for r in json.loads((RES / name).read_text())
            if r.get("status") == "ok"]


def axes(n):
    fig, axs = plt.subplots(1, n, figsize=(4.9 * n, 4.3), dpi=160)
    fig.patch.set_facecolor(SURF)
    for ax in np.atleast_1d(axs):
        ax.set_facecolor(SURF)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.xaxis.set_tick_params(length=0, labelsize=9.5)
        ax.yaxis.set_tick_params(length=0, labelsize=9, colors=MUT)
        ax.set_xlabel("payloads  M", fontsize=10, color=SEC)
    return fig, np.atleast_1d(axs)


def agg(rows, Ms, fn):
    return [np.mean([fn(r) for r in rows if r["M"] == M]) for M in Ms]


def chart_tubes():
    rows = load("benchmark_tubes_fair.json")
    Ms = sorted({r["M"] for r in rows})
    fig, (ax1, ax2) = axes(2)
    for ax in (ax1, ax2):
        ax.set_xticks(Ms)
    for key, col, lab, ls in (
            ("t_relax", ORANGE, "inflation relaxation (shared)", "-"),
            ("t_tubes", AQUA, "tube SOCPs, sequential sum", "-"),
            ("t_tube_max", AQUA, "slowest single tube (parallel)", (0, (4, 2)))):
        v = agg(rows, Ms, lambda r, k=key: r[k])
        ax1.plot(Ms, v, ls=ls, color=col, lw=2, marker="o", ms=5, label=lab)
    ax1.set_yscale("log")
    ax1.set_title("Runtime (s, log scale)", fontsize=11, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=SEC, loc="upper left")
    for M in Ms:
        g = [r["gap_pct"] for r in rows if r["M"] == M]
        ax2.plot([M] * len(g), g, "o", color=AQUA, ms=3.5, alpha=0.3)
    ax2.plot(Ms, agg(rows, Ms, lambda r: r["gap_pct"]), "-", color=AQUA, lw=2,
             marker="o", ms=5, label="mean of 10 seeds")
    ax2.set_ylim(0, None)
    ax2.set_title("Certified gap vs lower bound (%)", fontsize=11, color=INK,
                  loc="left")
    ax2.legend(frameon=False, fontsize=8.5, labelcolor=SEC, loc="upper right")
    fig.suptitle("Inflation + decoupled tubes at constant congestion density"
                 "  (gap = (J − Σℓ)/Σℓ)",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RES / "benchmark_tubes_fair.png"
    fig.savefig(out, facecolor=SURF, bbox_inches="tight")
    print(out)


def chart_profiles():
    data = {"gauss": load("benchmark_tubes_fair.json"),
            "tent": load("benchmark_tubes_tent_full.json")}
    Ms = sorted({r["M"] for r in data["gauss"]})
    fig, (ax1, ax2) = axes(2)
    for ax in (ax1, ax2):
        ax.set_xticks(Ms)
    for prof, col in (("gauss", BLUE), ("tent", ORANGE)):
        tr = agg(data[prof], Ms, lambda r: r["t_relax"])
        ax1.plot(Ms, tr, "-", color=col, lw=2, marker="o", ms=5,
                 label=prof + " relaxation")
        for M in Ms:
            g = [r["gap_pct"] for r in data[prof] if r["M"] == M]
            ax2.plot([M] * len(g), g, "o", color=col, ms=3, alpha=0.25)
        ax2.plot(Ms, agg(data[prof], Ms, lambda r: r["gap_pct"]), "-",
                 color=col, lw=2, marker="o", ms=5, label=prof)
    ax1.set_yscale("log")
    ax1.set_title("Relaxation time (s, log scale)", fontsize=11, color=INK,
                  loc="left")
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=SEC, loc="upper left")
    ax2.set_ylim(0, None)
    ax2.set_title("Certified gap vs lower bound (%)", fontsize=11, color=INK,
                  loc="left")
    ax2.legend(frameon=False, fontsize=8.5, labelcolor=SEC, loc="upper left")
    fig.suptitle("Gauss vs tent push profiles, inflation + decoupled tubes",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = RES / "benchmark_gauss_vs_tent.png"
    fig.savefig(out, facecolor=SURF, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "tubes"):
        chart_tubes()
    if which in ("all", "profiles"):
        chart_profiles()
