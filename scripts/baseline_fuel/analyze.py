"""Analysis + figure pipeline for fuel-baseline campaign CSVs.

Usage: python analyze.py <results.csv> [outdir]

Works on both the local driver's CSV (V_rollout column) and the CSF harness
CSV (terminal_violation = rollout V). Prints the paper tables and writes
fig_success_budget and fig_fuel_paired as PNG + PDF.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                     "axes.spines.top": False, "axes.spines.right": False})

# fixed entity colors (validated categorical palette; color follows method)
STYLE = {
    "decentralized_gd":   dict(color="#1179a3", marker="o", ls="-",  label="D-GD (ours)"),
    "centralized_gd":     dict(color="#5aa9d6", marker="s", ls="--", label="C-GD (ours)"),
    "centralized_nlp_th": dict(color="#c22525", marker="^", ls="-",  label="NLP"),
    "centralized_ga":     dict(color="#b0308f", marker="D", ls="-",  label="GA"),
    "centralized_mppi":   dict(color="#d97f1e", marker="v", ls="--", label="C-MPPI"),
    "decentralized_mppi": dict(color="#6a3fb5", marker="P", ls="-",  label="D-MPPI"),
}
ORDER = list(STYLE)


def load(path):
    x = pd.read_csv(path)
    vcol = "V_rollout" if "V_rollout" in x.columns else "terminal_violation"
    x["V"] = x[vcol]
    x["ok"] = np.isfinite(x["cost_rollout"]) & (x["V"] < 1e-3)
    if "thrust_angle_deg" not in x.columns and "nu" in x.columns:
        x["thrust_angle_deg"] = np.degrees(x["nu"]).round(1)
    return x


def tables(x):
    n_cell = x.groupby(["method", "time_limit_s"]).size().max()
    print(f"=== success % (denominator {n_cell}/cell; crashes/missing = fail) ===")
    t = x.groupby(["method", "time_limit_s"]).ok.sum().unstack().reindex(ORDER)
    print((t / n_cell * 100).round(1).to_string())

    buds = sorted(x.time_limit_s.unique())
    bmax = buds[-1]

    print(f"\n=== success by cone angle at {bmax:g} s (ok / n) ===")
    s = x[x.time_limit_s == bmax]
    per = s.groupby(["thrust_angle_deg", "method"]).ok.agg(["sum", "size"]) \
        if "thrust_angle_deg" in x.columns else None
    if per is not None:
        per["cell"] = per["sum"].astype(str) + "/" + per["size"].astype(str)
        print(per["cell"].unstack()[ORDER].to_string())

    print(f"\n=== paired fuel vs D-GD (jointly solved, median of ratios) ===")
    for bud in buds:
        d = x[(x.time_limit_s == bud) & (x.method == "decentralized_gd") & x.ok]
        d = d.set_index("scenario_id").cost_rollout
        row = []
        for m in ORDER[1:]:
            sm = x[(x.time_limit_s == bud) & (x.method == m) & x.ok]
            sm = sm.set_index("scenario_id").cost_rollout
            c = sm.index.intersection(d.index)
            row.append(f"{STYLE[m]['label']} {float((sm[c] / d[c]).median()):.3f} (n={len(c)})"
                       if len(c) >= 5 else f"{STYLE[m]['label']} n={len(c)}")
        print(f"{bud:6g}s: " + " | ".join(row))

    n = x[x.method == "centralized_nlp_th"]
    if "converged" in x.columns:
        conv = n.groupby("time_limit_s").converged.apply(
            lambda s: (s.astype(str) == "True").sum())
        print("\nNLP converged per cell:", conv.to_dict())
        cf = int(((n.converged.astype(str) == "True") & (n.V >= 1e-3)).sum())
        print("NLP converged-but-rollout-fail:", cf)
    if "ipopt_status" in x.columns:
        bad = n[n.ipopt_status.astype(str).str.contains("MISSING|ABSTAIN", na=False)]
        print("NLP crashed/missing rows:", len(bad))


def fig_success(x, outdir):
    buds = sorted(x.time_limit_s.unique())
    n_cell = x.groupby(["method", "time_limit_s"]).size().max()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for m in ORDER:
        st = STYLE[m]
        y = [x[(x.method == m) & (x.time_limit_s == b)].ok.sum() / n_cell * 100
             for b in buds]
        ax.plot(buds, y, color=st["color"], marker=st["marker"], ls=st["ls"],
                lw=2, ms=6, label=st["label"])
    ax.set_xscale("log")
    ax.set_xticks(buds)
    ax.set_xticklabels([f"{b:g}" for b in buds])
    ax.set_xlim(buds[0] * 0.85, buds[-1] * 1.25)
    ax.set_xlabel("compute budget per agent [s]")
    ax.set_ylabel("success rate [%]")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="lower right", ncol=2, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_success_budget.{ext}"), dpi=300)
    plt.close(fig)


def fig_fuel(x, outdir):
    buds = sorted(x.time_limit_s.unique())
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for m in ORDER[1:]:
        st = STYLE[m]
        ys, bs = [], []
        for b in buds:
            d = x[(x.time_limit_s == b) & (x.method == "decentralized_gd") & x.ok]
            d = d.set_index("scenario_id").cost_rollout
            sm = x[(x.time_limit_s == b) & (x.method == m) & x.ok]
            sm = sm.set_index("scenario_id").cost_rollout
            c = sm.index.intersection(d.index)
            if len(c) >= 5:
                ys.append(float((sm[c] / d[c]).median()))
                bs.append(b)
        if bs:
            ax.plot(bs, ys, color=st["color"], marker=st["marker"], ls=st["ls"],
                    lw=2, ms=6, label=st["label"])
    ax.axhline(1.0, color="#555555", lw=1.2)
    ax.annotate("D-GD (ours) = 1.0", (buds[0], 1.0), xytext=(2, -12),
                textcoords="offset points", fontsize=8.5, color="#555555")
    ax.set_xscale("log")
    ax.set_xticks(buds)
    ax.set_xticklabels([f"{b:g}" for b in buds])
    ax.set_xlim(buds[0] * 0.85, buds[-1] * 1.25)
    ax.set_xlabel("compute budget per agent [s]")
    ax.set_ylabel("median fuel relative to D-GD\n(jointly solved scenarios)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_fuel_paired.{ext}"), dpi=300)
    plt.close(fig)


def main():
    path = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(path) or "."
    os.makedirs(outdir, exist_ok=True)
    x = load(path)
    tables(x)
    fig_success(x, outdir)
    fig_fuel(x, outdir)
    print(f"\nfigures -> {outdir}")


if __name__ == "__main__":
    main()
