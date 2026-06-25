"""Plot MPPI hyperparameter sensitivity results.

Produces:
  1. One plot per hyperparameter (sigma, n_iter, n_samples) — mean normalised
     cost vs parameter value, averaged over all other dims, ±1σ across repeats.
  2. tau_init_std robustness plot for the best known config.
  3. Raw cost vs tau_init_std for the best config.
  4. Summary table printed to terminal.

Normalisation: cost normalised to [0,1] per scenario.
Terminal violation is consistently near-zero across all runs and is not used
in ranking — this is noted separately.

Usage:
    python plot_mppi_sensitivity.py --input results/mppi_sensitivity_clean.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PARAMS     = ["sigma", "n_iter", "n_samples"]
METHODS    = ["centralized_mppi", "decentralized_mppi"]
COLOURS    = {"centralized_mppi": "#1f77b4", "decentralized_mppi": "#ff7f0e"}
LABELS     = {"centralized_mppi": "Centralised", "decentralized_mppi": "Decentralised"}
LOG_PARAMS = {"sigma"}

BEST_SIGMA, BEST_N_ITER, BEST_N_SAMPLES = 0.5, 10, 20


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      type=Path, default=Path("results/mppi_sensitivity_clean.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/sensitivity_plots"))
    p.add_argument("--exclude-scenarios", type=int, nargs="+", default=[2],
                   help="Scenarios to exclude (default: 2)")
    return p.parse_args()


def add_normalised_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise cost to [0,1] within each scenario."""
    df = df.copy()
    df["normalised_cost"] = float("nan")
    for scenario, grp in df.groupby("scenario"):
        cost = grp["cost"]
        df.loc[grp.index, "normalised_cost"] = (
            (cost - cost.min()) / (cost.max() - cost.min() + 1e-30)
        )
    return df


def plot_param(df, param, output_dir):
    """Mean ± 1σ normalised cost vs param, averaged over repeats and other dims."""
    fig, ax = plt.subplots(figsize=(7, 4))

    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        per_repeat = sub.groupby([param, "repeat"])["normalised_cost"].mean().reset_index()
        grouped    = per_repeat.groupby(param)["normalised_cost"]
        means      = grouped.mean()
        stds       = grouped.std().fillna(0)

        ax.plot(means.index, means.values, marker="o",
                color=COLOURS[method], label=LABELS[method])
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.15, color=COLOURS[method])

    ax.set_xlabel(param)
    ax.set_ylabel("Normalised cost (↓ better)")
    ax.set_title(f"Sensitivity to {param}")
    ax.legend()
    if param in LOG_PARAMS:
        ax.set_xscale("log")
    fig.tight_layout()
    out = output_dir / f"sensitivity_{param}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_tau_init(df, output_dir):
    """Normalised cost vs tau_init_std for best config."""
    fig, ax = plt.subplots(figsize=(7, 4))
    best = df[
        (df["sigma"]     == BEST_SIGMA) &
        (df["n_iter"]    == BEST_N_ITER) &
        (df["n_samples"] == BEST_N_SAMPLES)
    ]
    if best.empty:
        print("WARNING: no data for best config in tau_init plot")
        plt.close(fig)
        return

    for method in METHODS:
        sub = best[best["method"] == method]
        if sub.empty:
            continue
        grouped = sub.groupby("tau_init_std")["normalised_cost"]
        means   = grouped.mean()
        stds    = grouped.std().fillna(0)
        ax.plot(means.index, means.values, marker="o",
                color=COLOURS[method], label=LABELS[method])
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.15, color=COLOURS[method])

    ax.set_xlabel("tau_init_std")
    ax.set_ylabel("Normalised cost (↓ better)")
    ax.set_title(f"Robustness to initialisation noise\n"
                 f"(sigma={BEST_SIGMA}, n_iter={BEST_N_ITER}, n_samples={BEST_N_SAMPLES})")
    ax.legend()
    fig.tight_layout()
    out = output_dir / "sensitivity_tau_init_std.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_cost_tau_init(df, output_dir):
    """Raw cost vs tau_init_std for best config."""
    fig, ax = plt.subplots(figsize=(7, 4))
    best = df[
        (df["sigma"]     == BEST_SIGMA) &
        (df["n_iter"]    == BEST_N_ITER) &
        (df["n_samples"] == BEST_N_SAMPLES)
    ]
    for method in METHODS:
        sub = best[best["method"] == method]
        if sub.empty:
            continue
        grouped = sub.groupby("tau_init_std")["cost"]
        means   = grouped.mean()
        stds    = grouped.std().fillna(0)
        ax.plot(means.index, means.values, marker="o",
                color=COLOURS[method], label=LABELS[method])
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.15, color=COLOURS[method])

    ax.set_xlabel("tau_init_std")
    ax.set_ylabel("Cost")
    ax.set_title(f"Cost vs initialisation noise\n"
                 f"(sigma={BEST_SIGMA}, n_iter={BEST_N_ITER}, n_samples={BEST_N_SAMPLES})")
    ax.legend()
    fig.tight_layout()
    out = output_dir / "cost_vs_tau_init_std.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def print_summary(df):
    group_cols = PARAMS + ["tau_init_std"]
    print("\n── Best configs per method (lowest mean normalised cost) ──")
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        per_combo = sub.groupby(group_cols)["normalised_cost"].mean()
        best_idx  = per_combo.idxmin()
        best_dict = dict(zip(group_cols, best_idx))
        print(f"  {method}: {best_dict}  →  normalised_cost={per_combo.min():.4f}")

    print(f"\n── Mean cost at best config "
          f"(sigma={BEST_SIGMA}, n_iter={BEST_N_ITER}, n_samples={BEST_N_SAMPLES}) ──")
    best = df[
        (df["sigma"]     == BEST_SIGMA) &
        (df["n_iter"]    == BEST_N_ITER) &
        (df["n_samples"] == BEST_N_SAMPLES)
    ]
    print(best.groupby(["method", "tau_init_std"])["cost"]
              .agg(["mean", "std"]).round(2).to_string())

    print("\n── Terminal violation summary (all runs) ──")
    print(df.groupby("method")["terminal_violation"]
            .agg(["mean", "std", "max"]).to_string())


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["cost"])

    print(f"Rows after filtering: {len(df)}")

    df = add_normalised_cost(df)

    for param in PARAMS:
        plot_param(df, param, args.output_dir)

    plot_tau_init(df, args.output_dir)
    plot_cost_tau_init(df, args.output_dir)
    print_summary(df)
    print(f"\nDone. Plots → {args.output_dir}")