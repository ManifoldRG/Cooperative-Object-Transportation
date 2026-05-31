"""
scaling_study.py
----------------
Controlled agent-scaling experiment. A single base scenario (orbital params,
inertia, BCs, tf, epsilon, N) is generated once and held fixed across all
agent counts. Only the attachment vectors rs are regenerated per agent count.

Agent counts swept: [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
Time limits:        [60, 300, 600] seconds

Output CSV: one row per agent count, columns:
    n_agents, a, e, m, tf, epsilon_tol,
    <method>_cost_<Xs>, <method>_violation_<Xs>, <method>_runtime_<Xs>

Usage
-----
    python scaling_study.py
    python scaling_study.py --seed 42 --agent-counts 3 5 7 --time-limits 60
    python scaling_study.py --no-mppi
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacecraft_libraries.evaluation import (
    run_method_comparison,
    scaling_base_scenario,
    scaling_inject_agents,
)

AGENT_COUNTS: list[int]   = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
TIME_LIMITS:  list[float] = [60.0, 300.0, 600.0]

METHODS: list[str] = [
    "centralized_nlp",
    "centralized_ga",
    "decentralized_island_ga",
    "centralized_mppi",
    "decentralized_mppi",
]


def _build_fieldnames(methods: list[str], time_limits: list[float]) -> list[str]:
    meta = ["n_agents", "a", "e", "m", "tf", "epsilon_tol"]
    data: list[str] = []
    for method in methods:
        for tl in time_limits:
            tag = f"{int(tl)}s"
            data += [
                f"{method}_cost_{tag}",
                f"{method}_violation_{tag}",
                f"{method}_runtime_{tag}",
            ]
    return meta + data


def _empty_row(n_agents, sys_params, bc, epsilon, methods, time_limits) -> dict:
    row: dict = {
        "n_agents":    n_agents,
        "a":           sys_params.a,
        "e":           sys_params.e,
        "m":           sys_params.m,
        "tf":          bc.tf,
        "epsilon_tol": epsilon,
    }
    for method in methods:
        for tl in time_limits:
            tag = f"{int(tl)}s"
            row[f"{method}_cost_{tag}"]      = float("nan")
            row[f"{method}_violation_{tag}"] = float("nan")
            row[f"{method}_runtime_{tag}"]   = float("nan")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agent-scaling study: fixed base scenario, swept agent count."
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parser.add_argument(
        "--output", type=Path,
        default=Path(f"data/scaling_results/agent_scaling_{timestamp}.csv"),
    )
    parser.add_argument("--seed",         type=int,   default=None)
    parser.add_argument("--agent-counts", type=int,   nargs="*", default=AGENT_COUNTS)
    parser.add_argument("--time-limits",  type=float, nargs="*", default=TIME_LIMITS)
    parser.add_argument("--no-mppi",      action="store_true", help="Skip MPPI methods.")
    parser.add_argument("--mppi-iterations", type=int,   default=5)
    parser.add_argument("--mppi-samples",    type=int,   default=10)
    parser.add_argument("--mppi-sigma",      type=float, default=1e-1)
    parser.add_argument("--mppi-lambda",     type=float, default=1.0)
    parser.add_argument("--mppi-base-seed",  type=int,   default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    active_methods = [m for m in METHODS if not (args.no_mppi and "mppi" in m)]
    fieldnames     = _build_fieldnames(active_methods, args.time_limits)

    base_sys, base_bc, epsilon = scaling_base_scenario()
    print(
        f"\nBase scenario:"
        f"\n  a={base_sys.a:.3e}  e={base_sys.e:.3f}  m={base_sys.m:.1f}kg"
        f"\n  tf={base_bc.tf:.1f}s  epsilon={epsilon:.2e}  N={base_sys.N}"
        f"\n  Fixed for all agent counts."
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for n_agents in args.agent_counts:
            print(f"\n{'='*60}")
            print(f"  n_agents = {n_agents}")
            print(f"{'='*60}")

            sys_params, bc, eps = scaling_inject_agents(base_sys, base_bc, epsilon, n_agents)
            row = _empty_row(n_agents, sys_params, bc, eps, active_methods, args.time_limits)

            for time_limit in args.time_limits:
                tag = f"{int(time_limit)}s"
                print(f"\n  -- {time_limit:.0f}s --")

                try:
                    results = run_method_comparison(
                        sys_params, bc, eps,
                        max_runtime_s=time_limit,
                        show_progress=True,
                        silence_solver_output=True,
                        include_mppi=not args.no_mppi,
                        mppi_iterations=args.mppi_iterations,
                        mppi_samples=args.mppi_samples,
                        mppi_sigma=args.mppi_sigma,
                        mppi_lambda=args.mppi_lambda,
                        mppi_base_seed=args.mppi_base_seed,
                    )
                    for res in results:
                        method = res["method"]
                        if method not in active_methods:
                            continue
                        row[f"{method}_cost_{tag}"]      = res["cost"]
                        row[f"{method}_violation_{tag}"] = res["terminal_violation"]
                        row[f"{method}_runtime_{tag}"]   = res["runtime_s"]
                        cost_str = f"{res['cost']:.3e}" if np.isfinite(res["cost"]) else "FAILED"
                        print(
                            f"    {method:<30}"
                            f"  cost={cost_str}"
                            f"  violation={res['terminal_violation']:.3e}"
                            f"  runtime={res['runtime_s']:.1f}s"
                        )
                except Exception as exc:
                    print(f"  WARNING: {n_agents} agents @ {time_limit:.0f}s failed — {exc}")
                    traceback.print_exc()

            writer.writerow(row)
            f.flush()

    print(f"\nDone. Saved to {args.output}")


if __name__ == "__main__":
    main()