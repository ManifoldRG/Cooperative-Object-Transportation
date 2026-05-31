"""
monte_carlo_comparison.py
-------------------------
Monte Carlo comparison of all solvers across randomly generated scenarios
(scenario 5 from evaluation.get_scenario). Each run draws a fresh random
scenario, runs all solvers, and writes results incrementally to CSV.

Usage
-----
    python monte_carlo_comparison.py --runs 30 --time-limit 60
    python monte_carlo_comparison.py --runs 10 --time-limit 120 --no-mppi --seed 42
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

from spacecraft_libraries.evaluation import get_scenario, run_method_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo comparison across randomised scenarios."
    )
    parser.add_argument("--runs",       type=int,   default=3,
                        help="Number of Monte Carlo runs. Default: 10")
    parser.add_argument("--time-limit", type=float, default=6.0,
                        help="Per-solver time limit in seconds. Default: 60")
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Optional base RNG seed for reproducibility.")
    parser.add_argument("--no-mppi",    action="store_true", help="Skip MPPI methods.")
    parser.add_argument("--mppi-iterations", type=int,   default=5)
    parser.add_argument("--mppi-samples",    type=int,   default=10)
    parser.add_argument("--mppi-sigma",      type=float, default=1e-1)
    parser.add_argument("--mppi-lambda",     type=float, default=1.0)
    parser.add_argument("--mppi-base-seed",  type=int,   default=42)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parser.add_argument(
        "--output", type=Path,
        default=Path(f"data/monte_carlo_results/monte_carlo_{timestamp}.csv"),
    )
    return parser.parse_args()


def print_rows(run_id: int, rows: list[dict]) -> None:
    for row in rows:
        cost_str = f"{row['cost']:.3e}" if np.isfinite(row["cost"]) else "FAILED"
        viol_str = f"{row['terminal_violation']:.3e}" if np.isfinite(row["terminal_violation"]) else "FAILED"
        print(
            f"  [{run_id}] {row['method']:<30}"
            f"  cost={cost_str}"
            f"  violation={viol_str}"
            f"  runtime={row['runtime_s']:.1f}s"
        )


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Base RNG seed: {args.seed}")

    fieldnames = [
        "run_id", "time_limit_s", "method",
        "cost", "terminal_violation", "runtime_s",
        "n_agents", "a", "e", "m", "tf", "epsilon_tol",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed_runs: list[int] = []

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_id in range(1, args.runs + 1):
            print(f"\n--- Monte Carlo run {run_id}/{args.runs} ---")
            try:
                sys_params, bc, epsilon = get_scenario(5)  # random_scenario_generator

                meta = {
                    "n_agents":    len(sys_params.rs),
                    "a":           sys_params.a,
                    "e":           sys_params.e,
                    "m":           sys_params.m,
                    "tf":          bc.tf,
                    "epsilon_tol": epsilon,
                }
                print(
                    f"  a={meta['a']:.3e}  e={meta['e']:.3f}  "
                    f"m={meta['m']:.1f}kg  tf={meta['tf']:.1f}s  "
                    f"agents={meta['n_agents']}  eps={meta['epsilon_tol']:.2e}"
                )

                rows = run_method_comparison(
                    sys_params, bc, epsilon,
                    max_runtime_s=args.time_limit,
                    show_progress=True,
                    silence_solver_output=True,
                    include_mppi=not args.no_mppi,
                    mppi_iterations=args.mppi_iterations,
                    mppi_samples=args.mppi_samples,
                    mppi_sigma=args.mppi_sigma,
                    mppi_lambda=args.mppi_lambda,
                    mppi_base_seed=args.mppi_base_seed,
                )

                print_rows(run_id, rows)

                for row in rows:
                    row["run_id"]       = run_id
                    row["time_limit_s"] = args.time_limit
                    row.update(meta)

                writer.writerows(rows)
                f.flush()

            except Exception as exc:
                print(f"  WARNING: run {run_id} failed — {exc}")
                traceback.print_exc()
                failed_runs.append(run_id)
                continue

    print(f"\nCompleted {args.runs - len(failed_runs)}/{args.runs} runs.")
    if failed_runs:
        print(f"Failed run IDs: {failed_runs}")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()