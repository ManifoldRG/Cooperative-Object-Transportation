"""
baseline_comparison.py
-----------------------
Monte Carlo baseline comparison across 5 solvers:
    centralized_nlp        (cold)
    centralized_nlp_warm   (warm)
    centralized_ga
    centralized_mppi
    decentralized_mppi

Self-contained: does NOT modify or depend on run_method_comparison() in
spacecraft_libraries.evaluation.comparison, so it can't affect other
collaborators using that shared module. The only shared-library addition is
solve_centralized_nlp_warm in solvers/centralized_nlp.py, which is purely
additive (new function, nothing else touched).

Each run draws a fresh random scenario via random_scenario_generator(),
called directly with a randomly chosen agent count in [3, 6] — the agent cap
is applied here, not in the shared generator, so its default range (3-30)
is unaffected for everyone else.

Usage
-----
    python baseline_comparison.py --runs 30 --time-limit 60
    python baseline_comparison.py --runs 10 --time-limit 120 --seed 42
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spacecraft_libraries.evaluation.comparison import random_scenario_generator
from spacecraft_libraries.evaluation.metrics import lie_attitude_violation, terminal_violation
from spacecraft_libraries.solvers.centralized_nlp import solve_centralized_nlp, solve_centralized_nlp_warm
from spacecraft_libraries.solvers.centralized_ga import solve_centralized_ga
from spacecraft_libraries.solvers.centralized_mppi import solve_centralized_mppi
from spacecraft_libraries.solvers.decentralized_mppi import solve_decentralized_mppi

MIN_AGENTS = 3
MAX_AGENTS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo baseline comparison: cold/warm NLP, centralized GA, centralized/decentralized MPPI."
    )
    parser.add_argument("--runs",       type=int,   default=10,
                        help="Number of Monte Carlo runs. Default: 10")
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="Per-solver time limit in seconds. Default: 60")
    parser.add_argument("--seed",       type=int,   default=None,
                        help="Optional base RNG seed for reproducibility.")
    parser.add_argument("--mppi-iterations", type=int,   default=20)
    parser.add_argument("--mppi-samples",    type=int,   default=10)
    parser.add_argument("--mppi-sigma",      type=float, default=0.7)
    parser.add_argument("--mppi-lambda",     type=float, default=1.0)
    parser.add_argument("--mppi-seed",       type=int,   default=42)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parser.add_argument(
        "--output", type=Path,
        default=Path(f"data/baseline_comparison/baseline_{timestamp}.csv"),
    )
    return parser.parse_args()


def _extract_terminal_state(result: dict, method: str) -> dict:
    """Local copy — NLP solvers return a flat (N+1,12) state array, everything
    else returns a trajectory of StateVectorLie objects."""
    if method in ("centralized_nlp", "centralized_nlp_warm"):
        x = result["state"]
        return {"r": x[-1, 0:3], "v": x[-1, 3:6], "phi": x[-1, 6:9], "omega": x[-1, 9:12]}
    traj = result["trajectory"]
    s = traj.states[-1]
    return {"r": s.r, "v": s.v, "phi": s.phi, "omega": s.omega}


def run_baseline_solvers(sys_params, bc, epsilon, max_runtime_s: float,
                          mppi_iterations: int, mppi_samples: int,
                          mppi_sigma: float, mppi_lambda: float, mppi_seed: int) -> list[dict]:
    """Runs exactly the 5 baseline solvers on one scenario. Independent of
    run_method_comparison — decentralized_island_ga and centralized_nlp's
    include_mppi toggle don't apply here, this function's solver set is fixed."""
    solver_calls = [
        ("centralized_nlp", lambda: solve_centralized_nlp(
            sys_params, bc, max_iters=3000, max_runtime_s=max_runtime_s)),
        ("centralized_nlp_warm", lambda: solve_centralized_nlp_warm(
            sys_params, bc, max_iters=3000, max_runtime_s=max_runtime_s)),
        ("centralized_ga", lambda: solve_centralized_ga(
            sys_params, bc, epsilon, pop_size=10, generations=5000, max_runtime_s=max_runtime_s)),
        ("centralized_mppi", lambda: solve_centralized_mppi(
            sys_params, bc, epsilon,
            n_iter=mppi_iterations, n_samples=mppi_samples,
            sigma=mppi_sigma, lambda_=mppi_lambda, seed=mppi_seed,
            max_runtime_s=max_runtime_s)),
        ("decentralized_mppi", lambda: solve_decentralized_mppi(
            sys_params, bc, epsilon,
            n_iter=mppi_iterations, n_samples=mppi_samples,
            sigma=mppi_sigma, lambda_=mppi_lambda, base_seed=mppi_seed,
            max_runtime_s=max_runtime_s)),
    ]

    results = []
    for method_name, solver in solver_calls:
        try:
            results.append(solver())
        except Exception as exc:
            print(f"  [{method_name}] failed — {exc}")
            traceback.print_exc()
            results.append({
                "method": method_name,
                "cost": float("nan"),
                "runtime": float("nan"),
                "state": None,
                "trajectory": None,
            })

    table = []
    for result in results:
        cost = result.get("cost")
        if cost is None or (isinstance(cost, float) and np.isnan(cost)):
            table.append({
                "method": result["method"],
                "cost": float("nan"),
                "terminal_violation": float("nan"),
                "runtime_s": float("nan"),
            })
            continue
        terminal = _extract_terminal_state(result, result["method"])
        violation = (
            terminal_violation(terminal["r"], bc.xf.r)
            + terminal_violation(terminal["v"], bc.xf.v)
            + lie_attitude_violation(terminal["phi"], bc.xf.phi)
            + terminal_violation(terminal["omega"], bc.xf.omega)
        )
        table.append({
            "method": result["method"],
            "cost": float(result["cost"]),
            "terminal_violation": float(violation),
            "runtime_s": float(result["runtime"]),
        })
    return table


def print_rows(run_id: int, rows: list[dict]) -> None:
    for row in rows:
        cost_str = f"{row['cost']:.3e}" if np.isfinite(row["cost"]) else "FAILED"
        viol_str = f"{row['terminal_violation']:.3e}" if np.isfinite(row["terminal_violation"]) else "FAILED"
        print(
            f"  [{run_id}] {row['method']:<24}"
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
            print(f"\n--- Baseline run {run_id}/{args.runs} ---")
            try:
                # agent cap applied here, at the call site — the shared
                # random_scenario_generator's own default range (3-30) is untouched
                n_agents = random.randint(MIN_AGENTS, MAX_AGENTS)
                sys_params, bc, epsilon = random_scenario_generator(fixed_agents_num=n_agents)

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

                rows = run_baseline_solvers(
                    sys_params, bc, epsilon,
                    max_runtime_s=args.time_limit,
                    mppi_iterations=args.mppi_iterations,
                    mppi_samples=args.mppi_samples,
                    mppi_sigma=args.mppi_sigma,
                    mppi_lambda=args.mppi_lambda,
                    mppi_seed=args.mppi_seed,
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