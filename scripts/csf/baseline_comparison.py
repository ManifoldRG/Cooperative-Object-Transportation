#!/usr/bin/env python3
"""
baseline_comparison.py
-----------------------
Baseline comparison — one solver run per task, called by SLURM job array.

Usage:
    python baseline_comparison.py --task-id 1 \
        --scenarios results/scenarios.json \
        --output-dir results/tasks
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from spacecraft_libraries.data_structures import (
    BoundaryConditions, SystemParams, StateVectorLie,
)
from spacecraft_libraries.evaluation.metrics import lie_attitude_violation, terminal_violation
from spacecraft_libraries.solvers.centralized_nlp import solve_centralized_nlp, solve_centralized_nlp_warm
from spacecraft_libraries.solvers.centralized_ga import solve_centralized_ga
from spacecraft_libraries.solvers.centralized_mppi import solve_centralized_mppi
from spacecraft_libraries.solvers.decentralized_mppi import solve_decentralized_mppi

METHODS = [
    "centralized_nlp",
    "centralized_nlp_warm",
    "centralized_ga",
    "centralized_mppi",
    "decentralized_mppi",
]

FIELDNAMES = [
    "scenario_id", "method", "time_limit_s",
    "cost", "terminal_violation", "runtime_s",
    "n_agents", "a", "e", "m", "tf", "epsilon_tol",
]


def build_combos(scenarios):
    combos = []
    for sc, method in product(scenarios, METHODS):
        combos.append(dict(scenario_id=sc["scenario_id"], scenario=sc, method=method))
    return combos


def load_scenarios(path):
    with open(path) as f:
        return json.load(f)


def make_sys_bc(s):
    sys_params = SystemParams(
        mu=s["mu"], a=s["a"], e=s["e"], nu=s["nu"],
        I=np.diag(s["I_diag"]), m=s["m"],
        rs=[np.array(r) for r in s["rs"]], N=s["N"],
    )
    bc = BoundaryConditions(
        x0=StateVectorLie(r=np.array(s["x0_r"]), v=np.array(s["x0_v"]),
                          phi=np.array(s["x0_phi"]), omega=np.array(s["x0_omega"])),
        xf=StateVectorLie(r=np.array(s["xf_r"]), v=np.array(s["xf_v"]),
                          phi=np.array(s["xf_phi"]), omega=np.array(s["xf_omega"])),
        tf=s["tf"],
    )
    return sys_params, bc, s["epsilon"]


def _extract_terminal_state(result: dict, method: str) -> dict:
    """NLP solvers return a flat (N+1,12) state array, everything else
    returns a trajectory of StateVectorLie objects."""
    if method in ("centralized_nlp", "centralized_nlp_warm"):
        x = result["state"]
        return {"r": x[-1, 0:3], "v": x[-1, 3:6], "phi": x[-1, 6:9], "omega": x[-1, 9:12]}
    traj = result["trajectory"]
    s = traj.states[-1]
    return {"r": s.r, "v": s.v, "phi": s.phi, "omega": s.omega}


def run_one_solver(method: str, sys_params, bc, epsilon, max_runtime_s: float,
                    mppi_iterations: int, mppi_samples: int,
                    mppi_sigma: float, mppi_lambda: float, mppi_seed: int) -> dict:
    solvers = {
        "centralized_nlp": lambda: solve_centralized_nlp(
            sys_params, bc, max_iters=3000, max_runtime_s=max_runtime_s),
        "centralized_nlp_warm": lambda: solve_centralized_nlp_warm(
            sys_params, bc, max_iters=3000, max_runtime_s=max_runtime_s),
        "centralized_ga": lambda: solve_centralized_ga(
            sys_params, bc, epsilon, pop_size=10, generations=5000, max_runtime_s=max_runtime_s),
        "centralized_mppi": lambda: solve_centralized_mppi(
            sys_params, bc, epsilon,
            n_iter=mppi_iterations, n_samples=mppi_samples,
            sigma=mppi_sigma, lambda_=mppi_lambda, seed=mppi_seed,
            max_runtime_s=max_runtime_s),
        "decentralized_mppi": lambda: solve_decentralized_mppi(
            sys_params, bc, epsilon,
            n_iter=mppi_iterations, n_samples=mppi_samples,
            sigma=mppi_sigma, lambda_=mppi_lambda, base_seed=mppi_seed,
            max_runtime_s=max_runtime_s),
    }

    try:
        result = solvers[method]()
    except Exception as exc:
        print(f"  [{method}] failed — {exc}")
        traceback.print_exc()
        return {"cost": float("nan"), "terminal_violation": float("nan"), "runtime_s": float("nan")}

    cost = result.get("cost")
    if cost is None or (isinstance(cost, float) and np.isnan(cost)):
        return {"cost": float("nan"), "terminal_violation": float("nan"), "runtime_s": float("nan")}

    terminal = _extract_terminal_state(result, method)
    violation = (
        terminal_violation(terminal["r"], bc.xf.r)
        + terminal_violation(terminal["v"], bc.xf.v)
        + lie_attitude_violation(terminal["phi"], bc.xf.phi)
        + terminal_violation(terminal["omega"], bc.xf.omega)
    )
    return {
        "cost": float(result["cost"]),
        "terminal_violation": float(violation),
        "runtime_s": float(result["runtime"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline comparison, one solver per task."
    )
    p.add_argument("--task-id",    type=int,  required=True)
    p.add_argument("--scenarios",  type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/tasks"))
    p.add_argument("--time-limit", type=float, default=60.0)
    p.add_argument("--mppi-iterations", type=int,   default=20)
    p.add_argument("--mppi-samples",    type=int,   default=10)
    p.add_argument("--mppi-sigma",      type=float, default=0.7)
    p.add_argument("--mppi-lambda",     type=float, default=1.0)
    p.add_argument("--mppi-seed",       type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(args.scenarios)
    combos    = build_combos(scenarios)
    total     = len(combos)

    print(f"Total combos: {total}", flush=True)
    print(f"Task ID: {args.task_id} / {total}", flush=True)

    idx = args.task_id - 1
    if idx < 0 or idx >= total:
        print(f"Task ID {args.task_id} out of range (1-{total})")
        sys.exit(1)

    combo = combos[idx]
    print(f"Combo: sc={combo['scenario_id']} method={combo['method']}", flush=True)

    sys_params, bc, epsilon = make_sys_bc(combo["scenario"])
    print(
        f"  a={sys_params.a:.3e}  e={sys_params.e:.3f}  m={sys_params.m:.1f}kg  "
        f"tf={bc.tf:.1f}s  agents={len(sys_params.rs)}  eps={epsilon:.2e}"
    )

    result = run_one_solver(
        combo["method"], sys_params, bc, epsilon,
        max_runtime_s=args.time_limit,
        mppi_iterations=args.mppi_iterations,
        mppi_samples=args.mppi_samples,
        mppi_sigma=args.mppi_sigma,
        mppi_lambda=args.mppi_lambda,
        mppi_seed=args.mppi_seed,
    )

    row = {
        "scenario_id":  combo["scenario_id"],
        "method":       combo["method"],
        "time_limit_s": args.time_limit,
        "n_agents":     len(sys_params.rs),
        "a":            sys_params.a,
        "e":            sys_params.e,
        "m":            sys_params.m,
        "tf":           bc.tf,
        "epsilon_tol":  epsilon,
        **result,
    }

    out = args.output_dir / f"task_{args.task_id:04d}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerow(row)

    cost_str = f"{row['cost']:.3e}" if np.isfinite(row["cost"]) else "FAILED"
    print(f"Done. cost={cost_str}, runtime={row['runtime_s']:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()