#!/usr/bin/env python3
"""
Usage:
    python baseline_comparison.py --task-id 1 \
        --scenarios results/scenarios.json \
        --output-dir results/tasks

Baseline comparison, one solver run per SLURM array task.

Method roster (2026-08-11): cold NLP, warm NLP, centralized GA, and the
Greedy Sampler (centralized + decentralized) — GS is the production sampler
that replaced MPPI after the tuning marathon (argmax selection over projected
samples, relative sigma, white noise, deadline-driven). The MPPI solvers
remain in spacecraft_libraries for comparison studies but are no longer part
of the baseline grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
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
from spacecraft_libraries.solvers.centralized_nlp_th import solve_centralized_nlp_th
from spacecraft_libraries.solvers.centralized_ga import solve_centralized_ga
from spacecraft_libraries.solvers.greedy_sampler import (
    solve_centralized_gs, solve_decentralized_gs,
)
from spacecraft_libraries.solvers.mppi_core import make_nominal_tau
from spacecraft_libraries.solvers.parametric_oracle import ScenarioOracle

# NLP entries use solve_centralized_nlp_th — the joint NLP with the CORRECT
# reference dynamics (TH + body-frame thrusts). The old og_opts.full_nlp
# solves different physics (CW translation, Hill-frame thrusts); its costs
# were never comparable and all pre-2026-08-17 NLP columns are artifacts.
# "_warm" seeds the thrust variables with the bilevel pipeline's first
# iterate (inner solve at the projected shooting nominal).
METHODS = [
    "centralized_nlp_th",
    "centralized_nlp_th_warm",
    "centralized_ga",
    "centralized_gs",
    "decentralized_gs",
]
NLP_METHODS = {"centralized_nlp_th", "centralized_nlp_th_warm"}
GS_METHODS = {"centralized_gs", "decentralized_gs"}
SEEDED_METHODS = {"centralized_ga"} | GS_METHODS

TIME_LIMITS = [60.0, 300.0, 600.0, 1200.0]
# GS is deadline-driven (unbounded iterations, wall clock terminates); the
# schedule sets only the sample batch size between greedy jumps.
SAMPLE_SCHEDULE = {
    300.0:  12,
    600.0:  17,
    1200.0: 24,
}

FIELDNAMES = [
    "scenario_id", "method", "time_limit_s",
    "cost", "terminal_violation", "runtime_s",
    "n_agents", "a", "e", "m", "tf", "epsilon_tol",
    "solver_seed",                      # GA + both GS
    "tau_init_std", "sigma", "step_size", "noise_mode", "n_samples",  # GS only
]


def build_combos(scenarios):
    combos = []
    for sc, method, time_limit in product(scenarios, METHODS, TIME_LIMITS):
        combos.append(dict(scenario_id=sc["scenario_id"], scenario=sc,
                            method=method, time_limit=time_limit))
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


def derive_solver_seed(scenario: dict, time_limit: float, fallback_seed: int) -> int:
    base = scenario.get("seed")
    if base is None:
        base = fallback_seed
    return int(base) * 10007 + int(time_limit)


def run_centralized_ga_seeded(sys_params, bc, epsilon, pop_size, generations,
                               max_runtime_s, seed):
    # np.random.seed requires < 2**32; derive_solver_seed values exceed it
    # (scenario timestamp seeds * 10007 ~ 1.8e13), so reduce mod 2**32.
    np.random.seed(seed % (2**32))
    return solve_centralized_ga(sys_params, bc, epsilon, pop_size=pop_size,
                                 generations=generations, max_runtime_s=max_runtime_s)


def _warm_thrust_guess(sys_params, bc, epsilon):
    """Bilevel first iterate as the joint NLP's thrust guess: inner solve at
    the projected shooting nominal. ~1-2 s."""
    oracle = ScenarioOracle(sys_params, bc, epsilon)
    nominal = make_nominal_tau(sys_params, bc, epsilon,
                               np.random.default_rng(0), tau_init_scale=0.1)
    tau_p, _ = oracle.project(nominal)
    if tau_p is None:
        return None
    ok, _, x, _ = oracle.inner_cost(tau_p)
    if not ok or x is None:
        return None
    n_u = len(sys_params.rs) * sys_params.N * 3
    return x[:n_u]


def _extract_terminal_state(result: dict, method: str) -> dict:
    if method in NLP_METHODS:
        x = result["state"]
        return {"r": x[-1, 0:3], "v": x[-1, 3:6], "phi": x[-1, 6:9], "omega": x[-1, 9:12]}
    traj = result["trajectory"]
    s = traj.states[-1]
    return {"r": s.r, "v": s.v, "phi": s.phi, "omega": s.omega}


def run_one_solver(method: str, sys_params, bc, epsilon, max_runtime_s: float,
                    n_samples: int, sigma: float, step_size: float,
                    noise_mode: str, tau_init_std: float, solver_seed: int) -> dict:
    gs_kwargs = dict(n_samples=n_samples, sigma=sigma,
                     tau_init_scale=tau_init_std, noise_mode=noise_mode,
                     step_size=step_size, max_runtime_s=max_runtime_s)
    solvers = {
        "centralized_nlp_th": lambda: solve_centralized_nlp_th(
            sys_params, bc, epsilon, max_iters=20000, max_runtime_s=max_runtime_s),
        "centralized_nlp_th_warm": lambda: solve_centralized_nlp_th(
            sys_params, bc, epsilon, max_iters=20000, max_runtime_s=max_runtime_s,
            U_guess=_warm_thrust_guess(sys_params, bc, epsilon)),
        "centralized_ga": lambda: run_centralized_ga_seeded(
            sys_params, bc, epsilon, pop_size=10, generations=5000,
            max_runtime_s=max_runtime_s, seed=solver_seed),
        "centralized_gs": lambda: solve_centralized_gs(
            sys_params, bc, epsilon, seed=solver_seed, **gs_kwargs),
        "decentralized_gs": lambda: solve_decentralized_gs(
            sys_params, bc, epsilon, base_seed=solver_seed, **gs_kwargs),
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
    p.add_argument("--sigma",      type=float, default=0.03,
                    help="GS relative sigma (fraction of warm-start nominal "
                         "RMS torque). 0.03 won the 2026-08 sweeps.")
    p.add_argument("--step-size",  type=float, default=1.0,
                    help="GS update step toward the batch-best projected "
                         "sample (1.0 = full greedy jump).")
    p.add_argument("--noise-mode", choices=["white", "smooth"], default="white")
    p.add_argument("--n-samples",  type=int,   default=None,
                    help="GS batch size. Default: per-budget SAMPLE_SCHEDULE.")
    p.add_argument("--tau-init-std", type=float, default=0.1,
                    help="Scale of the uniform random tau guess fed to the "
                         "multiple-shooting warm start (matches GA pop init).")
    p.add_argument("--seed-fallback", type=int, default=42,
                    help="Used only if a scenario has no stored 'seed' field.")
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
    print(f"Combo: sc={combo['scenario_id']} method={combo['method']} "
          f"time_limit={combo['time_limit']}s", flush=True)

    sys_params, bc, epsilon = make_sys_bc(combo["scenario"])
    print(
        f"  a={sys_params.a:.3e}  e={sys_params.e:.3f}  m={sys_params.m:.1f}kg  "
        f"tf={bc.tf:.1f}s  agents={len(sys_params.rs)}  eps={epsilon:.2e}"
    )

    if args.n_samples is not None:
        n_samples = args.n_samples
    else:
        n_samples = SAMPLE_SCHEDULE.get(combo["time_limit"], 10)
    solver_seed = derive_solver_seed(combo["scenario"], combo["time_limit"], args.seed_fallback)
    is_gs = combo["method"] in GS_METHODS
    if combo["method"] in SEEDED_METHODS:
        extra = (f"  n_samples={n_samples}  sigma={args.sigma}  step={args.step_size}  "
                 f"noise={args.noise_mode}  tau_init_std={args.tau_init_std}") if is_gs else ""
        print(f"  solver_seed={solver_seed}{extra}")

    result = run_one_solver(
        combo["method"], sys_params, bc, epsilon,
        max_runtime_s=combo["time_limit"],
        n_samples=n_samples,
        sigma=args.sigma,
        step_size=args.step_size,
        noise_mode=args.noise_mode,
        tau_init_std=args.tau_init_std,
        solver_seed=solver_seed,
    )

    is_seeded = combo["method"] in SEEDED_METHODS
    row = {
        "scenario_id":  combo["scenario_id"],
        "method":       combo["method"],
        "time_limit_s": combo["time_limit"],
        "n_agents":     len(sys_params.rs),
        "a":            sys_params.a,
        "e":            sys_params.e,
        "m":            sys_params.m,
        "tf":           bc.tf,
        "epsilon_tol":  epsilon,
        "solver_seed":  solver_seed if is_seeded else "",
        "tau_init_std": args.tau_init_std if is_gs else "",
        "sigma":        args.sigma if is_gs else "",
        "step_size":    args.step_size if is_gs else "",
        "noise_mode":   args.noise_mode if is_gs else "",
        "n_samples":    n_samples if is_gs else "",
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
