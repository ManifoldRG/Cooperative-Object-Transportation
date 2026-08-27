#!/usr/bin/env python3
"""
Usage:
    python baseline_comparison.py --task-id 1 \
        --scenarios results/scenarios.json \
        --output-dir results/tasks

Baseline comparison, one solver run per SLURM array task.

Method roster (2026-08-12): cold NLP, warm NLP, centralized GA, Greedy
Sampler (centralized + decentralized), Gradient Descent (centralized +
decentralized), and MPPI (centralized + decentralized) — 9 methods total.

Every stochastic solver (GS, GD, MPPI) gets INDEPENDENT hyperparameters per
centralized/decentralized variant — no sharing. The centralized and
decentralized versions of the same method are different search processes
(single joint search vs. per-island sequential search + consensus) and the
2026-08 sensitivity sweeps never established that their optima coincide, so
each of the 4 stochastic-variant slots (gs-c, gs-d, mppi-c, mppi-d) plus the
2 GD slots (gd-c, gd-d) takes its own CLI-supplied sigma/step_size/
tau_init_std/n_samples/rel_step/lambda.

GD's terminal state (r, v, phi, omega) is enforced as a hard equality
constraint in both the inner solve and the attitude projector, so its
terminal_violation is reported as 0.0 rather than recomputed from a
trajectory object (GD's result dict carries no such object).

MPPI is deadline-driven here: n_iter is fixed to a large sentinel
(MPPI_DEADLINE_ITERS) so max_runtime_s is the binding stop condition, same
convention GS uses.
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
from spacecraft_libraries.solvers.gradient_descent import (
    solve_centralized_gd, solve_decentralized_gd,
)
from spacecraft_libraries.solvers.centralized_mppi import solve_centralized_mppi
from spacecraft_libraries.solvers.decentralized_mppi import solve_decentralized_mppi


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
    "centralized_gd",
    "decentralized_gd",
    "centralized_mppi",
    "decentralized_mppi",
]
NLP_METHODS = {"centralized_nlp_th", "centralized_nlp_th_warm"}
GS_METHODS = {"centralized_gs", "decentralized_gs"}
GD_METHODS = {"centralized_gd", "decentralized_gd"}
MPPI_METHODS = {"centralized_mppi", "decentralized_mppi"}
SEEDED_METHODS = {"centralized_ga"} | GS_METHODS | GD_METHODS | MPPI_METHODS

# MPPI is deadline-driven for this grid: n_iter is a large sentinel so
# max_runtime_s is the binding stop condition (mirrors GS's DEADLINE_ITERS).
MPPI_DEADLINE_ITERS = 1_000_000

TIME_LIMITS = [60.0, 300.0, 600.0, 1200.0]
# Default batch-size-by-budget fallback, used only when a variant's own
# --*-n-samples is left unset. Same schedule for every stochastic method;
# override per variant on the command line if a method needs its own.
SAMPLE_SCHEDULE = {
    300.0:  12,
    600.0:  17,
    1200.0: 24,
}

FIELDNAMES = [
    "scenario_id", "method", "time_limit_s",
    "cost", "terminal_violation", "runtime_s",
    "n_agents", "a", "e", "m", "tf", "epsilon_tol", "nu",
    "solver_seed",                      # GA, all GS/GD/MPPI variants
    "tau_init_std", "sigma", "step_size", "noise_mode", "n_samples",  # GS
    "rel_step",                         # GD
    "lambda_",                          # MPPI
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


def resolve_params(method: str, args: argparse.Namespace, time_limit: float) -> dict:
    """Per-(method, variant) hyperparameters. No cross-variant sharing:
    centralized and decentralized versions of the same solver family each
    read their own CLI-supplied values, independently."""

    def batch(explicit):
        return explicit if explicit is not None else SAMPLE_SCHEDULE.get(time_limit, 10)

    if method == "centralized_gs":
        return dict(sigma=args.gs_c_sigma, step_size=args.gs_c_step_size,
                    tau_init_std=args.gs_c_tau_init_std,
                    n_samples=batch(args.gs_c_n_samples))
    if method == "decentralized_gs":
        return dict(sigma=args.gs_d_sigma, step_size=args.gs_d_step_size,
                    tau_init_std=args.gs_d_tau_init_std,
                    n_samples=batch(args.gs_d_n_samples))
    if method == "centralized_gd":
        return dict(rel_step=args.gd_c_rel_step, tau_init_std=args.gd_c_tau_init_std)
    if method == "decentralized_gd":
        return dict(rel_step=args.gd_d_rel_step, tau_init_std=args.gd_d_tau_init_std)
    if method == "centralized_mppi":
        return dict(sigma=args.mppi_c_sigma, lambda_=args.mppi_c_lambda,
                    n_samples=batch(args.mppi_c_n_samples))
    if method == "decentralized_mppi":
        return dict(sigma=args.mppi_d_sigma, lambda_=args.mppi_d_lambda,
                    n_samples=batch(args.mppi_d_n_samples))
    return {}


def run_one_solver(method: str, sys_params, bc, epsilon, max_runtime_s: float,
                    params: dict, noise_mode: str, solver_seed: int) -> dict:
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
            sys_params, bc, epsilon, seed=solver_seed,
            n_samples=params["n_samples"], sigma=params["sigma"],
            tau_init_scale=params["tau_init_std"], noise_mode=noise_mode,
            step_size=params["step_size"], max_runtime_s=max_runtime_s),
        "decentralized_gs": lambda: solve_decentralized_gs(
            sys_params, bc, epsilon, base_seed=solver_seed,
            n_samples=params["n_samples"], sigma=params["sigma"],
            tau_init_scale=params["tau_init_std"], noise_mode=noise_mode,
            step_size=params["step_size"], max_runtime_s=max_runtime_s),
        "centralized_gd": lambda: solve_centralized_gd(
            sys_params, bc, epsilon, seed=solver_seed,
            tau_init_scale=params["tau_init_std"], rel_step=params["rel_step"],
            max_runtime_s=max_runtime_s),
        "decentralized_gd": lambda: solve_decentralized_gd(
            sys_params, bc, epsilon, base_seed=solver_seed,
            tau_init_scale=params["tau_init_std"], rel_step=params["rel_step"],
            max_runtime_s=max_runtime_s),
        "centralized_mppi": lambda: solve_centralized_mppi(
            sys_params, bc, epsilon, seed=solver_seed,
            n_iter=MPPI_DEADLINE_ITERS, n_samples=params["n_samples"],
            sigma=params["sigma"], lambda_=params["lambda_"],
            max_runtime_s=max_runtime_s),
        "decentralized_mppi": lambda: solve_decentralized_mppi(
            sys_params, bc, epsilon, base_seed=solver_seed,
            n_iter=MPPI_DEADLINE_ITERS, n_samples=params["n_samples"],
            sigma=params["sigma"], lambda_=params["lambda_"],
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

    if method in GD_METHODS:
        # Terminal r,v,phi,omega are hard equality constraints in GD's inner
        # solve and projector; no trajectory object is returned to check.
        violation = 0.0
    else:
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
    p.add_argument("--noise-mode", choices=["white", "smooth"], default="white",
                    help="Shared across both GS variants (not yet split "
                         "per-variant — noise_mode/knots deferred).")

    # Greedy Sampler — centralized (2026-08-12 sweep; tau left at prior
    # baseline — sigma at the sweep's range ceiling)
    p.add_argument("--gs-c-sigma",        type=float, default=0.05)
    p.add_argument("--gs-c-step-size",    type=float, default=1.5)
    p.add_argument("--gs-c-tau-init-std", type=float, default=0.1)
    p.add_argument("--gs-c-n-samples",    type=int,   default=4)
    # Greedy Sampler — decentralized (same values as CGS for now, per your
    # call — sweep never separated centralized/decentralized winners)
    p.add_argument("--gs-d-sigma",        type=float, default=0.05)
    p.add_argument("--gs-d-step-size",    type=float, default=1.5)
    p.add_argument("--gs-d-tau-init-std", type=float, default=0.1)
    p.add_argument("--gs-d-n-samples",    type=int,   default=4)

    # Gradient Descent — centralized
    p.add_argument("--gd-c-rel-step",     type=float, default=0.03)
    p.add_argument("--gd-c-tau-init-std", type=float, default=0.1)
    # Gradient Descent — decentralized
    p.add_argument("--gd-d-rel-step",     type=float, default=0.03)
    p.add_argument("--gd-d-tau-init-std", type=float, default=0.1)

    # MPPI — centralized (sigma/lambda from 2026-08-12 sweep; n_samples left
    # on SAMPLE_SCHEDULE — the tuned n_iter=10/n_samples=20 pairing is
    # dropped in favor of deadline-capped iterations + budget-scaled batch)
    p.add_argument("--mppi-c-sigma",     type=float, default=1.0)
    p.add_argument("--mppi-c-lambda",    type=float, default=0.5)
    p.add_argument("--mppi-c-n-samples", type=int,   default=None)
    # MPPI — decentralized (sigma/lambda from 2026-08-12 sweep; n_samples
    # likewise left on SAMPLE_SCHEDULE, dropping the tuned n_iter=20/
    # n_samples=10 pairing)
    p.add_argument("--mppi-d-sigma",     type=float, default=1.0)
    p.add_argument("--mppi-d-lambda",    type=float, default=0.9)
    p.add_argument("--mppi-d-n-samples", type=int,   default=None)

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

    params = resolve_params(combo["method"], args, combo["time_limit"])
    solver_seed = derive_solver_seed(combo["scenario"], combo["time_limit"], args.seed_fallback)
    if combo["method"] in SEEDED_METHODS:
        print(f"  solver_seed={solver_seed}  params={params}")

    result = run_one_solver(
        combo["method"], sys_params, bc, epsilon,
        max_runtime_s=combo["time_limit"],
        params=params,
        noise_mode=args.noise_mode,
        solver_seed=solver_seed,
    )

    is_seeded = combo["method"] in SEEDED_METHODS
    is_gs = combo["method"] in GS_METHODS
    is_gd = combo["method"] in GD_METHODS
    is_mppi = combo["method"] in MPPI_METHODS
    row = {
        "scenario_id":  combo["scenario_id"],
        "method":       combo["method"],
        "time_limit_s": combo["time_limit"],
        "n_agents":     len(sys_params.rs),
        "nu": sys_params.nu,
        "a":            sys_params.a,
        "e":            sys_params.e,
        "m":            sys_params.m,
        "tf":           bc.tf,
        "epsilon_tol":  epsilon,
        "solver_seed":  solver_seed if is_seeded else "",
        "tau_init_std": params.get("tau_init_std", "") if (is_gs or is_gd) else "",
        "sigma":        params.get("sigma", "") if (is_gs or is_mppi) else "",
        "step_size":    params.get("step_size", "") if is_gs else "",
        "noise_mode":   args.noise_mode if is_gs else "",
        "n_samples":    params.get("n_samples", "") if (is_gs or is_mppi) else "",
        "rel_step":     params.get("rel_step", "") if is_gd else "",
        "lambda_":      params.get("lambda_", "") if is_mppi else "",
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