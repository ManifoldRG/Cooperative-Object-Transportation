#!/usr/bin/env python3
"""
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
from spacecraft_libraries.solvers.centralized_nlp import solve_centralized_nlp, solve_centralized_nlp_warm
from spacecraft_libraries.solvers.centralized_ga import solve_centralized_ga
from spacecraft_libraries.solvers.mppi_core import make_nominal_tau, run_mppi
from spacecraft_libraries.solvers.parametric_oracle import ScenarioOracle
from spacecraft_libraries.solvers.decentralized_mppi import (
    _build_line_of_sight_graph_with_degree, _max_consensus,
)
from spacecraft_libraries import new_opts

METHODS = [
    "centralized_nlp",
    "centralized_nlp_warm",
    "centralized_ga",
    "centralized_mppi",
    "decentralized_mppi",
]
MPPI_METHODS = {"centralized_mppi", "decentralized_mppi"}

TIME_LIMITS = [60.0, 300.0, 600.0, 1200.0]
# MPPI is deadline-driven: n_iter is set effectively unbounded and run_mppi's
# deadline_s check terminates at the wall clock, so MPPI consumes the same
# compute budget as the GA (which runs generations until its deadline).
# The schedule only sets the per-iteration sample batch size.
MPPI_DEADLINE_ITERS = 1_000_000
MPPI_SAMPLE_SCHEDULE = {
    300.0:  12,
    600.0:  17,
    1200.0: 24,
}

FIELDNAMES = [
    "scenario_id", "method", "time_limit_s",
    "cost", "terminal_violation", "runtime_s",
    "n_agents", "a", "e", "m", "tf", "epsilon_tol",
    "solver_seed", "tau_init_std",  # solver_seed: GA + both MPPI. tau_init_std: MPPI only.
    "mppi_sigma", "sigma_mode",     # MPPI only; sigma_mode: absolute | relative
    "noise_mode",                   # MPPI only: white | smooth (temporally correlated)
    "noise_knots",                  # MPPI only, meaningful when noise_mode=smooth
]

SEEDED_METHODS = {"centralized_ga"} | MPPI_METHODS


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


def _warm_nominal(sys_params, bc, epsilon, rng: np.random.Generator,
                  tau_init_scale: float) -> np.ndarray:
    """Multiple-shooting warm start — the SAME initialization family the GA's
    pop_gen_new uses for its population, so GA and MPPI start from the same
    information. tau_init_scale is the scale of the uniform random tau guess
    fed to the shooting IPOPT (GA default: 1e-1)."""
    return make_nominal_tau(sys_params, bc, epsilon, rng, tau_init_scale=tau_init_scale)


def run_centralized_mppi_warm_start(sys_params, bc, epsilon, n_iter, n_samples,
                                    sigma, lambda_, tau_init_std, seed,
                                    max_runtime_s, relative_sigma=False,
                                    noise_mode="white", noise_knots=8):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    N = sys_params.N

    oracle = ScenarioOracle(sys_params, bc, epsilon)
    nominal = _warm_nominal(sys_params, bc, epsilon, rng, tau_init_std)

    best_tau, best_cost, history = run_mppi(
        sys_params, bc, epsilon,
        nominal_tau=nominal,
        n_iter=n_iter, n_samples=n_samples,
        sigma=sigma, lambda_=lambda_,
        rng=rng,
        deadline_s=max_runtime_s,
        start_time=start,
        relative_sigma=relative_sigma,
        noise_mode=noise_mode,
        noise_knots=noise_knots,
        oracle=oracle,
    )

    traj, ctrl, q, cost = new_opts.opt_given_tau_ipopt_new(
        best_tau, N, epsilon, sys_params, bc, num_iter=1000
    )
    runtime = time.perf_counter() - start

    return {
        "method": "centralized_mppi",
        "tau": best_tau,
        "trajectory": traj,
        "control": ctrl,
        "attachment": q,
        "cost": float(cost),
        "runtime": runtime,
        "mppi": history.as_dict(),
    }


def run_decentralized_mppi_warm_start(sys_params, bc, epsilon, n_iter, n_samples,
                                      sigma, lambda_, tau_init_std, base_seed,
                                      max_runtime_s, line_of_sight_limit=100.0,
                                      graph_degree=None, relative_sigma=False,
                                      noise_mode="white", noise_knots=8):
    attach_vecs = np.asarray(sys_params.rs)
    num_agents = attach_vecs.shape[0]
    graph = _build_line_of_sight_graph_with_degree(attach_vecs, line_of_sight_limit, graph_degree)

    oracle = ScenarioOracle(sys_params, bc, epsilon)
    start = time.perf_counter()
    effective_limit = None if max_runtime_s is None else max_runtime_s * num_agents

    island_costs: list[float] = [float("inf")] * num_agents
    island_taus: list[np.ndarray] = [None] * num_agents
    island_histories: list[dict] = [None] * num_agents

    for i in range(num_agents):
        if effective_limit is not None and (time.perf_counter() - start) >= effective_limit:
            break
        rng = np.random.default_rng(base_seed + i)
        nominal = _warm_nominal(sys_params, bc, epsilon, rng, tau_init_std)
        # Each island gets its own slice of the serial budget (absolute
        # deadline (i+1)*max_runtime_s from start). With deadline-driven
        # n_iter this is what enforces per-agent budget parity with the
        # centralized run — a shared deadline would let island 0 consume
        # the entire 6x budget and starve the rest.
        island_deadline = None if max_runtime_s is None else (i + 1) * max_runtime_s
        best_tau, best_cost, history = run_mppi(
            sys_params, bc, epsilon,
            nominal_tau=nominal,
            n_iter=n_iter, n_samples=n_samples,
            sigma=sigma, lambda_=lambda_,
            rng=rng,
            deadline_s=island_deadline,
            start_time=start,
            relative_sigma=relative_sigma,
            noise_mode=noise_mode,
            noise_knots=noise_knots,
            oracle=oracle,
        )
        island_taus[i] = best_tau
        island_costs[i] = best_cost
        island_histories[i] = history.as_dict()

    for i in range(num_agents):
        if island_taus[i] is None:
            island_taus[i] = np.zeros((sys_params.N, 3))
            island_costs[i] = float("inf")
            island_histories[i] = {"n_iter_completed": 0, "n_samples_per_iter": n_samples,
                                   "best_cost_per_iter": [], "failed_sample_fraction": 1.0}

    _, _, winner_tau = _max_consensus(graph, island_costs, island_taus)

    # Final IPOPT through the winning tau (same finalization the GA does).
    winner_tau_proj = new_opts.tau_proj_nonlin_new(
        winner_tau, sys_params.N, epsilon, sys_params, bc
    )[0]
    winner_tau_proj = np.asarray(winner_tau_proj, dtype=float).reshape(sys_params.N, 3)
    traj, ctrl, q, cost = new_opts.opt_given_tau_ipopt_new(
        winner_tau_proj, sys_params.N, epsilon, sys_params, bc, num_iter=3000
    )
    runtime = time.perf_counter() - start

    return {
        "method": "decentralized_mppi",
        "tau": winner_tau_proj,
        "trajectory": traj,
        "control": ctrl,
        "attachment": q,
        "cost": float(cost),
        "runtime": runtime,
        "mppi": {
            "per_island_best_costs": [float(c) for c in island_costs],
            "per_island_history": island_histories,
            "num_agents": num_agents,
        },
    }


def run_centralized_ga_seeded(sys_params, bc, epsilon, pop_size, generations,
                               max_runtime_s, seed):
    # np.random.seed requires < 2**32; derive_solver_seed values exceed it
    # (scenario timestamp seeds * 10007 ~ 1.8e13), so reduce mod 2**32.
    np.random.seed(seed % (2**32))
    return solve_centralized_ga(sys_params, bc, epsilon, pop_size=pop_size,
                                 generations=generations, max_runtime_s=max_runtime_s)


def _extract_terminal_state(result: dict, method: str) -> dict:
    if method in ("centralized_nlp", "centralized_nlp_warm"):
        x = result["state"]
        return {"r": x[-1, 0:3], "v": x[-1, 3:6], "phi": x[-1, 6:9], "omega": x[-1, 9:12]}
    traj = result["trajectory"]
    s = traj.states[-1]
    return {"r": s.r, "v": s.v, "phi": s.phi, "omega": s.omega}


def run_one_solver(method: str, sys_params, bc, epsilon, max_runtime_s: float,
                    mppi_iterations: int, mppi_samples: int,
                    mppi_sigma: float, mppi_lambda: float,
                    tau_init_std: float, solver_seed: int,
                    relative_sigma: bool = False,
                    noise_mode: str = "white",
                    noise_knots: int = 8) -> dict:
    solvers = {
        "centralized_nlp": lambda: solve_centralized_nlp(
            sys_params, bc, max_iters=3000, max_runtime_s=max_runtime_s),
        "centralized_nlp_warm": lambda: solve_centralized_nlp_warm(
            sys_params, bc, max_iters=3000, max_runtime_s=max_runtime_s),
        "centralized_ga": lambda: run_centralized_ga_seeded(
            sys_params, bc, epsilon, pop_size=10, generations=5000,
            max_runtime_s=max_runtime_s, seed=solver_seed),
        "centralized_mppi": lambda: run_centralized_mppi_warm_start(
            sys_params, bc, epsilon,
            n_iter=mppi_iterations, n_samples=mppi_samples,
            sigma=mppi_sigma, lambda_=mppi_lambda, tau_init_std=tau_init_std,
            seed=solver_seed, max_runtime_s=max_runtime_s,
            relative_sigma=relative_sigma, noise_mode=noise_mode,
            noise_knots=noise_knots),
        "decentralized_mppi": lambda: run_decentralized_mppi_warm_start(
            sys_params, bc, epsilon,
            n_iter=mppi_iterations, n_samples=mppi_samples,
            sigma=mppi_sigma, lambda_=mppi_lambda, tau_init_std=tau_init_std,
            base_seed=solver_seed, max_runtime_s=max_runtime_s,
            relative_sigma=relative_sigma, noise_mode=noise_mode,
            noise_knots=noise_knots),
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
    p.add_argument("--mppi-iterations", type=int,   default=20)
    p.add_argument("--mppi-samples",    type=int,   default=None,
                    help="Batch size per MPPI iteration. Default: the "
                         "per-budget MPPI_SAMPLE_SCHEDULE; pass explicitly "
                         "to override the schedule.")
    p.add_argument("--mppi-sigma",      type=float, default=0.03,
                    help="Relative mode (default): fraction of warm-start "
                         "nominal RMS torque. 0.03 won the 2026-08-11 "
                         "relative-sigma sweep (best or near-best on 4/5 "
                         "scenarios, minimax regret).")
    p.add_argument("--sigma-mode", choices=["absolute", "relative"], default="relative",
                    help="'relative': --mppi-sigma is a fraction of the warm-"
                         "start nominal's RMS torque (transfers across scenario "
                         "scales); 'absolute': raw torque units (legacy).")
    p.add_argument("--noise-mode", choices=["white", "smooth"], default="white",
                    help="'smooth': temporally correlated MPPI perturbations "
                         "(Gaussian knots linearly interpolated across the "
                         "horizon) — low-dimensional smooth torque moves. "
                         "'white': independent per-timestep noise (legacy).")
    p.add_argument("--noise-knots", type=int, default=8,
                    help="Knot count for --noise-mode smooth: interpolation "
                         "anchors per axis. Few = smoother/lower-dim moves; "
                         "= N recovers white noise.")
    p.add_argument("--mppi-lambda",     type=float, default=1.0)
    p.add_argument("--tau-init-std",    type=float, default=0.1,
                    help="Scale of the uniform random tau guess fed to the "
                         "multiple-shooting warm start (make_nominal_tau). "
                         "0.1 matches the GA population init (pop_gen_new).")
    p.add_argument("--mppi-seed-fallback", type=int, default=42,
                    help="Used only if a scenario has no stored 'seed' field "
                         "(older scenario files generated without --seed)")
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

    mppi_iterations = MPPI_DEADLINE_ITERS
    if args.mppi_samples is not None:
        mppi_samples = args.mppi_samples
    else:
        mppi_samples = MPPI_SAMPLE_SCHEDULE.get(combo["time_limit"], 10)
    solver_seed = derive_solver_seed(combo["scenario"], combo["time_limit"], args.mppi_seed_fallback)
    is_mppi = combo["method"] in MPPI_METHODS
    if combo["method"] in SEEDED_METHODS:
        extra = f"  mppi_iterations={mppi_iterations}  mppi_samples={mppi_samples}  tau_init_std={args.tau_init_std}" if is_mppi else ""
        print(f"  solver_seed={solver_seed}{extra}")

    result = run_one_solver(
        combo["method"], sys_params, bc, epsilon,
        max_runtime_s=combo["time_limit"],
        mppi_iterations=mppi_iterations,
        mppi_samples=mppi_samples,
        mppi_sigma=args.mppi_sigma,
        mppi_lambda=args.mppi_lambda,
        tau_init_std=args.tau_init_std,
        solver_seed=solver_seed,
        relative_sigma=(args.sigma_mode == "relative"),
        noise_mode=args.noise_mode,
        noise_knots=args.noise_knots,
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
        "tau_init_std": args.tau_init_std if is_mppi else "",
        "mppi_sigma":   args.mppi_sigma if is_mppi else "",
        "sigma_mode":   args.sigma_mode if is_mppi else "",
        "noise_mode":   args.noise_mode if is_mppi else "",
        "noise_knots":  args.noise_knots if is_mppi else "",
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