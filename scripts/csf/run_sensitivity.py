#!/usr/bin/env python3
"""Single MPPI sensitivity task — called by SLURM job array.
Each task gets a unique TASK_ID mapping to a specific
(scenario, sigma, tau_init_std, gamma, method) combination.
Usage:
    python run_sensitivity_task.py --task-id 1 \
        --scenarios results/scenarios.json \
        --output-dir results/tasks
"""
from __future__ import annotations
import argparse, csv, json, sys, time, traceback
from itertools import product
from pathlib import Path

# scripts/csf/ -> scripts/ -> repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from spacecraft_libraries.data_structures import (
    BoundaryConditions, SystemParams, StateVectorLie,
)
from spacecraft_libraries.evaluation.metrics import terminal_violation, lie_attitude_violation
from spacecraft_libraries.solvers.mppi_core import run_mppi
from spacecraft_libraries import new_opts

# Total: 3 × 3 × 3 × 3 × 3 × 2 = 162 runs
SIGMA_VALS    = [0.1, 0.25, 0.5]
TAU_VALS      = [0.1, 0.5, 1.0]
LAMBDA_VALS   = [0.8, 0.9, 1.0]
GAMMA_CONFIGS = [
    dict(gamma=0.5, n_iter=50, n_samples=100),
    dict(gamma=1.0, n_iter=50, n_samples=50),
    dict(gamma=2.0, n_iter=100, n_samples=50),
]
METHODS    = ["centralized_mppi", "decentralized_mppi"]

FIELDNAMES = [
    "scenario_id", "method",
    "sigma", "lambda_", "gamma", "n_iter", "n_samples", "tau_init_std",
    "cost", "terminal_violation", "runtime_s",
    "failed_sample_fraction", "n_iter_completed",
]


def build_combos(scenarios):
    combos = []
    for sc, sigma, tau, lambda_, gcfg, method in product(
        scenarios, SIGMA_VALS, TAU_VALS, LAMBDA_VALS, GAMMA_CONFIGS, METHODS
    ):
        combos.append(dict(
            scenario_id=sc["scenario_id"],
            scenario=sc,
            sigma=sigma,
            tau_init_std=tau,
            lambda_=lambda_,
            gamma=gcfg["gamma"],
            n_iter=gcfg["n_iter"],
            n_samples=gcfg["n_samples"],
            method=method,
        ))
    return combos


def load_scenarios(path: Path):
    with path.open() as f:
        return json.load(f)


def make_sys_bc(s):
    sys_params = SystemParams(
        mu=s["mu"], a=s["a"], e=s["e"], nu=s["nu"],
        I=np.diag(s["I_diag"]), m=s["m"],
        rs=[np.array(r) for r in s["rs"]], N=s["N"],
    )
    bc = BoundaryConditions(
        x0=StateVectorLie(
            r=np.array(s["x0_r"]), v=np.array(s["x0_v"]),
            phi=np.array(s["x0_phi"]), omega=np.array(s["x0_omega"])),
        xf=StateVectorLie(
            r=np.array(s["xf_r"]), v=np.array(s["xf_v"]),
            phi=np.array(s["xf_phi"]), omega=np.array(s["xf_omega"])),
        tf=s["tf"],
    )
    return sys_params, bc, s["epsilon"]


def _violation(traj, bc):
    s = traj.states[-1]
    return float(
        terminal_violation(s.r, bc.xf.r)
        + terminal_violation(s.v, bc.xf.v)
        + lie_attitude_violation(s.phi, bc.xf.phi)
        + terminal_violation(s.omega, bc.xf.omega)
    )


def _run_centralized(sys_params, bc, epsilon, nominal, rng, combo):
    t0 = time.perf_counter()
    best_tau, best_cost, history = run_mppi(
        sys_params, bc, epsilon, nominal,
        n_iter=combo["n_iter"], n_samples=combo["n_samples"],
        sigma=combo["sigma"], lambda_=combo["lambda_"], rng=rng,
    )
    traj, _, _, cost = new_opts.opt_given_tau_ipopt_new(
        best_tau, sys_params.N, epsilon, sys_params, bc, num_iter=1000
    )
    return traj, float(cost), time.perf_counter() - t0, history.as_dict()


def _run_decentralized(sys_params, bc, epsilon, rng, combo):
    from scipy.spatial.distance import cdist
    import networkx as nx

    attach_vecs = np.asarray(sys_params.rs)
    num_agents  = attach_vecs.shape[0]
    N           = sys_params.N
    t0          = time.perf_counter()

    distances = cdist(attach_vecs, attach_vecs)
    g = nx.Graph()
    g.add_nodes_from(range(num_agents))
    for i in range(num_agents):
        neighbors = np.where((distances[i] < 100.0) & (distances[i] > 0))[0]
        g.add_edges_from((i, int(j)) for j in neighbors)

    island_costs, island_taus, island_histories = [], [], []
    for i in range(num_agents):
        island_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        nominal = island_rng.normal(0.0, combo["tau_init_std"], size=(N, 3))
        best_tau, best_cost, history = run_mppi(
            sys_params, bc, epsilon, nominal,
            n_iter=combo["n_iter"], n_samples=combo["n_samples"],
            sigma=combo["sigma"], lambda_=combo["lambda_"], rng=island_rng,
        )
        island_taus.append(best_tau)
        island_costs.append(best_cost)
        island_histories.append(history.as_dict())

    state = {
        i: (1.0 / float(island_costs[i])
            if np.isfinite(island_costs[i]) and island_costs[i] > 0 else 0.0, i)
        for i in range(num_agents)
    }
    for _ in range(100):
        updated = {
            i: max([state[i]] + [state[j] for j in g.neighbors(i)],
                   key=lambda t: t[0])
            for i in range(num_agents)
        }
        if updated == state:
            break
        state = updated
    _, winner_id = next(iter(state.values()))

    winner_tau_proj = new_opts.tau_proj_nonlin_new(
        island_taus[winner_id], N, epsilon, sys_params, bc
    )[0]
    winner_tau_proj = np.asarray(winner_tau_proj, dtype=float).reshape(N, 3)
    traj, _, _, cost = new_opts.opt_given_tau_ipopt_new(
        winner_tau_proj, N, epsilon, sys_params, bc, num_iter=1000
    )
    return traj, float(cost), time.perf_counter() - t0, island_histories[winner_id]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id",    type=int,  required=True)
    p.add_argument("--scenarios",  type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/tasks"))
    return p.parse_args()


def main():
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
          f"sigma={combo['sigma']} tau={combo['tau_init_std']} "
          f"gamma={combo['gamma']} n_iter={combo['n_iter']} "
          f"n_samples={combo['n_samples']}", flush=True)

    sys_params, bc, epsilon = make_sys_bc(combo["scenario"])
    N   = sys_params.N
    rng = np.random.default_rng(args.task_id * 42)

    row = {k: float("nan") for k in FIELDNAMES}
    row.update(dict(
        scenario_id=combo["scenario_id"], method=combo["method"],
        sigma=combo["sigma"], lambda_=combo["lambda_"],
        gamma=combo["gamma"], n_iter=combo["n_iter"],
        n_samples=combo["n_samples"], tau_init_std=combo["tau_init_std"],
    ))

    try:
        nominal = rng.normal(0.0, combo["tau_init_std"], size=(N, 3))
        if combo["method"] == "centralized_mppi":
            traj, cost, rt, mppi_info = _run_centralized(
                sys_params, bc, epsilon, nominal, rng, combo)
        else:
            traj, cost, rt, mppi_info = _run_decentralized(
                sys_params, bc, epsilon, rng, combo)

        row["cost"]               = cost
        row["runtime_s"]          = rt
        row["terminal_violation"] = _violation(traj, bc)
        row["failed_sample_fraction"] = float(
            mppi_info.get("failed_sample_fraction", float("nan")))
        n_iter_val = mppi_info.get("n_iter_completed", float("nan"))
        row["n_iter_completed"] = (
            int(n_iter_val) if not np.isnan(float(n_iter_val)) else float("nan"))
    except Exception:
        traceback.print_exc()

    out = args.output_dir / f"task_{args.task_id:04d}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerow(row)

    print(f"Done. cost={row['cost']}, runtime={row['runtime_s']:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()