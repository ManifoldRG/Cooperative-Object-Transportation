"""MPPI hyperparameter sensitivity study — fast grid (~2 hours).

Sweeps sigma and tau_init_std across 3 Monte Carlo scenarios for both
centralized and decentralized MPPI. Calls run_mppi directly to inject
N(0, tau_init_std) nominal taus.

Usage:
    python mppi_sensitivity_fast.py [--output results/mppi_sensitivity_fast.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from spacecraft_libraries.evaluation.comparison import random_scenario_generator
from spacecraft_libraries.evaluation.metrics import terminal_violation, lie_attitude_violation
from spacecraft_libraries.solvers.mppi_core import run_mppi
from spacecraft_libraries import new_opts


N_MC_SCENARIOS = 3
SIGMA_VALS     = np.linspace(0.3, 0.7, 5).tolist()
LAMBDA_VAL     = 1.0
N_ITER         = 5
N_SAMPLES      = 10
TAU_INIT_STDS  = [0.01, 2.0]
N_REPEATS      = 1
METHODS        = ["centralized_mppi", "decentralized_mppi"]

FIELDNAMES = [
    "scenario", "method", "repeat",
    "sigma", "lambda_", "n_iter", "n_samples", "tau_init_std",
    "cost", "terminal_violation", "runtime_s",
    "failed_sample_fraction", "n_iter_completed",
]


def _violation(traj, bc) -> float:
    s = traj.states[-1]
    return float(
        terminal_violation(s.r, bc.xf.r)
        + terminal_violation(s.v, bc.xf.v)
        + lie_attitude_violation(s.phi, bc.xf.phi)
        + terminal_violation(s.omega, bc.xf.omega)
    )


def _nan_row(**kwargs) -> dict:
    row = {k: float("nan") for k in FIELDNAMES}
    row.update(kwargs)
    return row


def _run_centralized(sys_params, bc, epsilon, nominal, rng, sigma):
    t0 = time.perf_counter()
    best_tau, best_cost, history = run_mppi(
        sys_params, bc, epsilon, nominal,
        n_iter=N_ITER, n_samples=N_SAMPLES,
        sigma=sigma, lambda_=LAMBDA_VAL, rng=rng,
    )
    traj, ctrl, q, cost = new_opts.opt_given_tau_ipopt_new(
        best_tau, sys_params.N, epsilon, sys_params, bc, num_iter=3000
    )
    return traj, float(cost), time.perf_counter() - t0, history.as_dict()


def _run_decentralized(sys_params, bc, epsilon, rng, sigma, tau_init_std):
    from scipy.spatial.distance import cdist
    import networkx as nx

    attach_vecs = np.asarray(sys_params.rs)
    num_agents = attach_vecs.shape[0]
    N = sys_params.N

    distances = cdist(attach_vecs, attach_vecs)
    g = nx.Graph()
    g.add_nodes_from(range(num_agents))
    for i in range(num_agents):
        neighbors = np.where((distances[i] < 100.0) & (distances[i] > 0))[0]
        g.add_edges_from((i, int(j)) for j in neighbors)

    island_costs, island_taus, island_histories = [], [], []
    t0 = time.perf_counter()

    for i in range(num_agents):
        island_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        nominal = island_rng.normal(0.0, tau_init_std, size=(N, 3))
        best_tau, best_cost, history = run_mppi(
            sys_params, bc, epsilon, nominal,
            n_iter=N_ITER, n_samples=N_SAMPLES,
            sigma=sigma, lambda_=LAMBDA_VAL, rng=island_rng,
        )
        island_taus.append(best_tau)
        island_costs.append(best_cost)
        island_histories.append(history.as_dict())

    # Max-consensus
    state = {
        i: (1.0 / float(island_costs[i])
            if np.isfinite(island_costs[i]) and island_costs[i] > 0 else 0.0, i)
        for i in range(num_agents)
    }
    for _ in range(100):
        updated = {
            i: max([state[i]] + [state[j] for j in g.neighbors(i)], key=lambda t: t[0])
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
    traj, ctrl, q, cost = new_opts.opt_given_tau_ipopt_new(
        winner_tau_proj, N, epsilon, sys_params, bc, num_iter=3000
    )

    return traj, float(cost), time.perf_counter() - t0, island_histories[winner_id]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=Path("results/mppi_sensitivity_fast_plussave.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {N_MC_SCENARIOS} random scenarios...", flush=True)
    mc_scenarios = []
    scenario_log = []
    for i in range(N_MC_SCENARIOS):
        sys_p, bc, eps = random_scenario_generator()
        mc_scenarios.append((i + 1, sys_p, bc, eps))
        print(f"  Scenario {i+1}: N={sys_p.N}, n_agents={len(sys_p.rs)}, tf={bc.tf:.1f}s",
              flush=True)
        # Record all scenario params for reproducibility
        scenario_log.append({
            "scenario_id": i + 1,
            "mu": sys_p.mu,
            "a": sys_p.a,
            "e": sys_p.e,
            "nu": sys_p.nu,
            "m": sys_p.m,
            "I_diag": np.diag(sys_p.I).tolist(),
            "N": sys_p.N,
            "tf": bc.tf,
            "rs": [r.tolist() for r in sys_p.rs],
            "n_agents": len(sys_p.rs),
            "x0_r": bc.x0.r.tolist(),
            "x0_v": bc.x0.v.tolist(),
            "x0_phi": bc.x0.phi.tolist(),
            "x0_omega": bc.x0.omega.tolist(),
            "xf_r": bc.xf.r.tolist(),
            "xf_v": bc.xf.v.tolist(),
            "xf_phi": bc.xf.phi.tolist(),
            "xf_omega": bc.xf.omega.tolist(),
            "epsilon": eps,
        })

    # Save scenario params to JSON for reproducibility
    import json
    scenario_log_path = args.output.parent / (args.output.stem + "_scenarios.json")
    with scenario_log_path.open("w") as f:
        json.dump(scenario_log, f, indent=2)
    print(f"Scenario params saved -> {scenario_log_path}", flush=True)

    total_runs = N_MC_SCENARIOS * len(SIGMA_VALS) * len(TAU_INIT_STDS) * N_REPEATS * len(METHODS)
    run_idx = 0
    print(f"Total runs: {total_runs}", flush=True)

    with args.output.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    for (scenario, sys_params, bc, epsilon), sigma, tau_init_std in product(
        mc_scenarios, SIGMA_VALS, TAU_INIT_STDS,
    ):
        N = sys_params.N

        for repeat in range(N_REPEATS):
            seed = repeat * 1000 + scenario * 100
            rng = np.random.default_rng(seed)

            for method in METHODS:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] sc={scenario} {method} "
                    f"sigma={sigma} tau_std={tau_init_std} rep={repeat}",
                    flush=True,
                )

                base = dict(
                    scenario=scenario, method=method, repeat=repeat,
                    sigma=sigma, lambda_=LAMBDA_VAL, n_iter=N_ITER,
                    n_samples=N_SAMPLES, tau_init_std=tau_init_std,
                )
                row = _nan_row(**base)

                try:
                    nominal = rng.normal(0.0, tau_init_std, size=(N, 3))

                    if method == "centralized_mppi":
                        traj, cost, rt, mppi_info = _run_centralized(
                            sys_params, bc, epsilon, nominal, rng, sigma,
                        )
                    else:
                        traj, cost, rt, mppi_info = _run_decentralized(
                            sys_params, bc, epsilon, rng, sigma, tau_init_std,
                        )

                    row["cost"] = cost
                    row["runtime_s"] = rt
                    row["terminal_violation"] = _violation(traj, bc)
                    row["failed_sample_fraction"] = float(
                        mppi_info.get("failed_sample_fraction", float("nan"))
                    )
                    n_iter_val = mppi_info.get("n_iter_completed", float("nan"))
                    row["n_iter_completed"] = (
                        int(n_iter_val)
                        if not np.isnan(float(n_iter_val))
                        else float("nan")
                    )

                except Exception:
                    traceback.print_exc()

                with args.output.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f"\nDone. Results -> {args.output}")


if __name__ == "__main__":
    main()