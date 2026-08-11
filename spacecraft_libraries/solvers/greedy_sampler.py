"""Greedy Sampler (GS): stochastic hill-climbing on the feasible torque
manifold — the architecture the 2026-08 MPPI tuning marathon converged to,
promoted to a first-class method.

Identical machinery to MPPI (warm start, smooth temporally-correlated
perturbations, relative sigma, per-sample projection, deadline-driven loop)
with one difference: no softmin. Each iteration the nominal JUMPS to the
batch-best projected sample (run_mppi selection="greedy"). This is the
lambda -> 0 limit of MPPI made exact, which every lambda sweep (before and
after the projected-update fix) showed to be the optimal regime on this
problem class.

Defaults bake in the tuned recipe: relative sigma 0.03, WHITE noise (chosen
as the production default 2026-08-11 for simplicity — no knot parameter;
smooth noise remains available via noise_mode="smooth" and is slightly better
on polishing scenarios), multiple-shooting warm start at tau_init_scale 0.1,
step_size 1.0 (full greedy jump), parametric oracle.
"""
from __future__ import annotations

import time

import numpy as np

from .. import new_opts
from ..data_structures import BoundaryConditions, SystemParams
from .decentralized_mppi import _build_line_of_sight_graph_with_degree, _max_consensus
from .mppi_core import make_nominal_tau, run_mppi
from .parametric_oracle import ScenarioOracle

DEADLINE_ITERS = 1_000_000


def solve_centralized_gs(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    n_samples: int = 12,
    sigma: float = 0.03,
    seed: int = 42,
    tau_init_scale: float = 0.1,
    noise_mode: str = "white",
    noise_knots: int = 8,
    step_size: float = 1.0,
    max_runtime_s: float | None = None,
    use_oracle: bool = True,
):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    oracle = ScenarioOracle(sys_params, bc, epsilon) if use_oracle else None
    nominal = make_nominal_tau(sys_params, bc, epsilon, rng, tau_init_scale=tau_init_scale)

    best_tau, best_cost, history = run_mppi(
        sys_params, bc, epsilon,
        nominal_tau=nominal,
        n_iter=DEADLINE_ITERS, n_samples=n_samples,
        sigma=sigma, lambda_=1.0,  # lambda unused in greedy selection
        rng=rng,
        deadline_s=max_runtime_s, start_time=start,
        relative_sigma=True,
        noise_mode=noise_mode, noise_knots=noise_knots,
        oracle=oracle,
        selection="greedy",
        step_size=step_size,
    )

    traj, ctrl, q, cost = new_opts.opt_given_tau_ipopt_new(
        best_tau, sys_params.N, epsilon, sys_params, bc, num_iter=1000)
    runtime = time.perf_counter() - start

    return {
        "method": "centralized_gs",
        "tau": best_tau,
        "trajectory": traj,
        "control": ctrl,
        "attachment": q,
        "cost": float(cost),
        "runtime": runtime,
        "gs": history.as_dict(),
    }


def solve_decentralized_gs(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    n_samples: int = 12,
    sigma: float = 0.03,
    base_seed: int = 42,
    tau_init_scale: float = 0.1,
    noise_mode: str = "white",
    noise_knots: int = 8,
    step_size: float = 1.0,
    max_runtime_s: float | None = None,
    line_of_sight_limit: float = 100.0,
    graph_degree=None,
    use_oracle: bool = True,
):
    """One greedy-sampler island per agent (seed base_seed+i, absolute
    per-island budget slice (i+1)*max_runtime_s), then line-of-sight
    max-consensus on the best island. Same budget convention as the other
    decentralized solvers (effective wall = budget * num_agents, serial)."""
    attach_vecs = np.asarray(sys_params.rs)
    num_agents = attach_vecs.shape[0]
    graph = _build_line_of_sight_graph_with_degree(
        attach_vecs, line_of_sight_limit, graph_degree)

    oracle = ScenarioOracle(sys_params, bc, epsilon) if use_oracle else None
    start = time.perf_counter()
    effective_limit = None if max_runtime_s is None else max_runtime_s * num_agents

    island_costs = [float("inf")] * num_agents
    island_taus = [None] * num_agents
    island_histories = [None] * num_agents

    for i in range(num_agents):
        if effective_limit is not None and (time.perf_counter() - start) >= effective_limit:
            break
        rng = np.random.default_rng(base_seed + i)
        nominal = make_nominal_tau(sys_params, bc, epsilon, rng,
                                   tau_init_scale=tau_init_scale)
        island_deadline = None if max_runtime_s is None else (i + 1) * max_runtime_s
        best_tau, best_cost, history = run_mppi(
            sys_params, bc, epsilon,
            nominal_tau=nominal,
            n_iter=DEADLINE_ITERS, n_samples=n_samples,
            sigma=sigma, lambda_=1.0,
            rng=rng,
            deadline_s=island_deadline, start_time=start,
            relative_sigma=True,
            noise_mode=noise_mode, noise_knots=noise_knots,
            oracle=oracle,
            selection="greedy",
            step_size=step_size,
        )
        island_taus[i] = best_tau
        island_costs[i] = best_cost
        island_histories[i] = history.as_dict()

    for i in range(num_agents):
        if island_taus[i] is None:
            island_taus[i] = np.zeros((sys_params.N, 3))
            island_costs[i] = float("inf")
            island_histories[i] = {"n_iter_completed": 0, "n_samples_per_iter": n_samples,
                                   "best_cost_per_iter": [], "failed_sample_fraction": 1.0,
                                   "sigma_abs": None}

    winner_id, _, winner_tau = _max_consensus(graph, island_costs, island_taus)

    # winner_tau is already projected (best_tau from run_mppi is the projected
    # best sample) — finalize with the inner solve directly, symmetric with
    # the centralized contract (no redundant re-projection).
    traj, ctrl, q, cost = new_opts.opt_given_tau_ipopt_new(
        winner_tau, sys_params.N, epsilon, sys_params, bc, num_iter=1000)
    runtime = time.perf_counter() - start

    return {
        "method": "decentralized_gs",
        "tau": winner_tau,
        "trajectory": traj,
        "control": ctrl,
        "attachment": q,
        "cost": float(cost),
        "runtime": runtime,
        "gs": {
            "per_island_best_costs": [float(c) for c in island_costs],
            "per_island_history": island_histories,
            "winner_island": int(winner_id),
            "num_agents": num_agents,
        },
    }
