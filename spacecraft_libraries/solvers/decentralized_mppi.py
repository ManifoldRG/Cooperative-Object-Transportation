"""Decentralized MPPI: per-agent MPPI islands + line-of-sight max-consensus.

Each agent owns an independent MPPI process with its own random seed and its
own nominal tau. After all islands finish, agents do max-consensus on
(1/cost, tau) across the same line-of-sight graph the decentralized GA uses,
then everyone finalizes the agreed-upon tau through the inner IPOPT.

Parallelization across islands is a future enhancement; the current loop is
serial to match the existing `decentralized_island_ga` pattern.
"""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from .. import new_opts
from ..data_structures import BoundaryConditions, SystemParams
from .mppi_core import make_nominal_tau, run_mppi


def _build_line_of_sight_graph(attach_vecs: np.ndarray, limit: float) -> nx.Graph:
    n = attach_vecs.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    distances = cdist(attach_vecs, attach_vecs)
    for i in range(n):
        neighbors = np.where((distances[i] < limit) & (distances[i] > 0))[0]
        g.add_edges_from((i, int(j)) for j in neighbors)
    return g


def _max_consensus(
    graph: nx.Graph,
    island_costs: list[float],
    island_taus: list[np.ndarray],
    max_iters: int = 100,
) -> tuple[int, float, np.ndarray]:
    """Max-consensus on (1/cost, owner_id) over the graph. Returns the global
    winner's (owner_id, cost, tau). Identical semantics to the GA's
    `GraphManager.run_consensus` loop.
    """
    state = {
        i: (1.0 / float(island_costs[i]) if np.isfinite(island_costs[i]) and island_costs[i] > 0 else 0.0, i)
        for i in graph.nodes
    }
    nodes = list(graph.nodes)
    for _ in range(max_iters):
        updated = {
            i: max([state[i]] + [state[j] for j in graph.neighbors(i)], key=lambda t: t[0])
            for i in nodes
        }
        if updated == state:
            break
        state = updated

    winner_fit, winner_id = next(iter(state.values()))
    return winner_id, island_costs[winner_id], island_taus[winner_id]


def solve_decentralized_mppi(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    n_iter: int = 5,
    n_samples: int = 10,
    sigma: float = 1e-1,
    lambda_: float = 1.0,
    base_seed: int = 42,
    line_of_sight_limit: float = 100.0,
    max_runtime_s: float | None = None,
):
    attach_vecs = np.asarray(sys_params.rs)
    num_agents = attach_vecs.shape[0]
    graph = _build_line_of_sight_graph(attach_vecs, line_of_sight_limit)

    start = time.perf_counter()
    # Mirror the GA's runtime-budget convention: scale by num_agents to model
    # ideal-parallelism wallclock equivalence. The loop is serial today.
    effective_limit = None if max_runtime_s is None else max_runtime_s * num_agents

    island_costs: list[float] = [float("inf")] * num_agents
    island_taus: list[np.ndarray] = [None] * num_agents
    island_histories: list[dict] = [None] * num_agents

    for i in range(num_agents):
        if effective_limit is not None and (time.perf_counter() - start) >= effective_limit:
            break
        rng = np.random.default_rng(base_seed + i)
        nominal = make_nominal_tau(sys_params, bc, epsilon, rng)
        best_tau, best_cost, history = run_mppi(
            sys_params,
            bc,
            epsilon,
            nominal_tau=nominal,
            n_iter=n_iter,
            n_samples=n_samples,
            sigma=sigma,
            lambda_=lambda_,
            rng=rng,
            deadline_s=effective_limit,
            start_time=start,
        )
        island_taus[i] = best_tau
        island_costs[i] = best_cost
        island_histories[i] = history.as_dict()

    # Replace any unfinished islands with sentinel-bad entries so consensus
    # gives them weight zero.
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
