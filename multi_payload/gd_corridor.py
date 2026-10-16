"""COT envelope-gradient solver with a translational corridor (modified copy of
spacecraft_libraries/solvers/gradient_descent.py).

Identical algorithm to the paper's GD (multi-start projected gradient descent
on the body-frame torque trajectory, exact SOCP inner, envelope gradient,
attitude-feasibility projection through ScenarioOracle.project, relative step
size with backtracking), with three changes:
  * every inner solve is socp_corridor.solve_inner_socp with the payload's
    corridor (rbar, eps, rho) and waypoints;
  * initial_tau seeds the first start (and island 0 of the decentralized
    variant), so the oracle starts from the allocation's attitude reference;
  * the best inner solution is kept as a dict (fuel, slack, U, r, v) instead
    of the packed IPOPT vector.
"""
from __future__ import annotations

import time

import numpy as np

from spacecraft_libraries.data_structures import BoundaryConditions, SystemParams
from spacecraft_libraries.solvers.mppi_core import make_nominal_tau
from spacecraft_libraries.solvers.parametric_oracle import ScenarioOracle

from .socp_corridor import envelope_grad_socp, solve_inner_socp


def solve_centralized_gd(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    corridor=None,
    waypoints=None,
    seed: int = 42,
    tau_init_scale: float = 0.1,
    rel_step: float = 0.03,
    step_grow: float = 1.5,
    step_shrink: float = 0.5,
    min_rel_step: float = 1e-5,
    max_runtime_s: float | None = None,
    restarts: bool = True,
    max_restarts: int | None = None,
    initial_tau=None,
    random_restart_scale: float = 1.0,
):
    """Multi-start projected gradient descent with a corridor. See the
    original solve_centralized_gd for the algorithm; arguments are the same
    plus corridor/waypoints (passed to every inner solve)."""
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    def time_left():
        return max_runtime_s is None or (time.perf_counter() - start) < max_runtime_s

    oracle = ScenarioOracle(sys_params, bc, epsilon, max_solve_cpu_s=max_runtime_s)
    history = {'n_inner': 0, 'n_proj': 0, 'grad_norm': [], 'restart_J': [],
               'restart_iters': []}

    def inner(tau):
        history['n_inner'] += 1
        return solve_inner_socp(sys_params, bc, tau, corridor=corridor,
                                waypoints=waypoints)

    def _descend(start_tau=None):
        oracle.reset_warm()
        if start_tau is not None:
            nominal = np.asarray(start_tau, dtype=float).reshape(sys_params.N, 3)
        else:
            nominal = make_nominal_tau(sys_params, bc, epsilon, rng,
                                       tau_init_scale=tau_init_scale)
        if 'ref_rms' not in history:
            history['ref_rms'] = max(float(np.sqrt(np.mean(nominal ** 2))), 1e-9)
        tau, _ = oracle.project(nominal)
        history['n_proj'] += 1
        if tau is None:
            tau = nominal
        sol = inner(tau)
        if not sol.get('ok'):
            return tau, float('inf'), None, 0
        J = sol['J']
        tau_scale = max(float(np.sqrt(np.mean(tau ** 2))), 1e-9)
        rel = rel_step
        iters = 0
        while time_left():
            grad = envelope_grad_socp(sys_params, bc, tau, sol)
            gnorm = float(np.linalg.norm(grad))
            history['grad_norm'].append(gnorm)
            if gnorm <= 1e-14:
                break
            direction = grad / gnorm
            improved = False
            while rel >= min_rel_step and time_left():
                step = rel * tau_scale * np.sqrt(tau.size)
                tau_cand_p, _ = oracle.project(tau - step * direction.reshape(tau.shape))
                history['n_proj'] += 1
                if tau_cand_p is None:
                    rel *= step_shrink
                    continue
                sol_c = inner(tau_cand_p)
                if sol_c.get('ok') and sol_c['J'] < J - 1e-12:
                    tau, J, sol = tau_cand_p, sol_c['J'], sol_c
                    tau_scale = max(float(np.sqrt(np.mean(tau ** 2))), 1e-9)
                    rel = min(rel * step_grow, 1.0)
                    improved = True
                    break
                rel *= step_shrink
            iters += 1
            if not improved:
                break
        return tau, J, sol, iters

    best_tau, best_J, best_sol = None, float('inf'), None
    n_starts = 0
    while time_left() and (max_restarts is None or n_starts < max_restarts):
        if n_starts == 0:
            start_tau = initial_tau
        else:
            ref = (max(float(np.sqrt(np.mean(best_tau ** 2))), 1e-9)
                   if best_tau is not None else history.get('ref_rms', 1.0))
            start_tau = rng.normal(0.0, random_restart_scale * ref, (sys_params.N, 3))
        tau_s, J_s, sol_s, iters_s = _descend(start_tau)
        n_starts += 1
        history['restart_J'].append(float(J_s))
        history['restart_iters'].append(iters_s)
        if J_s < best_J:
            best_tau, best_J, best_sol = tau_s, J_s, sol_s
        if not restarts:
            break

    if best_sol is None:
        raise RuntimeError("gradient descent: every start failed to produce a finite cost")
    return _package('centralized_gd', best_tau, best_J, best_sol,
                    time.perf_counter() - start, history, n_starts)


def _package(method, tau, J, sol, runtime, history, n_starts):
    return {
        'method': method,
        'tau': tau,
        'control': sol['U'],
        'r': sol['r'],
        'v': sol['v'],
        'cost': float(J),
        'fuel': sol['fuel'],
        'slack': sol['slack'],
        'runtime': runtime,
        'gd': {
            'restarts': n_starts,
            'iterations': int(np.sum(history['restart_iters'])),
            'inner_solves': history['n_inner'],
            'projections': history['n_proj'],
            'restart_J': history['restart_J'],
            'final_grad_norm': history['grad_norm'][-1] if history['grad_norm'] else None,
        },
    }


def _island_worker(packed):
    (i, sys_params, bc, epsilon, corridor, waypoints, seed, tau_init_scale,
     rel_step, max_runtime_s, max_restarts, init_tau, random_restart_scale) = packed
    try:
        return i, solve_centralized_gd(
            sys_params, bc, epsilon, corridor=corridor, waypoints=waypoints,
            seed=seed, tau_init_scale=tau_init_scale, rel_step=rel_step,
            max_runtime_s=max_runtime_s, restarts=max_restarts != 1,
            max_restarts=max_restarts, initial_tau=init_tau,
            random_restart_scale=random_restart_scale)
    except Exception:
        return i, None


def solve_decentralized_gd(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    corridor=None,
    waypoints=None,
    base_seed: int = 42,
    tau_init_scale: float = 0.1,
    rel_step: float = 0.03,
    max_runtime_s: float | None = None,
    max_restarts_per_island: int | None = None,
    line_of_sight_limit: float = 100.0,
    graph_degree=None,
    random_restart_scale: float = 1.0,
    initial_tau=None,
    parallel: bool = False,
):
    """Decentralized GD with a corridor: one island per attached agent, then
    line-of-sight max-consensus (same convention as the paper's D-GD: islands
    run serially unless parallel=True, so wall ~ max_runtime_s * agents).
    Island 0 starts from initial_tau (or the shooting nominal); islands 1+
    from random torques with their own seeds."""
    from spacecraft_libraries.solvers.decentralized_mppi import (
        _build_line_of_sight_graph_with_degree, _max_consensus)

    attach = np.asarray(sys_params.rs)
    A = attach.shape[0]
    graph = _build_line_of_sight_graph_with_degree(attach, line_of_sight_limit, graph_degree)
    start = time.perf_counter()

    if initial_tau is not None:
        ref_rms = max(float(np.sqrt(np.mean(np.asarray(initial_tau) ** 2))), 1e-9)
    else:
        nominal0 = make_nominal_tau(sys_params, bc, epsilon, np.random.default_rng(base_seed),
                                    tau_init_scale=tau_init_scale)
        ref_rms = max(float(np.sqrt(np.mean(nominal0 ** 2))), 1e-9)

    def init_tau(i):
        if i == 0:
            return initial_tau
        return np.random.default_rng(base_seed + i).normal(
            0.0, random_restart_scale * ref_rms, (sys_params.N, 3))

    costs = [float('inf')] * A
    results = [None] * A
    if parallel:
        import concurrent.futures as cf
        import os
        jobs = [(i, sys_params, bc, epsilon, corridor, waypoints, base_seed + i,
                 tau_init_scale, rel_step, max_runtime_s, max_restarts_per_island,
                 init_tau(i), random_restart_scale) for i in range(A)]
        with cf.ProcessPoolExecutor(max_workers=min(A, max(1, (os.cpu_count() or 2) - 1))) as pool:
            for i, res in pool.map(_island_worker, jobs):
                if res is not None:
                    costs[i], results[i] = res['cost'], res
    else:
        for i in range(A):
            remaining = None
            if max_runtime_s is not None:
                remaining = (i + 1) * max_runtime_s - (time.perf_counter() - start)
                if remaining <= 0:
                    continue
            i_, res = _island_worker((i, sys_params, bc, epsilon, corridor, waypoints,
                                      base_seed + i, tau_init_scale, rel_step, remaining,
                                      max_restarts_per_island, init_tau(i),
                                      random_restart_scale))
            if res is not None:
                costs[i], results[i] = res['cost'], res

    taus = [(r['tau'] if r is not None else np.zeros((sys_params.N, 3))) for r in results]
    winner, _, _ = _max_consensus(graph, costs, taus)
    if results[winner] is None:
        raise RuntimeError("decentralized GD: no island produced a solution")
    out = dict(results[winner])
    out['method'] = 'decentralized_gd'
    out['runtime'] = time.perf_counter() - start
    out['gd'] = dict(out['gd'], per_island_costs=[float(c) for c in costs],
                     winner_island=int(winner), num_agents=A)
    return out
