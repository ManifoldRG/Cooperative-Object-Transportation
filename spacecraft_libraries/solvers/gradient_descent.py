"""Projected gradient descent on the body-frame torque trajectory.

Third solver family alongside the GA and MPPI: instead of sampling around a
nominal tau, descend the exact gradient of the inner thrust-allocation
problem's optimal value.

The inner problem (same as opt_given_tau_ipopt_new) is built ONCE per scenario
as a parametric NLP with tau as a CasADi parameter, INCLUDING the symbolic
attitude rollout R_k(tau) â€” so tau's two entry points (torque-allocation
equality rows and the rotation matrices in the translational dynamics) are
both captured. The gradient of the optimal value J(tau) then comes from the
envelope theorem:

    dJ/dtau = d/dtau [ f(x) + lam_g^T g(x; tau) ]  evaluated at (x*, lam_g*)

which needs only IPOPT's converged primal/dual solution and one symbolic
Jacobian â€” no differentiation through the solver, no extra solves.

Attitude terminal feasibility is NOT part of the inner problem (it constrains
r/v only), so like the GA/MPPI pipelines we keep tau on the attitude-feasible
manifold by re-projecting through tau_proj_nonlin_new â€” i.e. projected
gradient descent. The projection's derivative is ignored (standard PGD).

Line search: backtracking on the true objective (project + inner solve per
trial), growing the step on success. The initial step is RELATIVE to the
warm-start torque scale (alpha chosen so the first move is ~rel_step * RMS(tau)
along the normalized gradient), mirroring the relative-sigma lesson from the
MPPI tuning: absolute step sizes do not transfer across scenario scales.
"""
from __future__ import annotations

import os
import sys
import time

import casadi as ca
import numpy as np

from .. import new_opts
from ..data_structures import BoundaryConditions, SystemParams
from .mppi_core import make_nominal_tau
from .parametric_oracle import ScenarioOracle, build_inner_parametric  # noqa: F401 (re-export)


def _project(tau, sys_params, bc, epsilon):
    try:
        tau_proj = new_opts.tau_proj_nonlin_new(
            tau, sys_params.N, epsilon, sys_params, bc, allow_raising_error=True)[0]
    except Exception:
        return None
    if tau_proj is None:
        return None
    return np.asarray(tau_proj, dtype=float).reshape(sys_params.N, 3)


def solve_centralized_gd(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
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
    restart_perturb_rel: float = 0.0,
    random_restart_scale: float = 1.0,
    keep_outs=None,
):
    """Multi-start projected gradient descent.

    Each start draws a fresh multiple-shooting warm nominal, projects it, and
    descends to first-order stationarity; the best (tau, J) over all starts is
    returned. With a wall-clock budget this makes compute fungible: more
    budget = more restarts = more basins explored, which is exactly the same
    resource-exchange the decentralized islands give the samplers (a
    decentralized GD is just these restarts split across agents + consensus).

    rel_step: first trial step moves tau by ~rel_step * RMS(tau) along the
    normalized gradient (relative scaling, cf. relative sigma).
    restarts=False reproduces the single-start behavior.

    initial_tau: start the FIRST descent from this tau instead of a shooting
    nominal (enables replanning warm starts and basin probes).

    Restart starts (2+): RANDOM cold torques by default — Gaussian with std
    random_restart_scale * RMS(reference), reference = best tau so far (or
    the first nominal if no finite best). This is what makes restarts explore
    different basins: fresh shooting nominals are DETERMINISTIC in outcome
    (the shooting IPOPT converges to the same tau regardless of its random
    init — measured 2026-08-11), so pre-2026-08-17 restarts re-ran the
    identical descent and explored nothing. Best (tau, J) over all starts is
    kept. restart_perturb_rel > 0 overrides with local perturbation around
    best_tau (polish mode) instead of cold draws.
    """
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    def time_left():
        return max_runtime_s is None or (time.perf_counter() - start) < max_runtime_s

    oracle = ScenarioOracle(sys_params, bc, epsilon, keep_outs=keep_outs)

    history = {'n_inner': 0, 'n_proj': 0, 'grad_norm': [], 'restart_J': [],
               'restart_iters': []}

    def _descend(start_tau=None):
        """One start: nominal (or given tau) -> project -> descend to
        stationarity. Returns (tau, J, x_sol, iters) — J=inf if failed."""
        if start_tau is not None:
            nominal = np.asarray(start_tau, dtype=float).reshape(sys_params.N, 3)
        else:
            nominal = make_nominal_tau(sys_params, bc, epsilon, rng,
                                       tau_init_scale=tau_init_scale)
        if 'ref_rms' not in history:
            history['ref_rms'] = max(float(np.sqrt(np.mean(
                np.asarray(nominal, dtype=float) ** 2))), 1e-9)
        tau, _ = oracle.project(nominal)
        history['n_proj'] += 1
        if tau is None:
            tau = np.asarray(nominal, dtype=float).reshape(sys_params.N, 3)
        ok, J, x_sol, lam = oracle.inner_cost(tau)
        history['n_inner'] += 1
        if not ok or not np.isfinite(J):
            return tau, float('inf'), x_sol, 0

        tau_scale = max(float(np.sqrt(np.mean(tau ** 2))), 1e-9)
        rel = rel_step
        iters = 0
        while time_left():
            grad = oracle.grad(x_sol, lam, tau)
            gnorm = float(np.linalg.norm(grad))
            history['grad_norm'].append(gnorm)
            if gnorm <= 1e-14:
                break
            direction = grad / gnorm

            improved = False
            while rel >= min_rel_step and time_left():
                step = rel * tau_scale * np.sqrt(tau.size)
                tau_cand = tau - step * direction.reshape(tau.shape)
                tau_cand_p, _ = oracle.project(tau_cand)
                history['n_proj'] += 1
                if tau_cand_p is None:
                    rel *= step_shrink
                    continue
                ok_c, J_c, x_c, lam_c = oracle.inner_cost(tau_cand_p)
                history['n_inner'] += 1
                if ok_c and np.isfinite(J_c) and J_c < J - 1e-12:
                    tau, J, x_sol, lam = tau_cand_p, J_c, x_c, lam_c
                    tau_scale = max(float(np.sqrt(np.mean(tau ** 2))), 1e-9)
                    rel = min(rel * step_grow, 1.0)
                    improved = True
                    break
                rel *= step_shrink
            iters += 1
            if not improved:
                break  # line search exhausted: first-order stationary
        return tau, J, x_sol, iters

    best_tau, best_J, best_x = None, float('inf'), None
    n_starts = 0
    while time_left() and (max_restarts is None or n_starts < max_restarts):
        if n_starts == 0:
            start_tau = initial_tau
        elif restart_perturb_rel > 0 and best_tau is not None:
            scale = restart_perturb_rel * max(float(np.sqrt(np.mean(best_tau ** 2))), 1e-9)
            start_tau = best_tau + rng.normal(0.0, scale, best_tau.shape)
        else:
            # random cold start: explores basins (fresh shooting nominals are
            # deterministic in outcome and would just repeat descent 1)
            ref = (max(float(np.sqrt(np.mean(best_tau ** 2))), 1e-9)
                   if best_tau is not None else history.get('ref_rms', 1.0))
            start_tau = rng.normal(0.0, random_restart_scale * ref,
                                   (sys_params.N, 3))
        tau_s, J_s, x_s, iters_s = _descend(start_tau)
        n_starts += 1
        history['restart_J'].append(float(J_s))
        history['restart_iters'].append(iters_s)
        if J_s < best_J:
            best_tau, best_J, best_x = tau_s, J_s, x_s
        if not restarts:
            break

    tau, J, x_sol = best_tau, best_J, best_x
    runtime = time.perf_counter() - start

    if x_sol is None:
        raise RuntimeError("gradient descent: every start failed to produce a finite cost")

    return _package_gd_result('centralized_gd', tau, J, x_sol, runtime,
                              history, n_starts, sys_params)


def _package_gd_result(method, tau, J, x_sol, runtime, history, n_starts, sys_params):

    # Final trajectory from the best inner solution (r/v pinned by constraints);
    # attitude terminal comes from the projection, checked by the caller.
    num_steps = sys_params.N
    num_agents = len(sys_params.rs)
    nU = num_agents * num_steps * 3
    U_opt = x_sol[:nU].reshape(num_agents, num_steps, 3)
    r_opt = x_sol[nU:nU + (num_steps + 1) * 3].reshape(num_steps + 1, 3)
    v_opt = x_sol[nU + (num_steps + 1) * 3:].reshape(num_steps + 1, 3)

    return {
        'method': method,
        'tau': tau,
        'control': U_opt,
        'r': r_opt,
        'v': v_opt,
        'cost': float(J),
        'runtime': runtime,
        'gd': {
            'restarts': n_starts,
            'iterations': int(np.sum(history['restart_iters'])),
            'inner_solves': history['n_inner'],
            'projections': history['n_proj'],
            'restart_J': history['restart_J'],
            'restart_iters': history['restart_iters'],
            'final_grad_norm': history['grad_norm'][-1] if history['grad_norm'] else None,
        },
    }


def _island_worker(packed):
    """Top-level worker for parallel islands (must be picklable on Windows
    spawn). Returns (island_index, result_dict_or_None)."""
    (i, sys_params, bc, epsilon, seed, tau_init_scale, rel_step,
     max_runtime_s, max_restarts, init_tau, random_restart_scale,
     keep_outs) = packed
    try:
        res = solve_centralized_gd(
            sys_params, bc, epsilon, seed=seed,
            tau_init_scale=tau_init_scale, rel_step=rel_step,
            max_runtime_s=max_runtime_s,
            restarts=max_restarts != 1,
            max_restarts=max_restarts,
            initial_tau=init_tau,
            random_restart_scale=random_restart_scale,
            keep_outs=keep_outs)
        return i, res
    except Exception:
        return i, None


def solve_decentralized_gd(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    base_seed: int = 42,
    tau_init_scale: float = 0.1,
    rel_step: float = 0.03,
    max_runtime_s: float | None = None,
    max_restarts_per_island: int | None = None,
    line_of_sight_limit: float = 100.0,
    graph_degree=None,
    island_random_starts: bool = True,
    random_restart_scale: float = 1.0,
    keep_outs=None,
    parallel: bool = False,
):
    """Decentralized GD: one gradient-descent island per agent (seed
    base_seed+i, its own slice of the serial budget), then the same
    line-of-sight max-consensus the decentralized GA/MPPI use.

    Island 0 starts from the shooting warm start (the reliable single start);
    islands 1+ each start from a RANDOM cold torque drawn with their own seed
    (Gaussian, std = random_restart_scale * RMS(warm nominal)) — so on
    multimodal scenarios (tight cones) the islands genuinely explore
    different basins and consensus is a real hedge, not a formality. Islands
    with leftover budget also random-restart internally (post-convergence
    restarts keep best-so-far). island_random_starts=False restores the old
    all-warm-start behavior. Budget convention matches the other
    decentralized solvers: effective wall = max_runtime_s * num_agents,
    islands run serially here.
    """
    from .decentralized_mppi import (
        _build_line_of_sight_graph_with_degree, _max_consensus,
    )

    attach_vecs = np.asarray(sys_params.rs)
    num_agents = attach_vecs.shape[0]
    graph = _build_line_of_sight_graph_with_degree(
        attach_vecs, line_of_sight_limit, graph_degree)

    start = time.perf_counter()
    effective_limit = None if max_runtime_s is None else max_runtime_s * num_agents

    # Torque scale anchor for the random island starts: the (deterministic)
    # shooting warm nominal's RMS. Computed once; ~0.5 s.
    ref_rms = None
    if island_random_starts:
        nominal0 = make_nominal_tau(sys_params, bc, epsilon,
                                    np.random.default_rng(base_seed),
                                    tau_init_scale=tau_init_scale)
        ref_rms = max(float(np.sqrt(np.mean(np.asarray(nominal0) ** 2))), 1e-9)

    island_costs = [float('inf')] * num_agents
    island_results = [None] * num_agents

    def _island_init_tau(i):
        if island_random_starts and i > 0:
            rng_i = np.random.default_rng(base_seed + i)
            return rng_i.normal(0.0, random_restart_scale * ref_rms,
                                (sys_params.N, 3))
        return None  # island 0: shooting warm start

    if parallel:
        # True parallelism: every island gets the full per-agent budget
        # simultaneously (each agent is its own computer). Wall time ~=
        # max_runtime_s instead of num_agents * max_runtime_s.
        # DEFAULT OFF: CSF SLURM tasks are single-core and run many trials
        # in parallel — spawning 6 workers per trial would oversubscribe the
        # node. Enable for local interactive runs (parallel=True); Windows
        # callers need an `if __name__ == "__main__"` guard.
        import concurrent.futures as _cf
        import os as _os
        jobs = [(i, sys_params, bc, epsilon, base_seed + i, tau_init_scale,
                 rel_step, max_runtime_s, max_restarts_per_island,
                 _island_init_tau(i), random_restart_scale, keep_outs)
                for i in range(num_agents)]
        workers = min(num_agents, max(1, (_os.cpu_count() or 2) - 1))
        with _cf.ProcessPoolExecutor(max_workers=workers) as pool:
            for i, res in pool.map(_island_worker, jobs):
                if res is not None:
                    island_costs[i] = res['cost']
                    island_results[i] = res
    else:
        for i in range(num_agents):
            if effective_limit is not None and (time.perf_counter() - start) >= effective_limit:
                break
            remaining = None
            if max_runtime_s is not None:
                # Absolute per-island slice (i+1)*budget from start, cf. dMPPI.
                remaining = (i + 1) * max_runtime_s - (time.perf_counter() - start)
                if remaining <= 0:
                    continue
            try:
                res = solve_centralized_gd(
                    sys_params, bc, epsilon, seed=base_seed + i,
                    tau_init_scale=tau_init_scale, rel_step=rel_step,
                    max_runtime_s=remaining,
                    restarts=max_restarts_per_island != 1,
                    max_restarts=max_restarts_per_island,
                    initial_tau=_island_init_tau(i),
                    random_restart_scale=random_restart_scale,
                    keep_outs=keep_outs)
            except Exception:
                continue
            island_costs[i] = res['cost']
            island_results[i] = res

    island_taus = [(r['tau'] if r is not None else np.zeros((sys_params.N, 3)))
                   for r in island_results]
    winner_id, winner_cost, _ = _max_consensus(graph, island_costs, island_taus)
    if island_results[winner_id] is None:
        raise RuntimeError("decentralized GD: no island produced a solution")

    best = island_results[winner_id]
    runtime = time.perf_counter() - start
    out = dict(best)
    out['method'] = 'decentralized_gd'
    out['runtime'] = runtime
    out['gd'] = dict(best['gd'])
    out['gd']['per_island_costs'] = [float(c) for c in island_costs]
    out['gd']['winner_island'] = int(winner_id)
    out['gd']['num_agents'] = num_agents
    return out
