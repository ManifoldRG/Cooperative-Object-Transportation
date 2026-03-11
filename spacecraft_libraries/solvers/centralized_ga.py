from __future__ import annotations

import time

import numpy as np

from spacecraft_libraries.data_structures import BoundaryConditions, SystemParams
from spacecraft_libraries.genetic_code import pop_gen_new
from spacecraft_libraries.new_opts import opt_given_tau_ipopt_new, tau_proj_nonlin_new
from spacecraft_libraries.solvers.centralized_nlp import SolverRun


def solve_centralized_ga(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float = 1e-5,
    pop_size: int = 8,
) -> SolverRun:
    t0 = time.perf_counter()
    population = pop_gen_new(bc, sys_params, sys_params.N, epsilon, pop_size=pop_size)
    best_cost = np.inf
    best = None
    for tau_guess in population:
        tau_proj, _ = tau_proj_nonlin_new(tau_guess, sys_params.N, epsilon, sys_params, bc)
        traj, ctrl, q_hist, cost = opt_given_tau_ipopt_new(tau_proj, sys_params.N, epsilon, sys_params, bc)
        if cost < best_cost:
            best_cost = float(cost)
            best = (traj, ctrl, q_hist)
    runtime_s = time.perf_counter() - t0
    if best is None:
        raise RuntimeError("Centralized GA failed to produce a feasible candidate")
    traj, ctrl, q_hist = best
    return SolverRun(
        method="centralized_ga",
        trajectory=traj.as_array(),
        control=ctrl.U,
        q_history=np.asarray(q_hist),
        cost=best_cost,
        runtime_s=runtime_s,
    )
