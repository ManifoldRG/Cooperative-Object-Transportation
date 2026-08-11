from __future__ import annotations

import time

import numpy as np

from .. import new_opts
from ..data_structures import BoundaryConditions, SystemParams
from .mppi_core import make_nominal_tau, run_mppi


def solve_centralized_mppi(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    n_iter: int = 5,
    n_samples: int = 10,
    sigma: float = 1e-1,
    lambda_: float = 1.0,
    seed: int = 42,
    max_runtime_s: float | None = None,
    use_oracle: bool = True,
):
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    oracle = None
    if use_oracle:
        from .parametric_oracle import ScenarioOracle
        oracle = ScenarioOracle(sys_params, bc, epsilon)

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
        deadline_s=max_runtime_s,
        start_time=start,
        oracle=oracle,
    )

    traj, ctrl, q, cost = new_opts.opt_given_tau_ipopt_new(
        best_tau, sys_params.N, epsilon, sys_params, bc, num_iter=1000 #changed to 1000
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
