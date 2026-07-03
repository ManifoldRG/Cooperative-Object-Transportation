"""Model Predictive Path Integral (MPPI) loop on the body-frame torque trajectory.

Each sample is a Gaussian perturbation of the nominal tau trajectory. The cost
of a sample is obtained by the same path the GA fitness uses: project the
perturbed tau through `tau_proj_nonlin_new` onto the rotational-feasibility
manifold, then call `opt_given_tau_ipopt_new` (the inner thrust-allocation
IPOPT) to obtain the realized fuel cost sum_k ||U_k||^2. Failed samples are
assigned +inf cost and contribute zero weight to the path-integral update.

This module is shared by `solve_centralized_mppi` and `solve_decentralized_mppi`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Optional

import casadi as ca
import numpy as np

from .. import genetic_code, new_opts
from ..data_structures import BoundaryConditions, SystemParams


@dataclass
class MPPIHistory:
    n_iter_completed: int = 0
    n_samples_per_iter: int = 0
    best_cost_per_iter: list = field(default_factory=list)
    failed_samples: int = 0
    total_samples: int = 0

    @property
    def failed_sample_fraction(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.failed_samples / self.total_samples

    def as_dict(self) -> dict:
        return {
            "n_iter_completed": self.n_iter_completed,
            "n_samples_per_iter": self.n_samples_per_iter,
            "best_cost_per_iter": list(self.best_cost_per_iter),
            "failed_sample_fraction": self.failed_sample_fraction,
        }


def make_nominal_tau(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a nominal tau via the same multiple-shooting routine the GA uses.

    `multiple_shooting_optimization_new` seeds its IPOPT solve with a small
    uniform random tau guess drawn from numpy's global RNG (genetic_code.py:225).
    We temporarily set the global seed from `rng` so different islands get
    different nominals.
    """
    N = sys_params.N
    dt = bc.tf / N
    I_casadi = ca.DM(sys_params.I)
    dt_casadi = ca.DM(dt)
    epsilon_casadi = ca.DM(epsilon)

    seed = int(rng.integers(0, 2**31 - 1))
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        tau_opt, _ = genetic_code.multiple_shooting_optimization_new(
            bc, N, dt_casadi, I_casadi, epsilon_casadi
        )
    finally:
        np.random.set_state(state)

    return np.asarray(tau_opt, dtype=float).reshape(N, 3)


def evaluate_tau(
    tau: np.ndarray,
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
) -> tuple[np.ndarray, float]:
    """Project tau onto the rotational manifold, then solve the inner thrust
    allocation IPOPT to score it. Returns (projected_tau, cost). Cost is
    +inf on any failure.
    """
    N = sys_params.N
    try:
        tau_proj = new_opts.tau_proj_nonlin_new(tau, N, epsilon, sys_params, bc)[0]
        tau_proj = np.asarray(tau_proj, dtype=float).reshape(N, 3)
        _, _, _, cost = new_opts.opt_given_tau_ipopt_new(
            tau_proj, N, epsilon, sys_params, bc, num_iter=1000 #changed to 1000
        )
        cost = float(cost)
        if not np.isfinite(cost) or cost <= 0:
            return tau_proj, float("inf")
        return tau_proj, cost
    except Exception:
        return np.asarray(tau, dtype=float).reshape(N, 3), float("inf")


def run_mppi(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    nominal_tau: np.ndarray,
    *,
    n_iter: int,
    n_samples: int,
    sigma: float,
    lambda_: float,
    rng: np.random.Generator,
    deadline_s: Optional[float] = None,
    start_time: Optional[float] = None,
) -> tuple[np.ndarray, float, MPPIHistory]:
    """Run the MPPI loop and return (best_tau, best_cost, history).

    `best_tau` is the lowest-cost projected tau ever sampled (NOT just the final
    nominal), so we never regress below the best seen sample.
    """
    N = sys_params.N
    nominal = np.asarray(nominal_tau, dtype=float).reshape(N, 3).copy()

    history = MPPIHistory(n_samples_per_iter=n_samples)
    best_tau = nominal.copy()
    best_cost = float("inf")

    # Score the nominal first so we never return worse than starting point.
    nominal_proj, nominal_cost = evaluate_tau(nominal, sys_params, bc, epsilon)
    history.total_samples += 1
    if not np.isfinite(nominal_cost):
        history.failed_samples += 1
    else:
        best_tau = nominal_proj
        best_cost = nominal_cost
        nominal = nominal_proj  # start MPPI from the projected feasible nominal

    if start_time is None:
        start_time = time.perf_counter()

    for it in range(n_iter):
        if deadline_s is not None and (time.perf_counter() - start_time) >= deadline_s:
            break

        eps_samples = rng.normal(0.0, sigma, size=(n_samples, N, 3))
        costs = np.full(n_samples, np.inf)
        projected = [None] * n_samples

        for k in range(n_samples):
            if deadline_s is not None and (time.perf_counter() - start_time) >= deadline_s:
                break
            tau_k = nominal + eps_samples[k]
            tau_k_proj, J_k = evaluate_tau(tau_k, sys_params, bc, epsilon)
            history.total_samples += 1
            if not np.isfinite(J_k):
                history.failed_samples += 1
            costs[k] = J_k
            projected[k] = tau_k_proj
            if J_k < best_cost:
                best_cost = J_k
                best_tau = tau_k_proj

        finite_mask = np.isfinite(costs)
        history.n_iter_completed = it + 1
        history.best_cost_per_iter.append(best_cost)

        if not finite_mask.any():
            continue

        J_min = costs[finite_mask].min()
        # Numerically stable softmin weights; zero weight on +inf samples.
        weights = np.zeros(n_samples)
        weights[finite_mask] = np.exp(-(costs[finite_mask] - J_min) / max(lambda_, 1e-12))
        w_sum = weights.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            continue

        # Weighted update on the perturbations (path-integral form).
        update = np.einsum("k,kij->ij", weights / w_sum, eps_samples)
        nominal = nominal + update

    return best_tau, best_cost, history
