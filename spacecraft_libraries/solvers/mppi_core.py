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
    sigma_abs: float | None = None  # effective absolute sigma actually sampled with

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
            "sigma_abs": self.sigma_abs,
        }


def make_nominal_tau(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    rng: np.random.Generator,
    tau_init_scale: float = 1e-1,
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
            bc, N, dt_casadi, I_casadi, epsilon_casadi, tau_init_scale=tau_init_scale
        )
    finally:
        np.random.set_state(state)

    return np.asarray(tau_opt, dtype=float).reshape(N, 3)


def evaluate_tau(
    tau: np.ndarray,
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    oracle=None,
) -> tuple[np.ndarray, float]:
    """Project tau onto the rotational manifold, then solve the inner thrust
    allocation IPOPT to score it. Returns (projected_tau, cost). Cost is
    +inf on any failure.

    With `oracle` (a parametric_oracle.ScenarioOracle) the build-once
    parametric solvers are used — same problems, ~5-7x faster per call.
    """
    if oracle is not None:
        return oracle.evaluate(tau)
    N = sys_params.N
    try:
        tau_proj = new_opts.tau_proj_nonlin_new(tau, N, epsilon, sys_params, bc, allow_raising_error=True )[0]
        if tau_proj is None :
            return np.asarray(tau, dtype=float).reshape(N, 3), float("inf")
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


def _sample_smooth_noise(
    rng: np.random.Generator,
    n_samples: int,
    N: int,
    sigma: float,
    knots: int,
) -> np.ndarray:
    """Temporally correlated perturbations: Gaussian values at `knots` evenly
    spaced timesteps per axis, linearly interpolated across the N-step horizon.

    Why: white per-timestep noise makes every sample a ~(N*3)-dimensional
    isotropic kick, whose probability of improving a near-optimal nominal
    collapses with dimension (observed: hundreds of consecutive rejected
    samples). Smooth noise restricts samples to a ~(knots*3)-dimensional
    subspace of slowly varying torque adjustments — the moves the dynamics
    actually respond to coherently. Marginal std at the knots equals sigma;
    between knots it is slightly lower (linear interpolation).
    """
    t_knots = np.linspace(0.0, N - 1.0, knots)
    t = np.arange(N, dtype=float)
    vals = rng.normal(0.0, sigma, size=(n_samples, knots, 3))
    out = np.empty((n_samples, N, 3))
    for k in range(n_samples):
        for ax in range(3):
            out[k, :, ax] = np.interp(t, t_knots, vals[k, :, ax])
    return out


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
    relative_sigma: bool = False,
    sigma_floor: float = 1e-6,
    noise_mode: str = "white",
    noise_knots: int = 8,
    oracle=None,
    selection: str = "softmin",
    step_size: float = 1.0,
) -> tuple[np.ndarray, float, MPPIHistory]:
    """Run the MPPI loop and return (best_tau, best_cost, history).

    `best_tau` is the lowest-cost projected tau ever sampled (NOT just the final
    nominal), so we never regress below the best seen sample.

    With `relative_sigma=True`, `sigma` is interpreted as a fraction of the
    warm-start nominal's RMS torque: sigma_abs = sigma * max(RMS(nominal),
    sigma_floor). The right absolute perturbation scale is scenario-dependent
    (torque magnitudes vary orders of magnitude across masses/inertias/
    horizons), so a relative sigma transfers across scenarios where an
    absolute one cannot. The effective value is recorded in history.sigma_abs.
    """
    N = sys_params.N
    nominal = np.asarray(nominal_tau, dtype=float).reshape(N, 3).copy()

    history = MPPIHistory(n_samples_per_iter=n_samples)
    best_tau = nominal.copy()
    best_cost = float("inf")

    # Score the nominal first so we never return worse than starting point.
    nominal_proj, nominal_cost = evaluate_tau(nominal, sys_params, bc, epsilon, oracle=oracle)
    history.total_samples += 1
    if not np.isfinite(nominal_cost):
        history.failed_samples += 1
    else:
        best_tau = nominal_proj
        best_cost = nominal_cost
        nominal = nominal_proj  # start MPPI from the projected feasible nominal

    if start_time is None:
        start_time = time.perf_counter()

    # Resolve the effective sampling std AFTER the nominal projection, so a
    # relative sigma is anchored to the feasible warm-start torque scale.
    sigma_eff = sigma
    if relative_sigma:
        rms = float(np.sqrt(np.mean(np.square(nominal))))
        sigma_eff = sigma * max(rms, sigma_floor)
    history.sigma_abs = sigma_eff

    for it in range(n_iter):
        if deadline_s is not None and (time.perf_counter() - start_time) >= deadline_s:
            break

        if noise_mode == "smooth":
            eps_samples = _sample_smooth_noise(rng, n_samples, N, sigma_eff,
                                               min(noise_knots, N))
        else:
            eps_samples = rng.normal(0.0, sigma_eff, size=(n_samples, N, 3))
        costs = np.full(n_samples, np.inf)
        projected = [None] * n_samples

        for k in range(n_samples):
            if deadline_s is not None and (time.perf_counter() - start_time) >= deadline_s:
                break
            tau_k = nominal + eps_samples[k]
            tau_k_proj, J_k = evaluate_tau(tau_k, sys_params, bc, epsilon, oracle=oracle)
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

        if selection == "greedy":
            # Greedy Sampler: move the nominal toward the batch-best
            # PROJECTED sample — the lambda->0 limit of the softmin, made
            # exact (no exponential underflow). step_size=1 jumps fully onto
            # the best sample (pure hill-climbing with uphill drift as free
            # exploration); step_size<1 relaxes toward it (damped walk). The
            # move happens even if the batch best is worse than the current
            # nominal; best-so-far tracking keeps the returned solution
            # monotone.
            k_best = int(np.argmin(np.where(finite_mask, costs, np.inf)))
            if projected[k_best] is not None:
                target = np.asarray(projected[k_best])
                nominal = nominal + step_size * (target - nominal)
            continue

        J_min = costs[finite_mask].min()
        # Numerically stable softmin weights; zero weight on +inf samples.
        weights = np.zeros(n_samples)
        weights[finite_mask] = np.exp(-(costs[finite_mask] - J_min) / max(lambda_, 1e-12))
        w_sum = weights.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            continue

        # Weighted update over the PROJECTED samples. The costs are measured
        # at the projections, so the center update must live in the same set:
        # averaging RAW perturbations would move the sampling center off the
        # feasible manifold and decouple it from the costs that justified the
        # move (center/score mismatch, worst at large steps). At one-hot
        # weights (small lambda) this is exactly a greedy jump to the
        # projected winner.
        w = weights / w_sum
        target = np.zeros_like(nominal)
        for k in range(n_samples):
            if w[k] > 0 and projected[k] is not None:
                target = target + w[k] * np.asarray(projected[k])
        nominal = nominal + step_size * (target - nominal)

    return best_tau, best_cost, history
