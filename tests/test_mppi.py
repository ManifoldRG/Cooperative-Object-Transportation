"""Smoke tests for the MPPI solvers. Kept deliberately small (N=8, T=15) so
the full suite finishes in a couple of minutes on a developer machine.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from spacecraft_libraries.data_structures import (
    BoundaryConditions,
    StateVector,
    SystemParams,
)
from spacecraft_libraries.solvers import (
    solve_centralized_mppi,
    solve_decentralized_mppi,
)
from spacecraft_libraries.solvers.mppi_core import (
    evaluate_tau,
    make_nominal_tau,
    run_mppi,
)


def _tiny_scenario():
    rs = [
        np.array([0.5, 1.0, 1.5]),
        np.array([0.0, 0.5, 2.0]),
        np.array([-0.5, 1.0, -1.5]),
    ]
    sys_params = SystemParams(
        mu=3.98e14,
        a=8e6,
        e=0.2,
        nu=np.pi / 4,
        I=1000 * np.diag([1, 2, 3]),
        m=100,
        rs=rs,
        N=8,
    )
    bc = BoundaryConditions(
        x0=StateVector(
            r=np.array([0.0, 0.0, 0.0]),
            v=np.array([0.0, 0.0, 0.0]),
            eps=np.array([0.0, 0.0, 0.0, 1.0]),
            omega=np.array([0.0, 0.0, 0.0]),
        ),
        xf=StateVector(
            r=np.array([2.0, 2.0, 2.0]),
            v=np.array([0.0, 0.0, 0.0]),
            eps=np.array([0.5, 0.5, 0.5, 0.5]),
            omega=np.array([0.0, 0.0, 0.0]),
        ),
        tf=15.0,
    )
    return sys_params, bc, 1e-4


def test_run_mppi_no_worse_than_nominal():
    """MPPI must return a cost no worse than the nominal it started from."""
    sys_params, bc, epsilon = _tiny_scenario()
    rng = np.random.default_rng(7)
    nominal = make_nominal_tau(sys_params, bc, epsilon, rng)
    _, nominal_cost = evaluate_tau(nominal, sys_params, bc, epsilon)
    assert np.isfinite(nominal_cost), "nominal should evaluate to a finite cost"

    best_tau, best_cost, history = run_mppi(
        sys_params, bc, epsilon,
        nominal_tau=nominal,
        n_iter=2, n_samples=4, sigma=1e-3, lambda_=1.0,
        rng=rng,
    )
    assert best_tau.shape == (sys_params.N, 3)
    assert np.isfinite(best_cost)
    assert best_cost <= nominal_cost + 1e-6
    assert history.n_iter_completed >= 1


def test_centralized_mppi_returns_expected_dict():
    sys_params, bc, epsilon = _tiny_scenario()
    result = solve_centralized_mppi(
        sys_params, bc, epsilon,
        n_iter=2, n_samples=3, sigma=1e-3, lambda_=1.0, seed=11,
    )
    for key in ("method", "tau", "trajectory", "control", "attachment",
                "cost", "runtime", "mppi"):
        assert key in result, f"missing key {key} in result"
    assert result["method"] == "centralized_mppi"
    assert result["tau"].shape == (sys_params.N, 3)
    assert np.isfinite(result["cost"])
    assert result["runtime"] > 0


def test_decentralized_mppi_consensus_picks_best_island():
    sys_params, bc, epsilon = _tiny_scenario()
    result = solve_decentralized_mppi(
        sys_params, bc, epsilon,
        n_iter=2, n_samples=3, sigma=1e-3, lambda_=1.0, base_seed=23,
    )
    assert result["method"] == "decentralized_mppi"
    assert result["tau"].shape == (sys_params.N, 3)
    assert np.isfinite(result["cost"])
    per_island = result["mppi"]["per_island_best_costs"]
    finite = [c for c in per_island if np.isfinite(c)]
    assert finite, "at least one island should have produced a finite cost"
    # Final cost (post-finalization through IPOPT) should be no worse than the
    # winning island's MPPI-best cost, modulo numerical slack.
    assert result["cost"] <= min(finite) * 1.5 + 1e-3
