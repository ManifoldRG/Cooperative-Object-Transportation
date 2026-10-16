"""Step 1: center-of-mass force references.

The translational dynamics depend on the agents only through the net force
applied at the center of mass. With F_k = R_k sum_i U_ik (Hill frame) the
paper's update
    r_{k+1} = r_k + dt v_k
    v_{k+1} = v_k + dt (Psi_vel(t_k) [r_k; v_k] + F_k / m)
is linear in (r, v, F), so the fuel-optimal reference
    min sum_k ||F_k||  s.t. dynamics, r/v pins, waypoints r_k = w
is an SOCP. It does not depend on the allocation or on the attitude. Since
sum_i ||U_ik|| >= ||sum_i U_ik|| = ||F_k||, its value without waypoints is a
lower bound on any payload's fuel.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from spacecraft_libraries.new_opts import th_psi_matrix


def psi_list(sc):
    """Velocity rows of the Tschauner-Hempel matrix at every step (shared)."""
    return [np.asarray(th_psi_matrix(sc.mu, sc.a, sc.e, k * sc.dt))[3:6, :]
            for k in range(sc.N)]


def solve_reference(sc, j, waypoints=None, psis=None, solver="CLARABEL"):
    """Fuel-optimal CoM-force trajectory of payload j.

    waypoints: {k: r (3,)} position equality constraints.
    Returns dict(ok, r (N+1,3), v (N+1,3), F (N,3), fuel, waypoints).
    """
    N, dt = sc.N, sc.dt
    P = sc.payloads[j]
    psis = psi_list(sc) if psis is None else psis
    waypoints = dict(waypoints or {})
    r = cp.Variable((N + 1, 3))
    v = cp.Variable((N + 1, 3))
    F = cp.Variable((N, 3))
    cons = [r[0] == P.x0.r, v[0] == P.x0.v, r[N] == P.xf.r, v[N] == P.xf.v]
    for k in range(N):
        Psi = psis[k]
        cons += [r[k + 1] == r[k] + dt * v[k],
                 v[k + 1] == v[k] + dt * (Psi[:, :3] @ r[k] + Psi[:, 3:] @ v[k]
                                          + F[k] / P.m)]
    for k, w in waypoints.items():
        cons.append(r[k] == np.asarray(w, float))
    prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
    try:
        prob.solve(solver=solver)
    except cp.error.SolverError:
        return dict(ok=False, waypoints=waypoints)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return dict(ok=False, waypoints=waypoints, status=prob.status)
    return dict(ok=True, r=np.asarray(r.value), v=np.asarray(v.value),
                F=np.asarray(F.value), fuel=float(prob.value), waypoints=waypoints)


def solve_all(sc, waypoints=None, psis=None):
    psis = psi_list(sc) if psis is None else psis
    waypoints = waypoints or [dict() for _ in range(sc.M)]
    return [solve_reference(sc, j, waypoints[j], psis) for j in range(sc.M)]
