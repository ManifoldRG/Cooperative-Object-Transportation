"""Step 1: center-of-mass force references.

The translational dynamics depend on the agents only through the net force
applied at the center of mass. With F_k = R_k sum_i U_ik (Hill frame) the
paper's update
    r_{k+1} = r_k + dt v_k
    v_{k+1} = v_k + dt (Psi_vel(t_k) [r_k; v_k] + F_k / m)
is linear in (r, v, F), so the optimal reference
    min sum_k c(F_k)  s.t. dynamics, r/v pins, waypoints r_k = w
is an SOCP, with c = ||.|| (objective "fuel") or ||.||^2 ("energy"). It
does not depend on the allocation or on the attitude. Lower bounds on the
payload's cost with A agents, from its value without waypoints:
    fuel:   sum_i ||U_ik|| >= ||F_k||            ->  J >= ell
    energy: sum_i ||U_ik||^2 >= ||F_k||^2 / A    ->  J >= ell / A  (Cauchy-Schwarz)
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np

from spacecraft_libraries.new_opts import th_psi_matrix


def psi_list(sc):
    """Velocity rows of the Tschauner-Hempel matrix at every step (shared)."""
    return [np.asarray(th_psi_matrix(sc.mu, sc.a, sc.e, k * sc.dt))[3:6, :]
            for k in range(sc.N)]


def cost_expr(rows, objective):
    """sum of row norms (fuel) or of squared row norms (energy), cvxpy."""
    if objective == "energy":
        return cp.sum_squares(rows)
    if objective == "fuel":
        return cp.sum(cp.norm(rows, 2, axis=1))
    raise ValueError(f"unknown objective {objective!r}")


def cost_value(U, objective):
    """Same cost for a numeric array whose last axis is the thrust vector."""
    n = np.linalg.norm(np.asarray(U, float), axis=-1)
    return float(np.sum(n ** 2) if objective == "energy" else np.sum(n))


def solve_reference(sc, j, waypoints=None, psis=None, solver="CLARABEL", objective="fuel"):
    """Optimal CoM-force trajectory of payload j.

    waypoints: {k: r (3,)} position equality constraints.
    Returns dict(ok, r (N+1,3), v (N+1,3), F (N,3), cost, fuel, waypoints).
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
    prob = cp.Problem(cp.Minimize(cost_expr(F, objective)), cons)
    try:
        prob.solve(solver=solver)
    except cp.error.SolverError:
        return dict(ok=False, waypoints=waypoints)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return dict(ok=False, waypoints=waypoints, status=prob.status)
    Fv = np.asarray(F.value)
    return dict(ok=True, r=np.asarray(r.value), v=np.asarray(v.value), F=Fv,
                cost=float(prob.value), fuel=cost_value(Fv, "fuel"), waypoints=waypoints,
                objective=objective)


def solve_all(sc, waypoints=None, psis=None, objective="fuel"):
    psis = psi_list(sc) if psis is None else psis
    waypoints = waypoints or [dict() for _ in range(sc.M)]
    return [solve_reference(sc, j, waypoints[j], psis, objective=objective)
            for j in range(sc.M)]
