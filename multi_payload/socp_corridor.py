"""Exact inner SOCP with a translational ball corridor (modified copy of
spacecraft_libraries/solvers/socp_inner.py).

Adds, for k = 1..N-1 (the endpoints are pinned),
    ||r_k - rbar_k|| <= eps_k + s_k,   s_k >= 0,
with the penalty rho * sum_k s_k in the objective (elastic corridor; rho=None
makes it a hard corridor with s = 0). Optional position waypoints r_k = w.

Gradient: unchanged. The corridor and waypoint rows do not depend on tau, so
by the envelope theorem they contribute no explicit term to dJ/dtau; their
effect reaches the gradient through the duals of the torque and v-dynamics
rows, which envelope_grad_socp already uses. With slack the descended value is
fuel + l2 term + rho * sum s; sol["fuel"] reports sum ||U|| alone.
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from spacecraft_libraries.solvers.socp_inner import (  # noqa: F401 (re-exports)
    FUEL_L2_REG, attitude_rollout, envelope_grad_socp)
from spacecraft_libraries.new_opts import th_psi_matrix


def solve_inner_socp(sys_params, bc, tau, corridor=None, waypoints=None,
                     solver="CLARABEL", **solver_kwargs):
    """Exact inner SOCP for a fixed tau, with optional corridor.

    corridor: dict(rbar (N+1,3), eps (N+1,), rho float|None).
    waypoints: {k: r (3,)}.
    Returns the socp_inner dict plus fuel, slack (N+1,).
    """
    t0 = time.perf_counter()
    N = sys_params.N
    A = len(sys_params.rs)
    dt = bc.tf / N
    tau = np.asarray(tau, float).reshape(N, 3)
    Rs, _ = attitude_rollout(sys_params, bc, tau)
    rs_body = [np.asarray(r, float) for r in sys_params.rs]
    cosnu = float(np.cos(sys_params.nu))

    U = cp.Variable((A * N, 3))
    r = cp.Variable((N + 1, 3))
    v = cp.Variable((N + 1, 3))
    t = cp.Variable(A * N)
    cons = [r[0] == bc.x0.r, v[0] == bc.x0.v,
            r[N] == bc.xf.r, v[N] == bc.xf.v]
    dyn_cons = []
    torque_cons = []
    for k in range(N):
        Uk = U[k * A:(k + 1) * A, :]
        Psi = np.asarray(th_psi_matrix(sys_params.mu, sys_params.a,
                                       sys_params.e, k * dt))[3:6, :]
        thrust = cp.sum(Uk, axis=0)
        cons.append(r[k + 1] == r[k] + dt * v[k])
        cd = (v[k + 1] == v[k] + dt * (Psi[:, :3] @ r[k] + Psi[:, 3:] @ v[k]
                                       + (Rs[k] @ thrust) / sys_params.m))
        cons.append(cd)
        dyn_cons.append(cd)
        torque = sum(cp.hstack([
            rs_body[i][1] * Uk[i, 2] - rs_body[i][2] * Uk[i, 1],
            rs_body[i][2] * Uk[i, 0] - rs_body[i][0] * Uk[i, 2],
            rs_body[i][0] * Uk[i, 1] - rs_body[i][1] * Uk[i, 0]])
            for i in range(A))
        ct = (torque == tau[k])
        cons.append(ct)
        torque_cons.append(ct)
        for i in range(A):
            cons.append(cp.norm(Uk[i], 2) <= t[k * A + i])
            rho_hat = rs_body[i] / np.linalg.norm(rs_body[i])
            cons.append(cosnu * cp.norm(Uk[i], 2) <= -(rho_hat @ Uk[i]))

    objective = cp.sum(t) + FUEL_L2_REG * cp.sum_squares(U)
    s = None
    if corridor is not None:
        rbar = np.asarray(corridor["rbar"], float)
        eps = np.asarray(corridor["eps"], float)
        rho = corridor.get("rho")
        ks = range(1, N)
        if rho is None:
            for k in ks:
                cons.append(cp.norm(r[k] - rbar[k], 2) <= eps[k])
        else:
            s = cp.Variable(N + 1, nonneg=True)
            for k in ks:
                cons.append(cp.norm(r[k] - rbar[k], 2) <= eps[k] + s[k])
            objective = objective + rho * cp.sum(s[1:N])
    for k, w in (waypoints or {}).items():
        cons.append(r[k] == np.asarray(w, float))

    prob = cp.Problem(cp.Minimize(objective), cons)
    try:
        prob.solve(solver=solver, verbose=False, **solver_kwargs)
    except Exception:
        return dict(ok=False, wall=time.perf_counter() - t0)
    if prob.status not in ("optimal",):
        return dict(ok=False, status=prob.status, wall=time.perf_counter() - t0)

    U_v = np.transpose(U.value.reshape(N, A, 3), (1, 0, 2))
    thrust_v = U_v.sum(axis=0)
    slack = np.zeros(N + 1) if s is None else np.maximum(np.asarray(s.value), 0.0)
    slack[0] = slack[N] = 0.0
    return dict(
        ok=True, J=float(prob.value),
        fuel=float(np.linalg.norm(U_v, axis=2).sum()), slack=slack,
        U=U_v, r=np.asarray(r.value), v=np.asarray(v.value),
        lam_torque=np.asarray([c.dual_value for c in torque_cons]),
        lam_dyn=np.asarray([c.dual_value for c in dyn_cons]),
        Rs=Rs, thrust=thrust_v, status=prob.status,
        wall=time.perf_counter() - t0)
