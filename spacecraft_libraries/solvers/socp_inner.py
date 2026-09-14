"""Exact-cone inner problem as an SOCP (cvxpy/CLARABEL) + envelope gradient.

Given a torque trajectory tau, the thrust-allocation inner problem with the
attitude trajectory rolled out numerically is an exact SOCP:

    min  sum_ik t_ik + l2_reg * ||U||^2
    s.t. r/v pins and discrete TH dynamics (R_k(tau) numeric),
         torque equality  sum_i rho_i x U_ik = tau_k,
         ||U_ik|| <= t_ik                      (fuel epigraph, EXACT),
         cos(nu) ||U_ik|| <= -rho_hat_i . U_ik (thrust cone, EXACT).

No smoothing anywhere: solutions are exact-cone feasible, unlike the IPOPT
formulations (recentered smooth_norm relaxes the cone by ~epsilon, which
fuel-optimal micro-thrusters exploit for ~2-3% fuel).

The envelope gradient dJ/dtau of the optimal value has two channels:
  1. the torque equalities (tau on the RHS): -dual of those rows;
  2. the attitude channel: tau -> omega -> R_k enters the dynamics rows;
     recovered from the dynamics duals via a NumPy adjoint (backward) pass
     over the recursion omega' = omega + dt I^-1 (tau - omega x I omega),
     R' = R exp(dt omega), using the SO(3) right Jacobian.
Validated against central finite differences (see scripts/validate_socp_grad).
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from ..new_opts import so3_exp, state_attitude_to_phi, th_psi_matrix

FUEL_L2_REG = 1e-3


def _skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _right_jacobian(phi):
    """SO(3) right Jacobian Jr(phi): exp(phi + d) ~ exp(phi) exp(Jr(phi) d)."""
    th = np.linalg.norm(phi)
    P = _skew(phi)
    if th < 1e-9:
        return np.eye(3) - 0.5 * P + P @ P / 6.0
    return (np.eye(3)
            - (1.0 - np.cos(th)) / th**2 * P
            + (th - np.sin(th)) / th**3 * (P @ P))


def attitude_rollout(sys_params, bc, tau):
    """Numeric rollout: returns Rs (N+1 rotation matrices), omes ((N+1,3))."""
    N = sys_params.N
    dt = bc.tf / N
    I = np.asarray(sys_params.I, float)
    I_inv = np.linalg.inv(I)
    Rs = [so3_exp(state_attitude_to_phi(bc.x0))]
    omes = [np.asarray(bc.x0.omega, float).copy()]
    for k in range(N):
        Rs.append(Rs[k] @ so3_exp(dt * omes[k]))
        omes.append(omes[k] + dt * (I_inv @ (tau[k] - np.cross(omes[k], I @ omes[k]))))
    return Rs, np.asarray(omes)


def solve_inner_socp(sys_params, bc, tau, solver="CLARABEL", **solver_kwargs):
    """Solve the exact inner SOCP for a fixed tau.

    Returns dict(ok, J, U (A,N,3), r ((N+1,3)), v ((N+1,3)),
                 lam_torque (N,3), lam_dyn (N,3), Rs, thrust (N,3), wall).
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

    prob = cp.Problem(cp.Minimize(cp.sum(t) + FUEL_L2_REG * cp.sum_squares(U)),
                      cons)
    try:
        prob.solve(solver=solver, verbose=False, **solver_kwargs)
    except Exception:
        return dict(ok=False, wall=time.perf_counter() - t0)
    if prob.status not in ("optimal",):
        return dict(ok=False, status=prob.status, wall=time.perf_counter() - t0)

    U_v = np.transpose(U.value.reshape(N, A, 3), (1, 0, 2))
    thrust_v = U_v.sum(axis=0)
    return dict(
        ok=True, J=float(prob.value),
        U=U_v, r=np.asarray(r.value), v=np.asarray(v.value),
        lam_torque=np.asarray([c.dual_value for c in torque_cons]),
        lam_dyn=np.asarray([c.dual_value for c in dyn_cons]),
        Rs=Rs, thrust=thrust_v, status=prob.status,
        wall=time.perf_counter() - t0)


def envelope_grad_socp(sys_params, bc, tau, sol):
    """dJ/dtau of the exact inner optimal value, from the SOCP duals plus a
    NumPy adjoint over the attitude recursion. Returns (N,3)."""
    N = sys_params.N
    dt = bc.tf / N
    I = np.asarray(sys_params.I, float)
    I_inv = np.linalg.inv(I)
    tau = np.asarray(tau, float).reshape(N, 3)
    Rs, omes = attitude_rollout(sys_params, bc, tau)
    thrust = sol["thrust"]           # (N,3), body frame
    lam_dyn = sol["lam_dyn"]         # (N,3) duals of v-dynamics rows
    lam_tq = sol["lam_torque"]       # (N,3) duals of torque rows

    # channel 1: L = f + y^T(torque - tau) => dJ/dtau_k = -y_k
    grad = -lam_tq.copy()

    # channel 2: G_k = dL/dR_k = -dt/m * lam_dyn_k thrust_k^T  (k = 0..N-1)
    G = [-(dt / sys_params.m) * np.outer(lam_dyn[k], thrust[k])
         for k in range(N)]
    E = [so3_exp(dt * omes[k]) for k in range(N)]

    # pass 1 - rotation cotangents (R_{k+1} = R_k E_k; R_0 constant):
    #   Rbar_{N-1} = G_{N-1};  Rbar_k = G_k + Rbar_{k+1} E_k^T
    #   Ebar_k = R_k^T Rbar_{k+1}   (E_{N-1} only feeds unused R_N -> 0)
    Rbar = [None] * N
    Rbar[N - 1] = G[N - 1]
    for k in range(N - 2, -1, -1):
        Rbar[k] = G[k] + Rbar[k + 1] @ E[k].T
    Ebar = [np.zeros((3, 3))] * N
    for k in range(N - 1):
        Ebar[k] = Rs[k].T @ Rbar[k + 1]

    # pass 2 - omega cotangents:
    #   obar_k = obar_k^(E) + F_k^T obar_{k+1},  obar_N = 0
    #   obar_k^(E) = dt Jr(dt w_k)^T vee(A - A^T), A = E_k^T Ebar_k
    #   F_k = I - dt I^-1 ([w_k]x I - [I w_k]x)
    # tau channel: grad_k += (dt I^-1)^T obar_{k+1}
    obar_next = np.zeros(3)   # obar_{N}
    for k in range(N - 1, -1, -1):
        if k <= N - 1:
            grad[k] += (dt * I_inv).T @ obar_next if k < N else 0.0
        A = E[k].T @ Ebar[k]
        obar_E = dt * _right_jacobian(dt * omes[k]).T @ _vee(A - A.T)
        w = omes[k]
        F = np.eye(3) - dt * I_inv @ (_skew(w) @ I - _skew(I @ w))
        obar_next = obar_E + F.T @ obar_next
    return grad
