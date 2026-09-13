"""Joint NLP with the CORRECT reference dynamics (Tschauner-Hempel +
body-frame thrusts + Lie-chart attitude).

Replaces og_opts.full_nlp as the direct-NLP baseline. full_nlp solves
DIFFERENT physics (CW-style A_vel from orbital_params, thrusts applied in the
Hill frame with no R rotation, inertial rotating attachment vectors), so its
costs were never comparable to the GA/GS/GD stack, which all score through
opt_given_tau_ipopt_new's TH + body-frame model. This solver co-optimizes
thrusts AND the attitude trajectory under exactly that model:

  decision:  U (agents x N x 3, BODY frame), r, v, phi, omega ((N+1) x 3)
  dynamics:  r' = r + dt v
             v' = v + dt (Psi_vel [r;v] + (1/m) R(phi_k) sum_i U_i)
             R(phi_{k+1}) = R(phi_k) exp(dt omega_k)   (via so3_log chart)
             omega' = omega + dt I^{-1}(sum_i rho_i x U_i - omega x I omega)
  pins:      full state at k=0 and k=N
  cones:     U_i . rho_i >= cos(nu) ||U_i|| ||rho_i||   (body frame)
  cost:      sum ||U||^2

Attitude states carry the projector's trust-region bounds (|phi_i| <= pi-1e-3,
|omega_i| <= 2(pi-1e-3)/dt) — the same principal-chart confinement, so this
NLP explores the same homotopy class as the bilevel stack (comparable by
construction). Wall-clock limit via IPOPT max_cpu_time.
"""
from __future__ import annotations

import os
import sys
import time

import casadi as ca
import numpy as np

from ..data_structures import BoundaryConditions, SystemParams
from ..new_opts import (
    so3_exp_casadi,
    so3_log_casadi,
    state_attitude_to_phi,
    th_psi_matrix,
    skew_casadi,
    smooth_norm,
)

_SUCCESS = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}


def solve_centralized_nlp_th(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    max_iters: int = 3000,
    max_runtime_s: float | None = None,
    U_guess: np.ndarray | None = None,
):
    start = time.perf_counter()
    N = sys_params.N
    num_agents = len(sys_params.rs)
    dt = bc.tf / N

    phi0 = state_attitude_to_phi(bc.x0)
    phif = state_attitude_to_phi(bc.xf)

    U = ca.SX.sym('U', num_agents * N * 3)
    r = ca.SX.sym('r', (N + 1) * 3)
    v = ca.SX.sym('v', (N + 1) * 3)
    phi = ca.SX.sym('phi', (N + 1) * 3)
    ome = ca.SX.sym('ome', (N + 1) * 3)

    def get_U(i, k):
        idx = (i * N + k) * 3
        return U[idx:idx + 3]

    def blk(x, k):
        return x[k * 3:k * 3 + 3]

    # FUEL objective (was ca.sumsqr(U) energy before 2026-09-07), in epigraph
    # form: minimize sum(t_ik) s.t. smooth_norm(U_ik) <= t_ik, t >= 0. The
    # direct smoothed-norm objective stalls IPOPT near the zero-thrust
    # solutions fuel-optimality pushes toward; the epigraph gives a linear
    # objective + smooth convex constraints and the same optimal value (bias
    # ~N*agents*FUEL_SMOOTH). Slacks stacked LAST: state slice offsets keep.
    # tiny L2 term restores strict convexity of the allocation (see
    # parametric_oracle: pure fuel is degenerate, IPOPT outcome flips on
    # 1e-15 tau perturbations); ~0.04% of the fuel cost
    FUEL_SMOOTH = 1e-4
    FUEL_L2_REG = 1e-3
    T_slack = ca.SX.sym('t_fuel', num_agents * N)
    cost = ca.sum1(T_slack) + FUEL_L2_REG * ca.sumsqr(U)

    def get_t(i, k):
        return T_slack[i * N + k]

    I_mat = ca.DM(np.asarray(sys_params.I))
    I_inv = ca.DM(np.linalg.inv(np.asarray(sys_params.I)))
    rs_body = [np.asarray(sys_params.rs[i], dtype=float) for i in range(num_agents)]

    g = []
    lbg = []
    ubg = []

    # initial pins
    for expr in [blk(r, 0) - bc.x0.r, blk(v, 0) - bc.x0.v,
                 blk(phi, 0) - phi0, blk(ome, 0) - np.asarray(bc.x0.omega, dtype=float)]:
        g.append(expr)
        lbg.extend([0] * 3)
        ubg.extend([0] * 3)

    for k in range(N):
        R_k = so3_exp_casadi(blk(phi, k))

        torque_curr = ca.SX.zeros(3)
        thrust_body = ca.SX.zeros(3)
        for i in range(num_agents):
            rho = ca.DM(rs_body[i])
            U_ik = get_U(i, k)
            torque_curr += skew_casadi(rho) @ U_ik
            thrust_body += U_ik
            # cone (body frame), same smooth form as the inner problem
            # recentered smoothing (see parametric_oracle): U=0 must be
            # feasible or fuel-optimal off-thrusters are forbidden
            # cone axis -rho (thrust inward, exhaust outward), matching the
            # oracle/legacy inner after the 2026-09-11 correction
            g.append(ca.dot(U_ik, -rho)
                     - ca.cos(sys_params.nu)
                     * (smooth_norm(U_ik, epsilon) - epsilon)
                     * float(np.linalg.norm(rs_body[i])))
            lbg.append(0)
            ubg.append(ca.inf)
            # fuel epigraph, recentered: t=0 feasible at U=0
            g.append(get_t(i, k) - (smooth_norm(U_ik, FUEL_SMOOTH) - FUEL_SMOOTH))
            lbg.append(0)
            ubg.append(ca.inf)

        # translational TH dynamics, body->inertial thrust via R(phi_k)
        Psi_k = th_psi_matrix(sys_params.mu, sys_params.a, sys_params.e, k * dt)
        Psi_vel = ca.DM(Psi_k[3:6, :])
        r_k, v_k = blk(r, k), blk(v, k)
        g.append(blk(r, k + 1) - (r_k + dt * v_k))
        g.append(blk(v, k + 1) - (v_k + dt * (Psi_vel @ ca.vertcat(r_k, v_k)
                                              + (1.0 / sys_params.m) * R_k @ thrust_body)))
        lbg.extend([0] * 6)
        ubg.extend([0] * 6)

        # attitude chart dynamics (same as the projector)
        R_next = R_k @ so3_exp_casadi(dt * blk(ome, k))
        g.append(blk(phi, k + 1) - so3_log_casadi(R_next))
        g.append(blk(ome, k + 1) - (blk(ome, k)
                                    + dt * I_inv @ (torque_curr
                                                    - ca.cross(blk(ome, k), I_mat @ blk(ome, k)))))
        lbg.extend([0] * 6)
        ubg.extend([0] * 6)

    # terminal pins (full state)
    for expr in [blk(r, N) - bc.xf.r, blk(v, N) - bc.xf.v,
                 blk(phi, N) - phif, blk(ome, N) - np.asarray(bc.xf.omega, dtype=float)]:
        g.append(expr)
        lbg.extend([0] * 3)
        ubg.extend([0] * 3)

    x = ca.vertcat(U, r, v, phi, ome, T_slack)
    g = ca.vertcat(*g)

    # variable bounds: thrusts free; states free except attitude trust region;
    # fuel slacks t >= 0 at the tail
    phi_bound = np.pi - 1e-3
    omega_bound = 2 * (np.pi - 1e-3) / dt
    lbx = np.concatenate([
        -np.inf * np.ones(num_agents * N * 3),
        -np.inf * np.ones((N + 1) * 6),
        np.tile([-phi_bound] * 3, N + 1),
        np.tile([-omega_bound] * 3, N + 1),
        np.zeros(num_agents * N),
    ])
    ubx = np.concatenate([
        np.inf * np.ones(num_agents * N * 3),
        np.inf * np.ones((N + 1) * 6),
        np.tile([phi_bound] * 3, N + 1),
        np.tile([omega_bound] * 3, N + 1),
        np.inf * np.ones(num_agents * N),
    ])

    opts = {"print_time": False,
            'ipopt': {'max_iter': max_iters, 'print_level': 0, 'sb': 'yes'}}
    if max_runtime_s is not None:
        opts['ipopt']['max_cpu_time'] = float(max_runtime_s)
    solver = ca.nlpsol('nlp_th', 'ipopt', {'x': x, 'f': cost, 'g': g}, opts)

    # initial guess: zero thrust, linear state interpolation
    if U_guess is None:
        U0 = np.zeros(num_agents * N * 3)
    else:
        U0 = np.asarray(U_guess, dtype=float).flatten()
    x0 = np.concatenate([
        U0,
        np.linspace(bc.x0.r, bc.xf.r, N + 1).flatten(),
        np.linspace(bc.x0.v, bc.xf.v, N + 1).flatten(),
        np.linspace(phi0, phif, N + 1).flatten(),
        np.linspace(np.asarray(bc.x0.omega, dtype=float),
                    np.asarray(bc.xf.omega, dtype=float), N + 1).flatten(),
        np.maximum(np.linalg.norm(U0.reshape(num_agents, N, 3), axis=2).flatten(),
                   0.1),  # fuel slacks: interior start consistent with U guess
    ])

    original_stdout = sys.stdout
    try:
        with open(os.devnull, 'w') as fnull:
            sys.stdout = fnull
            sol = solver(x0=x0, lbx=lbx, ubx=ubx,
                         lbg=np.array(lbg, dtype=float), ubg=np.array(ubg, dtype=float))
    finally:
        sys.stdout = original_stdout

    status = solver.stats().get('return_status', 'unknown')
    w = np.asarray(sol['x']).flatten()
    nU = num_agents * N * 3
    U_opt = w[:nU].reshape(num_agents, N, 3)
    r_opt = w[nU:nU + (N + 1) * 3].reshape(N + 1, 3)
    v_opt = w[nU + (N + 1) * 3:nU + (N + 1) * 6].reshape(N + 1, 3)
    phi_opt = w[nU + (N + 1) * 6:nU + (N + 1) * 9].reshape(N + 1, 3)
    # exact slice: fuel slacks sit after ome in x
    ome_opt = w[nU + (N + 1) * 9:nU + (N + 1) * 12].reshape(N + 1, 3)
    X = np.hstack([r_opt, v_opt, phi_opt, ome_opt])  # (N+1, 12), harness layout
    runtime = time.perf_counter() - start

    return {
        "method": "centralized_nlp_th",
        "control": U_opt,
        "state": X,
        "cost": float(sol['f']),
        "runtime": runtime,
        "ipopt_status": status,
        "converged": status in _SUCCESS,
    }
