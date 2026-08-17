"""Build-once parametric solvers shared by GA, MPPI and gradient descent.

The legacy oracle path (tau_proj_nonlin_new + opt_given_tau_ipopt_new)
reconstructs the full CasADi symbolic problem and a fresh IPOPT instance on
EVERY call — measured ~0.5-1 s per evaluation, of which the actual IPOPT solve
is a small fraction. Since every method's hot loop evaluates hundreds of tau
candidates against the SAME scenario, both problems are built here exactly
once per scenario with tau as a CasADi parameter:

  - inner thrust-allocation NLP: parametric in tau, INCLUDING the symbolic
    attitude rollout R_k(tau) (needed for gradient descent's envelope
    gradient; harmless for pure evaluation)
  - attitude projector: parametric in tau_hist (the point being projected)

Equivalence: same constraints, bounds, options and initial guesses as the
legacy functions, so solutions match the legacy path to solver tolerance.
The speedup is purely from skipping reconstruction (plus optional warm
starts on the inner problem).

Fairness note: this oracle is METHOD-NEUTRAL infrastructure. All solver
families must run through it (or none) for equal-compute comparisons.
"""
from __future__ import annotations

import os
import sys

import casadi as ca
import numpy as np

from ..data_structures import BoundaryConditions, SystemParams
from ..new_opts import (
    so3_exp,
    so3_exp_casadi,
    so3_log_casadi,
    state_attitude_to_phi,
    th_psi_matrix,
    skew_casadi,
    smooth_norm,
)

_SUCCESS = {"Solve_Succeeded", "Solved_To_Acceptable_Level"}


def _silent_call(solver, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(os.devnull, "w") as fnull:
            sys.stdout = fnull
            sol = solver(**kwargs)
    finally:
        sys.stdout = original_stdout
    return sol


def build_inner_parametric(sys_params: SystemParams, bc: BoundaryConditions, epsilon: float,
                           max_iter: int = 1000):
    """Inner thrust-allocation NLP with tau as parameter (symbolic R_k(tau)).

    Returns (solver, grad_fn, lbg, ubg, x_init, meta):
      solver(x0=..., p=tau_flat, lbg=..., ubg=...) -> sol
      grad_fn(x, lam_g, tau_flat) -> dJ/dtau via the envelope theorem
    """
    num_steps = sys_params.N
    num_agents = len(sys_params.rs)
    dt = bc.tf / num_steps

    tau_p = ca.SX.sym('tau_p', num_steps * 3)
    U = ca.SX.sym('U', num_agents * num_steps * 3)
    r = ca.SX.sym('r', (num_steps + 1) * 3)
    v = ca.SX.sym('v', (num_steps + 1) * 3)

    cost = ca.sumsqr(U)

    def get_U(i, k):
        idx = (i * num_steps + k) * 3
        return U[idx:idx + 3]

    def get_r(k):
        return r[k * 3:k * 3 + 3]

    def get_v(k):
        return v[k * 3:k * 3 + 3]

    def get_tau(k):
        return tau_p[k * 3:k * 3 + 3]

    constraints = []
    lbg = []
    ubg = []

    constraints.extend([get_r(0) - bc.x0.r, get_v(0) - bc.x0.v])
    lbg.extend([0] * 6)
    ubg.extend([0] * 6)

    I_mat = ca.DM(np.asarray(sys_params.I))
    I_inv = ca.DM(np.linalg.inv(np.asarray(sys_params.I)))
    phi0 = state_attitude_to_phi(bc.x0)
    Rs_sym = [ca.DM(so3_exp(phi0))]
    ome_k = ca.DM(np.asarray(bc.x0.omega, dtype=float))
    for k in range(num_steps):
        ome_dot = I_inv @ (get_tau(k) - ca.cross(ome_k, I_mat @ ome_k))
        R_next = Rs_sym[k] @ so3_exp_casadi(dt * ome_k)
        Rs_sym.append(R_next)
        ome_k = ome_k + dt * ome_dot

    rs_body = [np.asarray(sys_params.rs[i], dtype=float) for i in range(num_agents)]

    for k in range(num_steps):
        torque_curr = ca.SX.zeros(3)
        thrust_body = ca.SX.zeros(3)
        for i in range(num_agents):
            r_body = ca.DM(rs_body[i])
            U_ik = get_U(i, k)
            torque_curr += skew_casadi(r_body) @ U_ik
            thrust_body += U_ik

            dot_product = ca.dot(U_ik, r_body)
            norm_U = smooth_norm(U_ik, epsilon)
            norm_r = float(np.linalg.norm(rs_body[i]))
            constraints.append(dot_product - ca.cos(sys_params.nu) * norm_U * norm_r)
            lbg.append(0)
            ubg.append(ca.inf)

        constraints.append(torque_curr - get_tau(k))
        lbg.extend([0] * 3)
        ubg.extend([0] * 3)

        Psi_k = th_psi_matrix(sys_params.mu, sys_params.a, sys_params.e, k * dt)
        Psi_vel = ca.DM(Psi_k[3:6, :])
        r_k = get_r(k)
        v_k = get_v(k)
        r_next = r_k + dt * v_k
        v_next = v_k + dt * (Psi_vel @ ca.vertcat(r_k, v_k)
                             + (1.0 / sys_params.m) * Rs_sym[k] @ thrust_body)
        constraints.extend([get_r(k + 1) - r_next, get_v(k + 1) - v_next])
        lbg.extend([0] * 6)
        ubg.extend([0] * 6)

    constraints.extend([get_r(num_steps) - bc.xf.r, get_v(num_steps) - bc.xf.v])
    lbg.extend([0] * 6)
    ubg.extend([0] * 6)

    g = ca.vertcat(*constraints)
    x = ca.vertcat(U, r, v)

    nlp = {'x': x, 'p': tau_p, 'f': cost, 'g': g}
    opts = {"print_time": False,
            'ipopt': {'max_iter': max_iter, 'print_level': 0, 'sb': 'yes'}}
    solver = ca.nlpsol('inner_parametric', 'ipopt', nlp, opts)

    lam_g = ca.SX.sym('lam_g', g.shape[0])
    lagrangian = cost + ca.dot(lam_g, g)
    grad_fn = ca.Function('envelope_grad', [x, lam_g, tau_p], [ca.jacobian(lagrangian, tau_p)])

    x_init = np.concatenate([
        np.zeros(num_agents * num_steps * 3),
        np.tile(bc.x0.r, num_steps + 1),
        np.tile(bc.x0.v, num_steps + 1),
    ])
    meta = {'n_g': g.shape[0], 'n_x': x.shape[0]}
    return solver, grad_fn, np.array(lbg, dtype=float), np.array(ubg, dtype=float), x_init, meta


def build_projector_parametric(sys_params: SystemParams, bc: BoundaryConditions, epsilon: float,
                               keep_outs=None):
    """tau_proj_nonlin_new with tau_hist as a parameter. Same constraints,
    bounds and default initial guess as the legacy function.

    keep_outs: optional list of (b_body, s_inertial, theta_min_rad) attitude
    keep-out cones. For every timestep the body vector b, rotated to the
    inertial frame, must stay at least theta_min away from direction s:
    (R_k b) . s <= cos(theta_min). Nonconvex; punctures the feasible attitude
    path space and creates genuinely distinct routing basins (go left vs
    right around the forbidden cone)."""
    num_steps = sys_params.N
    dt = bc.tf / num_steps

    phi0 = state_attitude_to_phi(bc.x0)
    phif = state_attitude_to_phi(bc.xf)
    a0 = np.hstack((phi0, bc.x0.omega))
    af = np.hstack((phif, bc.xf.omega))

    I_casadi = ca.DM(np.asarray(sys_params.I))
    I_inv_casadi = ca.inv(I_casadi)
    dt_casadi = ca.DM(dt)

    tau_hist_p = ca.SX.sym('tau_hist_p', num_steps * 3)
    tau = [ca.SX.sym(f'tau_{k}', 3) for k in range(num_steps)]
    state = [ca.SX.sym(f'state_{k}', 6) for k in range(num_steps + 1)]

    cost = 0
    for k in range(num_steps):
        cost += ca.sumsqr(tau[k] - tau_hist_p[k * 3:k * 3 + 3])

    constraints = [state[0] - a0]
    lbg = [0] * 6
    ubg = [0] * 6

    for k in range(num_steps):
        phi_k = state[k][0:3]
        ome_k = state[k][3:6]
        ome_dot = I_inv_casadi @ (tau[k] - ca.cross(ome_k, I_casadi @ ome_k))
        ome_next = ome_k + dt_casadi * ome_dot
        R_k = so3_exp_casadi(phi_k)
        R_next = R_k @ so3_exp_casadi(dt_casadi * ome_k)
        phi_next = so3_log_casadi(R_next)
        constraints.append(state[k + 1] - ca.vertcat(phi_next, ome_next))
        lbg.extend([0] * 6)
        ubg.extend([0] * 6)

        if keep_outs:
            for b_body, s_inertial, theta_min in keep_outs:
                b = ca.DM(np.asarray(b_body, dtype=float))
                s_dir = ca.DM(np.asarray(s_inertial, dtype=float))
                constraints.append(ca.dot(R_k @ b, s_dir))
                lbg.append(-ca.inf)
                ubg.append(float(np.cos(theta_min)))

    constraints.append(state[num_steps] - af)
    lbg.extend([0] * 6)
    ubg.extend([0] * 6)

    opt_vars = ca.vertcat(*tau, *state)
    g = ca.vertcat(*constraints)

    phi_bound = np.pi - 1e-3
    omega_bound = 2 * (np.pi - 1e-3) / dt
    one_lb = np.array([-phi_bound] * 3 + [-omega_bound] * 3)
    one_ub = np.array([phi_bound] * 3 + [omega_bound] * 3)
    lbx = np.concatenate([-np.inf * np.ones(num_steps * 3), np.tile(one_lb, num_steps + 1)])
    ubx = np.concatenate([np.inf * np.ones(num_steps * 3), np.tile(one_ub, num_steps + 1)])

    nlp = {'x': opt_vars, 'p': tau_hist_p, 'f': cost, 'g': g}
    opts = {"print_time": False, 'ipopt': {'print_level': 0, 'sb': 'yes'}}
    solver = ca.nlpsol('proj_parametric', 'ipopt', nlp, opts)

    x0 = np.concatenate([np.zeros(num_steps * 3),
                         np.linspace(a0, af, num_steps + 1).flatten()])
    return solver, np.array(lbg, dtype=float), np.array(ubg, dtype=float), lbx, ubx, x0


class ScenarioOracle:
    """Per-scenario cache of the parametric projector + inner solver.

    project(tau)     -> (tau_proj (N,3), state_opt (N+1,6)) or (None, None)
    inner_cost(tau)  -> (ok, cost, x, lam_g); warm-started on the last solution
    grad(x, lam, tau)-> envelope dJ/dtau (for gradient descent)
    proj_compat(...) -> drop-in signature adapter for code expecting
                        tau_proj_nonlin_new(tau, N, eps, sys, bc, ...)
    """

    def __init__(self, sys_params: SystemParams, bc: BoundaryConditions, epsilon: float,
                 inner_max_iter: int = 1000, warm_start_inner: bool = True,
                 keep_outs=None):
        self.sys_params = sys_params
        self.bc = bc
        self.epsilon = epsilon
        self.N = sys_params.N
        (self._inner, self.grad_fn, self._ilbg, self._iubg,
         self._ix0_default, self.meta) = build_inner_parametric(
            sys_params, bc, epsilon, max_iter=inner_max_iter)
        (self._proj, self._plbg, self._pubg, self._plbx, self._pubx,
         self._px0) = build_projector_parametric(sys_params, bc, epsilon,
                                                 keep_outs=keep_outs)
        self._warm = warm_start_inner
        self._last_inner_x = None

    # -- projector ---------------------------------------------------------
    def project(self, tau):
        tau_flat = np.asarray(tau, dtype=float).flatten()
        try:
            sol = _silent_call(self._proj, x0=self._px0, p=tau_flat,
                               lbx=self._plbx, ubx=self._pubx,
                               lbg=self._plbg, ubg=self._pubg)
        except Exception:
            return None, None
        if self._proj.stats().get('return_status', '') not in _SUCCESS:
            return None, None
        w = np.asarray(sol['x']).flatten()
        tau_opt = w[:self.N * 3].reshape(self.N, 3)
        state_opt = w[self.N * 3:].reshape(self.N + 1, 6)
        return tau_opt, state_opt

    def proj_compat(self, tau_hist, N=None, epsilon=None, sys_params=None, bc=None,
                    num_iter=None, allow_raising_error=False):
        tau_opt, state_opt = self.project(tau_hist)
        if tau_opt is None:
            if allow_raising_error:
                return None, None
            raise RuntimeError("parametric projector failed")
        return tau_opt, state_opt

    # -- inner problem ------------------------------------------------------
    def inner_cost(self, tau):
        tau_flat = np.asarray(tau, dtype=float).flatten()
        x0 = self._last_inner_x if (self._warm and self._last_inner_x is not None) \
            else self._ix0_default
        try:
            sol = _silent_call(self._inner, x0=x0, p=tau_flat,
                               lbg=self._ilbg, ubg=self._iubg)
        except Exception:
            return False, float('inf'), None, None
        ok = self._inner.stats().get('return_status', '') in _SUCCESS
        x = np.asarray(sol['x']).flatten()
        lam = np.asarray(sol['lam_g']).flatten()
        if ok and self._warm:
            self._last_inner_x = x
        cost = float(sol['f'])
        return ok, cost, x, lam

    def grad(self, x, lam, tau):
        tau_flat = np.asarray(tau, dtype=float).flatten()
        return np.asarray(self.grad_fn(x, lam, tau_flat)).flatten()

    # -- combined hot-path evaluation (project then allocate) ---------------
    def evaluate(self, tau):
        """Mirror of mppi_core.evaluate_tau: (projected_tau, cost); +inf on
        any failure."""
        tau_proj, _ = self.project(tau)
        if tau_proj is None:
            return np.asarray(tau, dtype=float).reshape(self.N, 3), float('inf')
        ok, cost, _, _ = self.inner_cost(tau_proj)
        if not ok or not np.isfinite(cost) or cost <= 0:
            return tau_proj, float('inf')
        return tau_proj, cost
