import numpy as np
import cvxpy as cp
from .orbital_helpers import orbital_params
from .dynamics import *
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from .plotters import *
from .data_structures import *
import os
import sys


def hat(v):
    """NumPy hat map: R^3 -> so(3)."""
    v = np.asarray(v).reshape(3,)
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def vee(S):
    """NumPy vee map: so(3) -> R^3."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def hat_casadi(v):
    """CasADi hat map: R^3 -> so(3)."""
    return ca.vertcat(
        ca.horzcat(0, -v[2], v[1]),
        ca.horzcat(v[2], 0, -v[0]),
        ca.horzcat(-v[1], v[0], 0),
    )


def vee_casadi(S):
    """CasADi vee map: so(3) -> R^3."""
    return ca.vertcat(S[2, 1], S[0, 2], S[1, 0])


def so3_exp(phi):
    """NumPy exponential map for SO(3) with small-angle safeguard."""
    phi = np.asarray(phi).reshape(3,)
    theta = np.linalg.norm(phi)
    K = hat(phi)
    if theta < 1e-8:
        A = 1.0 - theta**2 / 6.0 + theta**4 / 120.0
        B = 0.5 - theta**2 / 24.0 + theta**4 / 720.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / (theta**2)
    return np.eye(3) + A * K + B * (K @ K)


def so3_log(R):
    """NumPy logarithm map for SO(3) with small-angle safeguard.

    Note: all trace-based SO(3) logs are numerically delicate near pi.
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    S = R - R.T
    v = vee(S)

    if theta < 1e-8:
        return 0.5 * v

    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-8:
        return 0.5 * theta * v

    return (theta / (2.0 * sin_theta)) * v


def so3_exp_casadi(phi):
    """CasADi exponential map for SO(3)."""
    theta = ca.sqrt(ca.dot(phi, phi) + 1e-16)
    K = hat_casadi(phi)
    A_reg = ca.sin(theta) / theta
    B_reg = (1 - ca.cos(theta)) / (theta**2)
    A_series = 1 - theta**2 / 6 + theta**4 / 120
    B_series = 0.5 - theta**2 / 24 + theta**4 / 720
    use_series = theta < 1e-6
    A = ca.if_else(use_series, A_series, A_reg)
    B = ca.if_else(use_series, B_series, B_reg)
    return ca.SX_eye(3) + A * K + B * (K @ K)


def so3_log_casadi(R):
    """CasADi logarithm map for SO(3).

    Note: all trace-based SO(3) logs are numerically delicate near pi.
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    cos_theta = ca.fmax(-1, ca.fmin(1, (tr - 1) / 2))
    theta = ca.acos(cos_theta)
    S = R - R.T
    v = vee_casadi(S)

    sin_theta = ca.sin(theta)
    scale_reg = theta / (2 * sin_theta + 1e-12)
    scale = ca.if_else(theta < 1e-6, 0.5, scale_reg)
    return scale * v


def quat_to_rotmat(q):
    """Quaternion [w, x, y, z] to rotation matrix."""
    q = np.asarray(q, dtype=float).reshape(4,)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    q = q / n
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])


def rotmat_to_quat(R):
    """Rotation matrix to quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def state_attitude_to_phi(x):
    """Compatibility helper: prefer x.phi, otherwise map quaternion x.eps -> phi."""
    if hasattr(x, 'phi') and getattr(x, 'phi') is not None:
        return np.asarray(getattr(x, 'phi'), dtype=float).reshape(3,)
    if x.eps is None:
        raise ValueError("State attitude is missing: expected either phi or eps.")
    return so3_log(quat_to_rotmat(x.eps))

#cleand?
def tau_proj_nonlin_new(tau_hist, N, epsilon, sys_params: SystemParams, bc: BoundaryConditions, num_iter=None):
    phi0 = state_attitude_to_phi(bc.x0)
    phif = state_attitude_to_phi(bc.xf)
    a0 = np.hstack((phi0, bc.x0.omega))
    af = np.hstack((phif, bc.xf.omega))
    num_steps = N  # Number of steps
    dt = bc.tf / num_steps

    # Convert constants to CasADi types
    I_casadi = ca.DM(sys_params.I)
    I_inv_casadi = ca.inv(I_casadi)
    dt_casadi = ca.DM(dt)

    # Define CasADi variables for control inputs and states
    tau = [ca.SX.sym(f'tau_{k}', 3) for k in range(num_steps)]  # Control torques
    state = [ca.SX.sym(f'state_{k}', 6) for k in range(num_steps + 1)]  # State (phi + angular velocity)

    cost = 0
    for k in range(num_steps):
        cost += ca.sumsqr(tau[k] - tau_hist[k])

    constraints = []
    lbg = []
    ubg = []

    # Initial state constraint (state[0] == a0)
    constraints.append(state[0] - a0)
    lbg.extend([0] * 6)
    ubg.extend([0] * 6)

    for k in range(num_steps):
        state_k = state[k]
        tau_k = tau[k]

        phi_k = state_k[0:3]
        ome_k = state_k[3:6]

        # omega_dot = I^{-1}(tau - omega x (I omega))
        ome_dot = I_inv_casadi @ (tau_k - ca.cross(ome_k, I_casadi @ ome_k))
        ome_next = ome_k + dt_casadi * ome_dot

        # SO(3) propagation: R_{k+1} = exp(phi_k^) exp(dt*omega_k^)
        R_k = so3_exp_casadi(phi_k)
        R_next = R_k @ so3_exp_casadi(dt_casadi * ome_k)
        phi_next = so3_log_casadi(R_next)

        state_k_next = ca.vertcat(phi_next, ome_next)
        constraints.append(state[k + 1] - state_k_next)
        lbg.extend([0] * 6)
        ubg.extend([0] * 6)

    # Final state constraint (state[num_steps] == af)
    constraints.append(state[num_steps] - af)
    lbg.extend([0] * 6)
    ubg.extend([0] * 6)

    opt_vars = ca.vertcat(*tau, *state)
    g = ca.vertcat(*constraints)

    lbg = np.array(lbg)
    ubg = np.array(ubg)

    tau_lower_bound = -ca.inf * np.ones(num_steps * 3)
    tau_upper_bound = ca.inf * np.ones(num_steps * 3)

    state_lower_bound = -ca.inf * np.ones((num_steps + 1) * 6)
    state_upper_bound = ca.inf * np.ones((num_steps + 1) * 6)

    lbx = np.concatenate([tau_lower_bound, state_lower_bound])
    ubx = np.concatenate([tau_upper_bound, state_upper_bound])

    nlp = {'x': opt_vars, 'f': cost, 'g': g}

    if num_iter is None:
        opts = {'ipopt': {'print_level': 0}}
    else:
        opts = {'ipopt': {'max_iter': num_iter, 'print_level': 0}}

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    original_stdout = sys.stdout

    tau_init_guess = np.zeros(num_steps * 3)
    state_init_guess = np.linspace(a0, af, num_steps + 1).flatten()
    x0 = np.concatenate([tau_init_guess, state_init_guess])

    try:
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    finally:
        sys.stdout = original_stdout

    w_opt = sol['x'].full().flatten()
    tau_opt = w_opt[:num_steps * 3].reshape(num_steps, 3)
    state_opt = w_opt[num_steps * 3:].reshape(num_steps + 1, 6)

    return tau_opt, state_opt

def opt_given_tau_ipopt_new(tau, N, epsilon, sys_params: SystemParams, bc: BoundaryConditions, num_iter=None):

    num_steps = N
    num_agents = len(sys_params.rs)
    dt = bc.tf / num_steps

    U_guess = np.zeros((num_agents, num_steps, 3))

    # Convert constants to CasADi types
    m_casadi = ca.DM(sys_params.m)
    epsilon_casadi = epsilon
    nu_casadi = sys_params.nu

    # Flatten U into a vector symbolic variable
    U = ca.SX.sym('U', num_agents * num_steps * 3)

    # Create CasADi variables for states (flattened)
    r = ca.SX.sym('r', (num_steps + 1) * 3)
    v = ca.SX.sym('v', (num_steps + 1) * 3)

    cost = ca.sumsqr(U)

    constraints = []
    constraints_lb = []
    constraints_ub = []

    def get_U(i, k):
        idx = (i * num_steps + k) * 3
        return U[idx:idx + 3]

    def get_r(k):
        idx = k * 3
        return r[idx:idx + 3]

    def get_v(k):
        idx = k * 3
        return v[idx:idx + 3]

    constraints.extend([
        get_r(0) - bc.x0.r,
        get_v(0) - bc.x0.v,
    ])
    constraints_lb.extend([0] * 3 + [0] * 3)
    constraints_ub.extend([0] * 3 + [0] * 3)

    # Pre-propagate rotational state using Lie algebra coordinates.
    I = np.asarray(sys_params.I)
    I_inv = np.linalg.inv(I)
    phi = [state_attitude_to_phi(bc.x0) for _ in range(num_steps + 1)]
    ome = [np.asarray(bc.x0.omega, dtype=float).copy() for _ in range(num_steps + 1)]
    Rs = [np.eye(3) for _ in range(num_steps + 1)]
    Rs[0] = so3_exp(phi[0])

    for k in range(num_steps):
        ome_dot = I_inv @ (np.asarray(tau[k]) - np.cross(ome[k], I @ ome[k]))
        ome[k + 1] = ome[k] + dt * ome_dot

        Rs[k + 1] = Rs[k] @ so3_exp(dt * ome[k])
        phi[k + 1] = so3_log(Rs[k + 1])

    # Build inertial attachment trajectories directly from R_k @ r_i_body.
    qs = [np.zeros((num_steps, 3)) for _ in range(num_agents)]
    for i in range(num_agents):
        r_i_body = np.asarray(sys_params.rs[i])
        for k in range(num_steps):
            qs[i][k] = Rs[k] @ r_i_body

    for k in range(num_steps):
        torque_curr = ca.SX.zeros(3)
        force_curr = ca.SX.zeros(3)
        for i in range(num_agents):
            Q_ik = ca.DM(qs[i][k])
            U_ik = get_U(i, k)
            torque_curr += skew_casadi(Q_ik) @ U_ik
            force_curr += U_ik

            dot_product = ca.dot(U_ik, Q_ik)
            norm_U = smooth_norm(U_ik, epsilon_casadi)
            norm_Q = smooth_norm(Q_ik, epsilon_casadi)
            pointing_constraint = dot_product - ca.cos(nu_casadi) * norm_U * norm_Q
            constraints.append(pointing_constraint)
            constraints_lb.append(0)
            constraints_ub.append(ca.inf)

        constraints.append(torque_curr - tau[k])
        constraints_lb.extend([0] * 3)
        constraints_ub.extend([0] * 3)

        f, f_dot, f_ddot = orbital_params(sys_params.mu, sys_params.a, sys_params.e, k * dt)

        A_vel = ca.SX.zeros(3, 6)
        A_vel[0, :] = ca.horzcat(3 * f_dot ** 2, 0, 0, 0, 2 * f_dot, 0)
        A_vel[1, :] = ca.horzcat(0, 0, 0, -2 * f_dot, 0, 0)
        A_vel[2, :] = ca.horzcat(0, 0, -f_ddot, 0, 0, 0)
        B_vel = (1 / sys_params.m) * ca.SX_eye(3)

        r_k = get_r(k)
        v_k = get_v(k)
        r_next = r_k + dt * v_k
        v_next = v_k + dt * (A_vel @ ca.vertcat(r_k, v_k) + B_vel @ force_curr)

        constraints.extend([
            get_r(k + 1) - r_next,
            get_v(k + 1) - v_next,
        ])
        constraints_lb.extend([0] * 3 + [0] * 3)
        constraints_ub.extend([0] * 3 + [0] * 3)

    constraints.extend([
        get_r(num_steps) - bc.xf.r,
        get_v(num_steps) - bc.xf.v,
    ])
    constraints_lb.extend([0] * 3 + [0] * 3)
    constraints_ub.extend([0] * 3 + [0] * 3)

    g = ca.vertcat(*constraints)
    lbg = np.array(constraints_lb)
    ubg = np.array(constraints_ub)

    opt_vars = ca.vertcat(U, r, v)

    opt_vars_init = np.concatenate([
        U_guess.flatten(),
        np.tile(bc.x0.r, num_steps + 1),
        np.tile(bc.x0.v, num_steps + 1),
    ])

    nlp = {'x': opt_vars, 'f': cost, 'g': g}

    if num_iter is None:
        opts = {'ipopt': {'print_level': 0}}
    else:
        opts = {'ipopt': {'max_iter': num_iter, 'print_level': 0}}

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    original_stdout = sys.stdout

    try:
        with open(os.devnull, 'w') as f:
            sol = solver(x0=opt_vars_init, lbg=lbg, ubg=ubg)
    finally:
        sys.stdout = original_stdout

    w_opt = sol['x'].full().flatten()
    min_cost = sol['f'].full().item()

    num_U = num_agents * num_steps * 3
    num_r = (num_steps + 1) * 3
    num_v = (num_steps + 1) * 3

    idx = 0
    U_opt = w_opt[idx:idx + num_U].reshape(num_agents, num_steps, 3)
    idx += num_U
    r_opt = w_opt[idx:idx + num_r].reshape(num_steps + 1, 3)
    idx += num_r
    v_opt = w_opt[idx:idx + num_v].reshape(num_steps + 1, 3)

    # Keep external compatibility by packaging quaternion attitude in StateVector.
    eps_opt = np.array([rotmat_to_quat(Rs[k]) for k in range(num_steps + 1)])
    ome_opt = np.array(ome)
    Q_opt = np.array(qs)
    state_list = [
        StateVector(r=r_opt[k], v=v_opt[k], eps=eps_opt[k], omega=ome_opt[k])
        for k in range(num_steps + 1)
    ]
    traj_opt = Trajectory(states=state_list, times=np.linspace(0, bc.tf, num_steps + 1))
    ctrl_opt = ControlHistory(tau=tau, U=U_opt, force=np.sum(U_opt, axis=0), dt=dt)

    return traj_opt, ctrl_opt, Q_opt, min_cost

def opt_given_tau_cvx_new(tau,sys_params:SystemParams, bc:BoundaryConditions, num_iter=None):
    delta = 1e-9
    num_steps = len(tau)
    dt = bc.tf / num_steps

    # Define Parameters
    tau_p = cp.Parameter(tau.shape, value=tau)

    # Propagate Attitude and q vectors

    ome = [bc.x0.omega for _ in range(num_steps+1)]
    eps = [bc.x0.eps for _ in range(num_steps+1)]

        # Position and velocity update using linearized dynamics
    f, f_dot, f_ddot = orbital_params(sys_params.mu, sys_params.a, sys_params.e, bc.tf)
    A_pos_val = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    A_vel_val = np.array([
        [3 * f_dot**2, 0, 0, 0, 2 * f_dot, 0],
        [0, 0, 0, -2 * f_dot, 0, 0],
        [0, 0, -f_ddot, 0, 0, 0]
    ])
    B_vel_val = np.array([
        [1/sys_params.m, 0, 0],
        [0, 1/sys_params.m, 0],
        [0, 0, 1/sys_params.m]
    ])

    A_pos = cp.Constant(A_pos_val)
    A_vel = cp.Constant(A_vel_val)
    B_vel = cp.Constant(B_vel_val)

    for k in range(num_steps):
        rot_dot = rotational_derivative(np.hstack([eps[k], ome[k]]), tau_p[k].value,sys_params.I)

        eps[k+1] = eps[k] + dt*rot_dot[0:4]
        eps[k+1] = eps[k+1] / np.linalg.norm(eps[k+1])

        ome[k+1] = ome[k] + dt*rot_dot[4:7]

    qs = [np.zeros((num_steps, 3)) for _ in range(len(sys_params.rs))]

    for i in range(len(qs)):
        qs[i][0] = sys_params.rs[i]
        for k in range(num_steps-1):
            qs[i][k+1] = qs[i][k] + dt * skewer(ome[k]) @ qs[i][k]
            qs[i][k+1] = np.linalg.norm(sys_params.rs[i])*qs[i][k+1] / np.linalg.norm(qs[i][k+1])

    # Define Variables
    U = [cp.Variable((num_steps, 3)) for _ in range(len(qs))]
    force = cp.Variable((num_steps, 3))

    r = cp.Variable((num_steps + 1, 3)) # Position
    v = cp.Variable((num_steps + 1, 3)) # Velocity

    # Constraints list
    constraints = []

    # Initial and final conditions
    constraints.append(r[0] == bc.x0.r)
    constraints.append(v[0] == bc.x0.v)
    constraints.append(r[-1] == bc.xf.r)
    constraints.append(v[-1] == bc.xf.v)

    U_stack = cp.vstack(U)
    objective = cp.Minimize(cp.sum_squares(U_stack))

    for i in range(len(U)):
        for k in range(num_steps):
            constraints.append(U[i][k] @ qs[i][k] - np.cos(sys_params.nu) * (cp.norm(U[i][k]) * cp.norm(qs[i][k])) >= 0)

    for k in range(num_steps):
        constraints.append(force[k] == cp.sum([U[i][k] for i in range(len(U))]))
        constraints.append(tau_p[k] == cp.sum([skewer(qs[i][k]) @ U[i][k] for i in range(len(U))]))

        # Dynamics constraints for position and velocity
        constraints.append(r[k + 1] == r[k]+dt*A_pos@cp.hstack([r[k],v[k]]))
        constraints.append(v[k + 1] == v[k]+dt*A_vel@cp.hstack([r[k],v[k]]) + dt * (B_vel @ force[k]))

    prob = cp.Problem(objective, constraints)
    if num_iter is None:
        prob.solve(solver=cp.SCS, verbose=True)
    else:
        prob.solve(solver=cp.SCS, max_iters = num_iter, verbose=True)

    if prob.status in [cp.INFEASIBLE, cp.UNBOUNDED]:
        return "Unsolved X", "Unsolved U", "Unsolved Q", prob

    return np.hstack([r.value, v.value, eps, ome]), [U[i].value for i in range(len(U))], qs, prob.value
