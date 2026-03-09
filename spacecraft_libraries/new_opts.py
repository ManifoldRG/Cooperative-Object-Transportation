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

#cleand?
def tau_proj_nonlin_new(tau_hist, N, epsilon, sys_params: SystemParams, bc: BoundaryConditions, num_iter=None):
    a0 = np.hstack((bc.x0.eps, bc.x0.omega))
    af = np.hstack((bc.xf.eps, bc.xf.omega))
    num_steps = N  # Number of steps
    dt = bc.tf / num_steps

    # Convert constants to CasADi types
    I_casadi = ca.DM(sys_params.I)
    dt_casadi = ca.DM(dt)
    epsilon_casadi = ca.DM(epsilon)

    # Define CasADi variables for control inputs and states
    tau = [ca.SX.sym(f'tau_{k}', 3) for k in range(num_steps)]  # Control torques
    state = [ca.SX.sym(f'state_{k}', 7) for k in range(num_steps + 1)]  # State (quaternion + angular velocity)

    cost = 0

    for k in range(num_steps):
        cost += ca.sumsqr(tau[k]-tau_hist[k])  # Minimize control effort

    # Initialize constraints list
    constraints = []
    lbg = []
    ubg = []

    # Initial state constraint (state[0] == a0)
    constraints.append(state[0] - a0)
    lbg.extend([0]*7)
    ubg.extend([0]*7)

    for k in range(num_steps):
        state_k = state[k]
        tau_k = tau[k]

        # Extract the quaternion and angular velocity from the current state
        eps_k = state_k[0:4]   # Quaternion part
        ome_k = state_k[4:7]  # Angular velocity part

        # Calculate the phi matrix for current angular velocity
        phi = Phi_casadi(ome_k, I_casadi)

        # Propagate quaternion and angular velocity using the provided dynamics
        rotational_update = phi @ ca.vertcat(eps_k, ome_k)
        eps_next = eps_k + dt_casadi * rotational_update[0:4]  # Quaternion update
        ome_next = ome_k + dt_casadi * rotational_update[4:7] + dt_casadi * (ca.inv(I_casadi) @ tau_k)  # Angular velocity update

        # Normalize the quaternion using smooth norm
        quat_next = eps_next / smooth_norm(eps_next, epsilon_casadi)

        # Combine the normalized quaternion and updated angular velocity
        state_k_next_normalized = ca.vertcat(quat_next, ome_next)

        # Append the dynamics constraint (ensure continuity between intervals)
        constraints.append(state[k + 1] - state_k_next_normalized)
        lbg.extend([0] * 7)
        ubg.extend([0] * 7)

    # Final state constraint (state[num_steps] == af)
    constraints.append(state[num_steps] - af)
    lbg.extend([0]*7)
    ubg.extend([0]*7)

    # Set up the optimization problem
    opt_vars = ca.vertcat(*tau, *state)  # Decision variables (controls + states)
    g = ca.vertcat(*constraints)  # All constraints

    # Convert bounds lists to NumPy arrays
    lbg = np.array(lbg)
    ubg = np.array(ubg)

    # Set bounds for tau (optional: if you want to limit tau further)
    tau_lower_bound = -ca.inf * np.ones(num_steps * 3)
    tau_upper_bound = ca.inf * np.ones(num_steps * 3)

    # For state variables, we leave them unbounded
    state_lower_bound = -ca.inf * np.ones((num_steps + 1) * 7)  # 7 state variables per step
    state_upper_bound = ca.inf * np.ones((num_steps + 1) * 7)

    # Combine bounds for tau and state variables
    lbx = np.concatenate([tau_lower_bound, state_lower_bound])
    ubx = np.concatenate([tau_upper_bound, state_upper_bound])

    # Define the NLP problem
    nlp = {'x': opt_vars, 'f': cost, 'g': g}

    if num_iter is None:
        opts = {
            'ipopt': {
                'print_level': 0,  # Solver verbosity level
            }
        }
    else:
        opts = {
            'ipopt': {
                'max_iter': num_iter,
                'print_level': 0,  # Solver verbosity level
            }
        }

    # Create a solver instance with IPOPT
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

            # Store the original stdout
    original_stdout = sys.stdout

    # Define initial guess for tau and state
    tau_init_guess = np.zeros(num_steps * 3)

    # Set initial state guess based on linear interpolation between a0 and af
    state_init_guess = np.linspace(a0, af, num_steps + 1)

    # Flatten the initial guess
    state_init_guess = state_init_guess.flatten()

    # Combine the initial guesses
    x0 = np.concatenate([tau_init_guess, state_init_guess])

    # Solve the optimization problem
    try:
        # Open os.devnull to redirect stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f  # Redirect stdout to /dev/null
            # Run the solver
            sol = solver(x0=x0, lbg=lbg, ubg=ubg)

    finally:
        # Restore stdout to its original state
        sys.stdout = original_stdout

    # Extract the solution
    w_opt = sol['x'].full().flatten()

    # Split the solution into tau and state
    tau_opt = w_opt[:num_steps * 3].reshape(num_steps, 3)  # Optimized torques
    state_opt = w_opt[num_steps * 3:].reshape(num_steps + 1, 7)  # Optimized states

    return tau_opt, state_opt

# cleaned?
def opt_given_tau_ipopt_new(tau, N, epsilon, sys_params: SystemParams, bc: BoundaryConditions, num_iter=None):

    num_steps = N
    num_agents = len(sys_params.rs)
    dt = bc.tf / num_steps

    U_guess = np.zeros((num_agents, num_steps, 3))

    # Convert constants to CasADi types
    I_casadi = ca.DM(sys_params.I)
    m_casadi = ca.DM(sys_params.m)
    dt_casadi = dt  # dt is already a scalar, so no need to convert
    epsilon_casadi = epsilon  # Assuming epsilon is a scalar
    nu_casadi = sys_params.nu  # nu in radians

    # Flatten U into a vector symbolic variable
    U = ca.SX.sym('U', num_agents * num_steps * 3)  # Control inputs

    # Create CasADi variables for states (flattened)
    r = ca.SX.sym('r', (num_steps + 1) * 3)
    v = ca.SX.sym('v', (num_steps + 1) * 3)

    # Define the cost function
    cost = ca.sumsqr(U)  # Sum of squares of control inputs over all agents and timesteps

    # Define constraints
    constraints = []
    constraints_lb = []
    constraints_ub = []

    # Helper functions to index into flattened arrays
    def get_U(i, k):
        idx = (i * num_steps + k) * 3
        return U[idx:idx + 3]

    def get_r(k):
        idx = k * 3
        return r[idx:idx + 3]

    def get_v(k):
        idx = k * 3
        return v[idx:idx + 3]

    # Initial conditions constraints
    constraints.extend([
        get_r(0) - bc.x0.r,
        get_v(0) - bc.x0.v,
    ])
    constraints_lb.extend([0] * 3 + [0] * 3)
    constraints_ub.extend([0] * 3 + [0] * 3)

    ome = [bc.x0.omega for _ in range(num_steps+1)]
    eps = [bc.x0.eps for _ in range(num_steps+1)]

    for k in range(num_steps):
        rot_dot = rotational_derivative(np.hstack([eps[k], ome[k]]), tau[k],sys_params.I)

        eps[k+1] = eps[k] + dt*rot_dot[0:4]
        eps[k+1] = eps[k+1] / np.linalg.norm(eps[k+1])

        ome[k+1] = ome[k] + dt*rot_dot[4:7]

    qs = [np.zeros((num_steps, 3)) for _ in range(len(sys_params.rs))]

    for i in range(len(qs)):
        qs[i][0] = sys_params.rs[i]
        for k in range(num_steps-1):
            qs[i][k+1] = qs[i][k] + dt * skewer(ome[k]) @ qs[i][k]
            qs[i][k+1] = np.linalg.norm(sys_params.rs[i])*qs[i][k+1] / np.linalg.norm(qs[i][k+1])

    # Dynamics equations and constraints
    for k in range(num_steps):
        # Compute torque and force
        torque_curr = ca.SX.zeros(3)
        force_curr = ca.SX.zeros(3)
        for i in range(num_agents):
            Q_ik = ca.DM(qs[i][k])
            U_ik = get_U(i, k)
            torque_curr += skew_casadi(Q_ik) @ U_ik
            force_curr += U_ik

            # Pointing Constraint
            dot_product = ca.dot(U_ik, Q_ik)
            norm_U = smooth_norm(U_ik, epsilon_casadi)
            norm_Q = smooth_norm(Q_ik, epsilon_casadi)
            pointing_constraint = dot_product - ca.cos(nu_casadi) * norm_U * norm_Q
            constraints.append(pointing_constraint)
            constraints_lb.append(0)
            constraints_ub.append(ca.inf)  # No upper bound

        # Torque constraint: sum_i skew(Q_i) * U_i == tau
        constraints.append(torque_curr - tau[k])
        constraints_lb.extend([0] * 3)
        constraints_ub.extend([0] * 3)

        # Orbital parameters
        f, f_dot, f_ddot = orbital_params(sys_params.mu, sys_params.a, sys_params.e, k * dt)

        A_vel = ca.SX.zeros(3, 6)
        A_vel[0, :] = ca.horzcat(3 * f_dot ** 2, 0, 0, 0, 2 * f_dot, 0)
        A_vel[1, :] = ca.horzcat(0, 0, 0, -2 * f_dot, 0, 0)
        A_vel[2, :] = ca.horzcat(0, 0, -f_ddot, 0, 0, 0)
        B_vel = (1 / sys_params.m) * ca.SX_eye(3)

        # State updates for position and velocity
        r_k = get_r(k)
        v_k = get_v(k)
        r_next = r_k + dt * v_k
        v_next = v_k + dt * (A_vel @ ca.vertcat(r_k, v_k) + B_vel @ force_curr)

        # Append dynamics constraints for position and velocity
        constraints.extend([
            get_r(k + 1) - r_next,
            get_v(k + 1) - v_next,
        ])
        constraints_lb.extend([0] * 3 + [0] * 3)
        constraints_ub.extend([0] * 3 + [0] * 3)


    # Final state constraints for position and velocity
    constraints.extend([
        get_r(num_steps) - bc.xf.r,
        get_v(num_steps) - bc.xf.v,
    ])
    constraints_lb.extend([0] * 3 + [0] * 3)
    constraints_ub.extend([0] * 3 + [0] * 3)

    # Flatten constraints
    g = ca.vertcat(*constraints)
    lbg = np.array(constraints_lb)
    ubg = np.array(constraints_ub)

    # Decision variables
    opt_vars = ca.vertcat(U, r, v)

    # Initial guess
    opt_vars_init = np.concatenate([
        U_guess.flatten(),
        np.tile(bc.x0.r, num_steps + 1),
        np.tile(bc.x0.v, num_steps + 1),
    ])

    # Define NLP problem
    nlp = {'x': opt_vars, 'f': cost, 'g': g}

    if num_iter is None:
        opts = {'ipopt': {'print_level': 0}}
    else:
        opts = {'ipopt': {'max_iter': num_iter, 'print_level': 0}}

    # Create solver instance
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    original_stdout = sys.stdout

    # Solve the optimization problem
    try:
        # Open os.devnull to redirect stdout
        with open(os.devnull, 'w') as f:
            #sys.stdout = f  # Redirect stdout to /dev/null
            # Run the solver
            sol = solver(x0=opt_vars_init, lbg=lbg, ubg=ubg)

    finally:
        # Restore stdout to its original state
        sys.stdout = original_stdout

    # Extract the solution
    w_opt = sol['x'].full().flatten()
    min_cost = sol['f'].full().item()

    # Extract optimized variables
    num_U = num_agents * num_steps * 3
    num_r = (num_steps + 1) * 3
    num_v = (num_steps + 1) * 3

    idx = 0
    U_opt = w_opt[idx:idx + num_U].reshape(num_agents, num_steps, 3)
    idx += num_U
    r_opt = w_opt[idx:idx + num_r].reshape(num_steps + 1, 3)
    idx += num_r
    v_opt = w_opt[idx:idx + num_v].reshape(num_steps + 1, 3)

    # Convert lists to arrays
    eps_opt = np.array(eps)
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