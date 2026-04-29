from .data_structures import *
import casadi as ca
from .dynamics import *
import sys
import os
from .new_opts import tau_proj_nonlin_new, opt_given_tau_ipopt_new, opt_given_tau_cvx_new


def _quaternion_to_phi(q):
    """Convert quaternion [w, x, y, z] to so(3) rotation vector phi."""
    q = np.asarray(q).flatten()
    if q.size == 3:
        return q
    if q.size != 4:
        raise ValueError("Expected boundary attitude as phi(3) or quaternion(4).")

    q = q / np.linalg.norm(q)
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    v_norm = np.linalg.norm(v)

    if v_norm < 1e-12:
        return np.zeros(3)

    angle = 2.0 * np.arctan2(v_norm, w)
    axis = v / v_norm
    return angle * axis


#og version
def multiple_shooting_optimization(a0, af, num_steps, dt_casadi, I_casadi, epsilon_casadi=None, num_iter=None):
    # Define CasADi variables for control inputs and states
    tau = [ca.SX.sym(f'tau_{k}', 3) for k in range(num_steps)]  # Control torques
    state = [ca.SX.sym(f'state_{k}', 6) for k in range(num_steps + 1)]  # State (phi + angular velocity)
    cost = 0

    for k in range(num_steps):
        cost += ca.sumsqr(tau[k])  # Minimize control effort

    # Initialize constraints list
    constraints = []
    lbg = []
    ubg = []

    # Initial state constraint (state[0] == a0)
    constraints.append(state[0] - a0)
    lbg.extend([0] * 6)
    ubg.extend([0] * 6)

    # Dynamics constraints using multiple shooting
    for k in range(num_steps):
        state_k = state[k]
        tau_k = tau[k]

        # Propagate the state on SO(3) using the exponential map.
        state_k_next = rotational_step_casadi(state_k, tau_k, dt_casadi, I_casadi)

        # Append the dynamics constraint (ensure continuity between intervals)
        constraints.append(state[k + 1] - state_k_next)
        lbg.extend([0] * 6)
        ubg.extend([0] * 6)

    # Final state constraint (state[num_steps] == af)
    constraints.append(state[num_steps] - af)
    lbg.extend([0] * 6)
    ubg.extend([0] * 6)

    # Set up the optimization problem
    opt_vars = ca.vertcat(*tau, *state)  # Decision variables (controls + states)
    g = ca.vertcat(*constraints)  # All constraints

    # Convert bounds lists to NumPy arrays
    lbg = np.array(lbg)
    ubg = np.array(ubg)

    # Set bounds for tau (optional: if you want to limit tau further)
    tau_lower_bound = -ca.inf * np.ones(num_steps * 3)  # Set torque lower bound to -100
    tau_upper_bound = ca.inf * np.ones(num_steps * 3)  # Set torque upper bound to 100

    ## For state variables, we leave them unbounded
    state_lower_bound = -ca.inf * np.ones((num_steps + 1) * 6)  # 6 state variables per step
    state_upper_bound = ca.inf * np.ones((num_steps + 1) * 6)

    # Combine bounds for tau and state variables
    lbx = np.concatenate([tau_lower_bound, state_lower_bound])
    ubx = np.concatenate([tau_upper_bound, state_upper_bound])

    # Define the NLP problem
    nlp = {'x': opt_vars, 'f': cost, 'g': g}

    if num_iter is None:
        opts = {
            'ipopt': {
                'max_iter': num_iter,
                'print_level': 0,  # Solver verbosity level
                'sb': 'yes'
            }
        }
    else:
        opts = {
            'ipopt': {
                'print_level': 0,  # Solver verbosity level
                'sb': 'yes'
            }
        }

    # Define initial guess for tau and state
    tau_init_guess = np.random.uniform(-1e-6, 1e-6, num_steps * 3)  # Random initial guess for tau

    # Set initial state guess based on linear interpolation between a0 and af
    state_init_guess = np.linspace(a0, af, num_steps + 1)

    # Flatten the initial guess
    state_init_guess = state_init_guess.flatten()

    # Combine the initial guesses
    x0 = np.concatenate([tau_init_guess, state_init_guess])

    original_stdout = sys.stdout
    # Create solver instance
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Run the solver

    try:
        # Open os.devnull to redirect stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f  # Redirect stdout to /dev/null
            sol = solver(x0=x0, lbg=lbg, ubg=ubg, ubx=ubx, lbx=lbx)
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout

    # Extract the solution
    w_opt = sol['x'].full().flatten()

    # Split the solution into tau and state
    tau_opt = w_opt[:num_steps * 3].reshape(num_steps, 3)  # Optimized torques
    state_opt = w_opt[num_steps * 3:].reshape(num_steps + 1, 6)  # Optimized states

    return tau_opt, state_opt

#cleaned ver
def multiple_shooting_optimization_new(bc: BoundaryConditions, num_steps, dt_casadi, I_casadi, epsilon_casadi=None, num_iter=None):
    # Boundary angular state is now [phi, omega] in R^6.
    a0 = np.hstack((_quaternion_to_phi(bc.x0.eps), bc.x0.omega))
    af = np.hstack((_quaternion_to_phi(bc.xf.eps), bc.xf.omega))
    # Define CasADi variables for control inputs and states
    tau = [ca.SX.sym(f'tau_{k}', 3) for k in range(num_steps)]  # Control torques
    state = [ca.SX.sym(f'state_{k}', 6) for k in range(num_steps + 1)]  # State (phi + angular velocity)
    cost = 0

    for k in range(num_steps):
        cost += ca.sumsqr(tau[k])  # Minimize control effort

    # Initialize constraints list
    constraints = []
    lbg = []
    ubg = []

    # Initial state constraint (state[0] == a0)
    constraints.append(state[0] - a0)
    lbg.extend([0] * 6)
    ubg.extend([0] * 6)

    # Dynamics constraints using multiple shooting
    for k in range(num_steps):
        state_k = state[k]
        tau_k = tau[k]

        # Propagate the state on SO(3) using the exponential map.
        state_k_next = rotational_step_casadi(state_k, tau_k, dt_casadi, I_casadi)

        # Append the dynamics constraint (ensure continuity between intervals)
        constraints.append(state[k + 1] - state_k_next)
        lbg.extend([0] * 6)
        ubg.extend([0] * 6)

    # Final state constraint (state[num_steps] == af)
    constraints.append(state[num_steps] - af)
    lbg.extend([0] * 6)
    ubg.extend([0] * 6)

    # Set up the optimization problem
    opt_vars = ca.vertcat(*tau, *state)  # Decision variables (controls + states)
    g = ca.vertcat(*constraints)  # All constraints

    # Convert bounds lists to NumPy arrays
    lbg = np.array(lbg)
    ubg = np.array(ubg)

    # Set bounds for tau (optional: if you want to limit tau further)
    tau_lower_bound = -ca.inf * np.ones(num_steps * 3)  # Set torque lower bound to -100
    tau_upper_bound = ca.inf * np.ones(num_steps * 3)  # Set torque upper bound to 100

    ## For state variables, we leave them unbounded
    state_lower_bound = -ca.inf * np.ones((num_steps + 1) * 6)  # 6 state variables per step
    state_upper_bound = ca.inf * np.ones((num_steps + 1) * 6)

    # Combine bounds for tau and state variables
    lbx = np.concatenate([tau_lower_bound, state_lower_bound])
    ubx = np.concatenate([tau_upper_bound, state_upper_bound])

    # Define the NLP problem
    nlp = {'x': opt_vars, 'f': cost, 'g': g}

    if num_iter is None:
        opts = {
            'ipopt': {
                'max_iter': num_iter,
                'print_level': 0,  # Solver verbosity level
                'sb': 'yes'
            }
        }
    else:
        opts = {
            'ipopt': {
                'print_level': 0,  # Solver verbosity level
                'sb': 'yes'
            }
        }

    # Define initial guess for tau and state
    tau_init_guess = np.random.uniform(-1e-6, 1e-6, num_steps * 3)  # Random initial guess for tau

    # Set initial state guess based on linear interpolation between a0 and af
    state_init_guess = np.linspace(a0, af, num_steps + 1)

    # Flatten the initial guess
    state_init_guess = state_init_guess.flatten()

    # Combine the initial guesses
    x0 = np.concatenate([tau_init_guess, state_init_guess])

    original_stdout = sys.stdout
    # Create solver instance
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Run the solver

    try:
        # Open os.devnull to redirect stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f  # Redirect stdout to /dev/null
            sol = solver(x0=x0, lbg=lbg, ubg=ubg, ubx=ubx, lbx=lbx)
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout

    # Extract the solution
    w_opt = sol['x'].full().flatten()

    # Split the solution into tau and state
    tau_opt = w_opt[:num_steps * 3].reshape(num_steps, 3)  # Optimized torques
    state_opt = w_opt[num_steps * 3:].reshape(num_steps + 1, 6)  # Optimized states

    return tau_opt, state_opt


#clean up and use the latest data structures
def pop_gen(a0, af, tf, I,  N, epsilon,pop_size=50, num_iter=None):
    num_steps = N  # Number of steps
    dt = tf / num_steps

    # Convert constants to CasADi types
    I_casadi = ca.DM(I)
    dt_casadi = ca.DM(dt)
    epsilon_casadi = ca.DM(epsilon)

    # Generate a population of feasible solutions
    population = []
    for _ in range(pop_size):
        # Solve the multiple-shooting optimization for each individual in the population
        tau_opt, _ = multiple_shooting_optimization(a0, af,tf, dt_casadi, I_casadi, epsilon_casadi, num_steps)
        population.append(tau_opt)

    return population

#clean up and use the latest data structures
def pop_gen_new(bc:BoundaryConditions, sys_params:SystemParams,  N, epsilon,pop_size=50, num_iter=None):
    num_steps = N  # Number of steps
    dt = bc.tf / num_steps

    # Convert constants to CasADi types
    I_casadi = ca.DM(sys_params.I)
    dt_casadi = ca.DM(dt)
    epsilon_casadi = ca.DM(epsilon)

    # Generate a population of feasible solutions
    population = []
    for _ in range(pop_size):
        tau_opt, state_opt = multiple_shooting_optimization_new(bc,num_steps, dt_casadi, I_casadi, epsilon_casadi)
        population.append(tau_opt)
    return population

#clean up to use data structures
def fitness_func(ga_instance,N,epsilon, sys_params: SystemParams, bc: BoundaryConditions, solution, solution_idx, projector=None):
    # Reshape the solution into the correct tau format (num_steps x 3)
    tau = solution.reshape((N, 3))
    if projector is None:
        projector = tau_proj_nonlin_new
    # Project + thrust-allocate. A bad candidate (e.g. one that drives the
    # linear-Euler quaternion projector into a regime where the thrust-allocation
    # IPOPT fails) must NOT take down the whole GA run — we just return zero
    # fitness for that individual so it gets selected against.
    try:
        tau = projector(tau, N, epsilon, sys_params, bc)[0]
        traj, _,_, cost = opt_given_tau_ipopt_new(tau,N, epsilon, sys_params, bc, num_iter=3000)
    except Exception:
        return 0.0

    if not np.isfinite(cost) or cost <= 0:
        return 0.0

    fitness = 1 / cost
    if fitness == np.inf:
      fitness = 0
    return fitness