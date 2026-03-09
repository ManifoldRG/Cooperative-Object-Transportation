from .data_structures import *
import casadi as ca
from .dynamics import *
import sys
import os
from .new_opts import tau_proj_nonlin_new, opt_given_tau_ipopt_new, opt_given_tau_cvx_new

#og version
def multiple_shooting_optimization(a0, af, num_steps, dt_casadi, I_casadi, epsilon_casadi, num_iter=None):
    # Define CasADi variables for control inputs and states
    tau = [ca.SX.sym(f'tau_{k}', 3) for k in range(num_steps)]  # Control torques
    state = [ca.SX.sym(f'state_{k}', 7) for k in range(num_steps + 1)]  # State (quaternion + angular velocity)
    cost = 0

    for k in range(num_steps):
        cost += ca.sumsqr(tau[k])  # Minimize control effort

    # Initialize constraints list
    constraints = []
    lbg = []
    ubg = []

    # Initial state constraint (state[0] == a0)
    constraints.append(state[0] - a0)
    lbg.extend([0] * 7)
    ubg.extend([0] * 7)

    # Dynamics constraints using multiple shooting
    for k in range(num_steps):
        state_k = state[k]
        tau_k = tau[k]

        # Propagate the state to the next step using Euler integration
        state_k_next = state_k + dt_casadi * rotational_derivative_casadi(state_k, tau_k, I_casadi)

        # Normalize the quaternion using smooth norm
        quat_next = state_k_next[0:4] / smooth_norm(state_k_next[0:4], epsilon_casadi)
        state_k_next_normalized = ca.vertcat(quat_next, state_k_next[4:7])

        # Append the dynamics constraint (ensure continuity between intervals)
        constraints.append(state[k + 1] - state_k_next_normalized)
        lbg.extend([0] * 7)
        ubg.extend([0] * 7)

    # Final state constraint (state[num_steps] == af)
    constraints.append(state[num_steps] - af)
    lbg.extend([0] * 7)
    ubg.extend([0] * 7)

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
                'max_iter': num_iter,
                'print_level': 0,  # Solver verbosity level
            }
        }
    else:
        opts = {
            'ipopt': {
                'print_level': 0,  # Solver verbosity level
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
    state_opt = w_opt[num_steps * 3:].reshape(num_steps + 1, 7)  # Optimized states

    return tau_opt, state_opt

#cleaned ver
def multiple_shooting_optimization_new(bc: BoundaryConditions, num_steps, dt_casadi, I_casadi, epsilon_casadi, num_iter=None):
    a0 = np.hstack((bc.x0.eps, bc.x0.omega))
    af = np.hstack((bc.xf.eps, bc.xf.omega))
    # Define CasADi variables for control inputs and states
    tau = [ca.SX.sym(f'tau_{k}', 3) for k in range(num_steps)]  # Control torques
    state = [ca.SX.sym(f'state_{k}', 7) for k in range(num_steps + 1)]  # State (quaternion + angular velocity)
    cost = 0

    for k in range(num_steps):
        cost += ca.sumsqr(tau[k])  # Minimize control effort

    # Initialize constraints list
    constraints = []
    lbg = []
    ubg = []

    # Initial state constraint (state[0] == a0)
    constraints.append(state[0] - a0)
    lbg.extend([0] * 7)
    ubg.extend([0] * 7)

    # Dynamics constraints using multiple shooting
    for k in range(num_steps):
        state_k = state[k]
        tau_k = tau[k]

        # Propagate the state to the next step using Euler integration
        state_k_next = state_k + dt_casadi * rotational_derivative_casadi(state_k, tau_k, I_casadi)

        # Normalize the quaternion using smooth norm
        quat_next = state_k_next[0:4] / smooth_norm(state_k_next[0:4], epsilon_casadi)
        state_k_next_normalized = ca.vertcat(quat_next, state_k_next[4:7])

        # Append the dynamics constraint (ensure continuity between intervals)
        constraints.append(state[k + 1] - state_k_next_normalized)
        lbg.extend([0] * 7)
        ubg.extend([0] * 7)

    # Final state constraint (state[num_steps] == af)
    constraints.append(state[num_steps] - af)
    lbg.extend([0] * 7)
    ubg.extend([0] * 7)

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
                'max_iter': num_iter,
                'print_level': 0,  # Solver verbosity level
            }
        }
    else:
        opts = {
            'ipopt': {
                'print_level': 0,  # Solver verbosity level
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
    state_opt = w_opt[num_steps * 3:].reshape(num_steps + 1, 7)  # Optimized states

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
def fitness_func(ga_instance,N,epsilon, sys_params: SystemParams, bc: BoundaryConditions, solution, solution_idx):
    # Reshape the solution into the correct tau format (num_steps x 3)
    tau = solution.reshape((N, 3))
    print("Using Nonlinear Torque Projection")
    tau = tau_proj_nonlin_new(tau, N, epsilon, sys_params, bc)[0]
    #NOTE: 3000 kept
    traj, _,_, cost = opt_given_tau_ipopt_new(tau,N, epsilon, sys_params, bc, num_iter=3000) #outputs of this function are traj_opt, ctrl_opt, Q_opt, and min cost

    fin_ang_state = np.concatenate((traj.states[-1].eps, traj.states[-1].omega))
    expected_final_ang_state = np.concatenate((bc.xf.eps, bc.xf.omega))
    alpha=1e4

    #fitness = 1 / (cost + (alpha*float(np.linalg.norm(fin_ang_state-expected_final_ang_state))**2))
    fitness = 1 / (cost)

    print("Current Cost: " + str(cost))
    if fitness == np.inf:
      fitness = 0
    return fitness