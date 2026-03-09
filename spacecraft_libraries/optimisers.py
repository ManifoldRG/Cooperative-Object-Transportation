import numpy as np
import cvxpy as cp
from.orbital_helpers import orbital_params
from .dynamics import *
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from .plotters import *
from .data_structures import *



#custom optimiser given torque using the data structures
def opt_given_tau_clean(ctrl_hist: ControlHistory, sys_params: SystemParams, bc: BoundaryConditions, grad_needed=True):
    """
    Same thing as before, solves for optimal cost given a specific torque vector
    :param ctrl_hist: control history to update and return the control values in
    :param sys_params: system parameters
    :param bc: boundary conditions
    :param grad_needed: is gradient needed
    :return: trajectory, control history, gradient if needed, qs values and optimal cost
    """
    #time setup
    tau=ctrl_hist.tau
    num_steps = tau.shape[0]
    dt = bc.tf/num_steps
    ctrl_hist.dt =dt

    #defining torque in cvxpy for better opt i.e. giving it the same shape and the values as the actual tau history that we have
    tau_p = cp.Parameter(ctrl_hist.tau.shape, value=ctrl_hist.tau)

    #propagation data
    rot_sv =[StateVector(
            r=None,
            v=None,
            eps=bc.x0.eps.copy(),
            omega=bc.x0.omega.copy()
        ) for _ in range(num_steps+1)]

    for k in range(num_steps):
        # Runge-Kutta 1st Order Method
        eps_dot_1, ome_dot_1 = another_state_derivative(tau_p[k].value, rot_sv[k], sys_params)

        rot_sv[k + 1].eps = rot_sv[k].eps + dt * eps_dot_1
        rot_sv[k + 1].eps = rot_sv[k + 1].eps / np.linalg.norm(rot_sv[k + 1].eps)

        rot_sv[k + 1].omega =  rot_sv[k].omega + dt * ome_dot_1

    qs = [np.zeros((num_steps, 3)) for _ in range(len(sys_params.rs))]

    for i in range(len(qs)):
        qs[i][0] = sys_params.rs[i]
        for k in range(num_steps - 1):
            qs[i][k + 1] = qs[i][k] + dt * skewer(rot_sv[k].omega) @ qs[i][k]
            qs[i][k + 1] = np.linalg.norm(sys_params.rs[i]) * qs[i][k + 1] / np.linalg.norm(qs[i][k + 1])

    # Define Variables
    U = [cp.Variable((num_steps, 3)) for _ in range(len(qs))]
    force = cp.Variable((num_steps, 3))

    r = cp.Variable((num_steps + 1, 3))  # Position
    v = cp.Variable((num_steps + 1, 3))  # Velocity

    # Constraints list
    constraints = []

    # Initial and final conditions
    constraints.append(r[0] == bc.x0.r)
    constraints.append(v[0] == bc.x0.v)
    constraints.append(r[-1] == bc.xf.r)
    constraints.append(v[-1] == bc.xf.v)

    objective = cp.Minimize(sum([cp.sum_squares(U[i]) for i in range(len(U))]))

    for k in range(num_steps):
        constraints.append(force[k] == cp.sum([U[i][k] for i in range(len(U))]))
        constraints.append(tau_p[k] == cp.sum([skewer(qs[i][k]) @ U[i][k] for i in range(len(U))]))

        # Position and velocity update using linearized dynamics
        f, f_dot, f_ddot = orbital_params(sys_params.mu, sys_params.a, sys_params.e, k * dt)
        A_pos = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        A_vel = np.array([
            [3 * f_dot ** 2, 0, 0, 1, 2 * f_dot, 0],
            [0, 0, 0, -2 * f_dot, 1, 0],
            [0, 0, -f_ddot, 0, 0, 1]
        ])
        B_vel = np.array([
            [1 / sys_params.m, 0, 0],
            [0, 1 / sys_params.m, 0],
            [0, 0, 1 / sys_params.m]
        ])

        # Dynamics constraints for position and velocity
        constraints.append(r[k + 1] == r[k] + dt * A_pos @ cp.hstack([r[k], v[k]]))
        constraints.append(v[k + 1] == v[k] + dt * A_vel @ cp.hstack([r[k], v[k]]) + dt * (B_vel @ force[k]))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=100000, verbose=True)

    ctrl_hist.U = [Ui.value for Ui in U]
    ctrl_hist.force = force.value

    states=[]
    for k in range(num_steps+1):
        states.append(StateVector(r=r[k].value.copy(), v=v[k].value.copy(), eps=rot_sv[k].eps.copy(), omega=rot_sv[k].omega.copy()))

    time_taken = np.linspace(0, bc.tf, num_steps+1)
    traj_opt = Trajectory(states=states, times=time_taken)
    if grad_needed:
        prob.solve(solver=cp.SCS, requires_grad=grad_needed, max_iters=100000)
        prob.backward()
        print("Value w/ grad:" + str(prob.value))
        print(prob.status)
        prob.solve(solver=cp.SCS, requires_grad=False, max_iters=100000)
        print("Value w/o grad:" + str(prob.value))
        print(prob.status)
        return traj_opt, ctrl_hist, tau_p.gradient, qs, prob.value
    else:
        prob.solve(solver=cp.SCS, requires_grad=grad_needed, max_iters=10000000, verbose=False)
        return traj_opt, ctrl_hist, qs, prob.value

# custom one with the data structures created
def tau_projection_linear_clean(alpha, grad, bc: BoundaryConditions, tau_hist, att_hist: Trajectory, sys_params:SystemParams):

    #state_hist[] contains attitude history i.e.

    num_steps = tau_hist.shape[0]
    I_inv = np.linalg.inv(sys_params.I)
    dt = bc.tf/num_steps

    #optimisation variables
    tau = cp.Variable((num_steps, 3))  # Control torques
    omega = cp.Variable((num_steps + 1, 3))  # Angular velocities
    epsilon = cp.Variable((num_steps + 1, 4))  # Quaternions

    # Define the objective function (e.g., minimize control effort with gradient influence)
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 0.0:
        normgrad = grad / grad_norm
    else:
        normgrad = np.zeros_like(grad)
    objective = cp.Minimize(cp.sum_squares(tau - (tau_hist - alpha * normgrad)))

    eps_hist = np.vstack([state.eps for state in att_hist.states])
    omega_hist = np.vstack([state.omega for state in att_hist.states])

    theta_hist = [Theta(e) for e in eps_hist]
    omegan_hist = [Omega(o) for o in omega_hist]

    eps_dot_hist = [0.5*omegan_hist[i]@eps_hist[i] for i in range(0,len(omegan_hist))]
    omega_dot_hist = [-I_inv@skewer(o)@sys_params.I@o for o in omega_hist]
    euler_lin_hist = [-I_inv@skewer(o)@sys_params.I-skewer(sys_params.I@o) for o in omega_hist]

    constraints = []

    # Define the constraints
    for k in range(num_steps):

        constraints += [
            epsilon[k+1] == epsilon[k]+dt*(eps_dot_hist[k] + 0.5*omegan_hist[k]@(epsilon[k]-eps_hist[k])+0.5*theta_hist[k]@(omega[k]-omega_hist[k])),
            omega[k+1] == omega[k]+dt*(omega_dot_hist[k] + euler_lin_hist[k]@(omega[k]-omega_hist[k])+I_inv@tau[k]),
        ]

    # Initial and final state constraints
    constraints += [
        epsilon[0] == bc.x0.eps,
        omega[0] == bc.x0.omega,
        epsilon[-1] == bc.xf.eps,
        omega[-1] == bc.xf.omega
    ]

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=100000, verbose=False)

    # issue here is that if the optimisation fails, there is a none value returned
    # Handle solver failure
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"[tau_proj_lin] Solver failed with status: {prob.status}")
        # Return previous valid histories so loop can continue
        return tau_hist, att_hist

    # Retrieve the optimized values
    tau_opt = tau.value
    omega_opt = omega.value
    epsilon_opt = epsilon.value

    # Update the histories
    att_states_updated = [StateVector(r=None,v=None,eps=epsilon_opt[k].copy(),omega=omega_opt[k].copy()) for k in range(num_steps + 1)
    ]

    att_traj_updated = Trajectory(states=att_states_updated, times=att_hist.times.copy())

    return tau_opt, att_traj_updated

#custom tau projector with the data structures given
def tau_proj_nonlin_clean(alpha: float,grad: np.ndarray,tau_hist: np.ndarray,sys_params: SystemParams,bc: BoundaryConditions):
    N = tau_hist.shape[0]
    dt = bc.tf / N
    I = sys_params.I
    I_inv = np.linalg.inv(I)

    tau_target = tau_hist - alpha * grad

    tau_var = cp.Variable((N, 3))

    def final_attitude_constraint(tau_vec_flat):
        tau_vec = tau_vec_flat.reshape((N, 3))

        eps = bc.x0.eps.copy()
        omega = bc.x0.omega.copy()

        for k in range(N):
            tau_k = tau_vec[k]
            eps_dot = 0.5 * Omega(omega) @ eps
            omega_dot = -I_inv @ skewer(omega) @ I @ omega + I_inv @ tau_k
            eps = eps + dt * eps_dot
            eps = eps / np.linalg.norm(eps)  # normalise quaternion
            omega = omega + dt * omega_dot
        return np.concatenate([eps - bc.xf.eps, omega - bc.xf.omega])

    objective = cp.Minimize(cp.sum_squares(tau_var - tau_target))

    cons = {'type': 'eq', 'fun': final_attitude_constraint}

    sol = minimize(lambda t: np.sum((t.reshape((N, 3)) - tau_target) ** 2),x0=tau_hist.flatten(),        method='trust-constr', constraints=[cons],options={'maxiter': 200, 'verbose': 0})

    tau_opt = sol.x.reshape((N, 3))
    eps = bc.x0.eps.copy()
    omega = bc.x0.omega.copy()

    att_hist = []
    att_hist.append(StateVector(None, None, eps.copy(), omega.copy()))

    for k in range(N):
        tau_k = tau_opt[k]
        eps_dot = 0.5 * Omega(omega) @ eps
        omega_dot = -I_inv @ skewer(omega) @ I @ omega + I_inv @ tau_k

        eps = eps + dt * eps_dot
        eps /= np.linalg.norm(eps)
        omega = omega + dt * omega_dot

        att_hist.append(StateVector(None, None, eps.copy(), omega.copy()))

    return tau_opt, att_hist

#custom function to use the data structures made
def build_initial_Data(bc: BoundaryConditions, num_steps)->Trajectory:
    times = np.linspace(0, bc.tf, num_steps+1)

    states = [StateVector(r=None, v=None, eps=bc.x0.eps, omega=bc.x0.omega) for _ in range(num_steps + 1)]

    return Trajectory(states, times)

#custom function to use the data strcutres made
def gradient_descent_loop_clean(ctrl_hist: ControlHistory,sys_params: SystemParams,bc: BoundaryConditions,alpha,num_iterations,use_linear_proj=True):
    tau_hist = ctrl_hist.tau.copy()  # shape (N,3)
    cost_history = []
    att_hist = None

    for it in range(num_iterations):
        # conv opt
        rot_sv, U_list, grad, qs, cost = opt_given_tau_clean(ctrl_hist=ControlHistory(U_list, None, tau_hist),sys_params=sys_params,bc=bc,grad_needed=True)
        cost_history.append(cost)

        if use_linear_proj:
            tau_hist, att_hist = tau_projection_linear_clean(alpha=alpha,grad=grad, att_hist=att_hist, bc=bc,sys_params=sys_params,tau_hist=tau_hist )
        else:
            tau_hist, att_hist = tau_proj_nonlin_clean(alpha=alpha,grad=grad,bc=bc,sys_params=sys_params,tau_hist=tau_hist)
        ctrl_hist.tau = tau_hist

    # final optimisation to return last state
    rot_sv, U_list, grad, qs, cost = opt_given_tau_clean(ctrl_hist=ctrl_hist,sys_params=sys_params, bc=bc, grad_needed=False)

    return tau_hist, rot_sv, U_list, qs, cost_history


