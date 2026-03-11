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
import os
import sys


def opt_given_tau(tau, sys_params: SystemParams, state_vec: StateVector, conditions: BoundaryConditions, grad_needed=True):
    #: notes for later: repurpose this to give out only the cost and that can be used as a fitness function for the PIM

    mu = sys_params.mu
    a=sys_params.a
    e=sys_params.e
    nu=sys_params.nu
    I=sys_params.I
    m=sys_params.m
    rs=sys_params.rs

    x0 = conditions.x0.as_array()
    xf = conditions.xf.as_array()
    tf=conditions.tf

    r=state_vec.r
    v=state_vec.v

    delta = 1e-9
    num_steps = len(tau)
    dt = tf / num_steps

    # Define Parameters
    tau_p = cp.Parameter(tau.shape, value=tau)

    # Propagate Attitude and q vectors

    ome = [x0[10:13] for _ in range(num_steps+1)]
    eps = [x0[6:10] for _ in range(num_steps+1)]

    for k in range(num_steps):

        # Runge-Kutta 1st Order Method
        eps_dot_1, ome_dot_1 = another_state_derivative_og(eps[k], ome[k], tau_p[k].value,I)

        # TODO: RETHINK THE RK4 WHEN YOU HAVE TIME HERE

        #quaterion euler update
        eps[k+1] = eps[k] + dt*eps_dot_1
        eps[k+1] = eps[k+1] / np.linalg.norm(eps[k+1])

        #anuglare velocity euler update
        ome[k+1] = ome[k] + dt*ome_dot_1

    qs = [np.zeros((num_steps, 3)) for _ in range(len(rs))] #cuz rs rn is 4 agents, implies qs will be num_stepsx3x4 matrix

    for i in range(len(qs)):
        qs[i][0] = rs[i]
        for k in range(num_steps-1):
            qs[i][k+1] = qs[i][k] + dt * skewer(ome[k]) @ qs[i][k] #again euler update method for the position. omega*qs = translational velocity effect*dt = change in position due to rotation update
            qs[i][k+1] = np.linalg.norm(rs[i])*qs[i][k+1] / np.linalg.norm(qs[i][k+1]) #normalised to original length

    # Define Variables
    U = [cp.Variable((num_steps, 3)) for _ in range(len(qs))] #thruster forces this should be num_Stepsx3x4 matrix
    force = cp.Variable((num_steps, 3)) #net force experiences by the payload?

    r = cp.Variable((num_steps + 1, 3))                  # Position of payload
    v = cp.Variable((num_steps + 1, 3))                  # Velocity of payload

    # Constraints list
    constraints = []

    # Initial and final conditions
    constraints.append(r[0] == x0[:3])
    constraints.append(v[0] == x0[3:6])
    constraints.append(r[-1] == xf[:3])
    constraints.append(v[-1] == xf[3:6])

    objective = cp.Minimize(sum([cp.sum_squares(U[i]) for i in range(len(U))])) #min cost

    for k in range(num_steps):
        constraints.append(force[k] == cp.sum([U[i][k] for i in range(len(U))])) #add all thrusters
        constraints.append(tau_p[k] == cp.sum([skewer(qs[i][k]) @ U[i][k] for i in range(len(U))])) #add all torque data to history

        # Position and velocity update using linearized dynamics
        f, f_dot, f_ddot = orbital_params(mu, a, e, k * dt)
        A_pos = np.array([ #using the equation only for the rotational parameters
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        A_vel = np.array([
            [3 * f_dot**2, 0, 0, 1, 2 * f_dot, 0],
            [0, 0, 0, -2 * f_dot, 1, 0],
            [0, 0, -f_ddot, 0, 0, 1]
        ])
        B_vel = np.array([
            [1/m, 0, 0],
            [0, 1/m, 0],
            [0, 0, 1/m]
        ])

        # Dynamics constraints for position and velocity
        constraints.append(r[k + 1] == r[k]+dt*A_pos@cp.hstack([r[k],v[k]])) #hcw equations
        constraints.append(v[k + 1] == v[k]+dt*A_vel@cp.hstack([r[k],v[k]]) + dt * (B_vel @ force[k])) # hcw equaations?

    prob = cp.Problem(objective, constraints) #solving the optimisation problem

    if grad_needed:
        prob.solve(solver=cp.SCS, requires_grad=grad_needed, max_iters=100000)
        prob.backward()
        print("Value w/ grad:" + str(prob.value))
        print(prob.status)
        prob.solve(solver=cp.SCS, requires_grad=False, max_iters=100000)
        print("Value w/o grad:" + str(prob.value))
        print(prob.status)
        return np.hstack([r.value, v.value, eps, ome]), [U[i].value for i in range(len(U))], tau_p.gradient, qs, prob.value
    else:
        prob.solve(solver=cp.SCS, requires_grad=grad_needed, max_iters=10000000, verbose=True)
        return np.hstack([r.value, v.value, eps, ome]), [U[i].value for i in range(len(U))], qs, prob.value


def tau_proj_lin(alpha, grad, a0, af, tf, tau_hist, att_hist, I):
    #aim of the functoin here is to update torque history
    # Number of time steps
    num_steps = len(tau_hist)
    I_inv = np.linalg.inv(I) #replace
    dt = tf / num_steps
    delta = 1e-2

    # Define the optimization variables
    tau = cp.Variable((num_steps, 3))         # Control torques
    omega = cp.Variable((num_steps+1, 3))       # Angular velocities
    epsilon = cp.Variable((num_steps+1, 4))     # Quaternions

    # Define the parameters
    a0_param = cp.Parameter(7, value=a0)
    af_param = cp.Parameter(7, value=af)
    tf_param = cp.Parameter(value=tf)
    J_param = cp.Parameter((3, 3), value=I)

    # Define the objective function (e.g., minimize control effort with gradient influence)
    normgrad = grad/np.linalg.norm(grad)
    objective = cp.Minimize(cp.sum_squares(tau-(tau_hist-alpha*normgrad)))

    epsilon_hist = att_hist[:,0:4]
    omega_hist = att_hist[:,4:7]

    Theta_hist = [Theta(e) for e in epsilon_hist]
    Omega_hist = [Omega(o) for o in omega_hist]

    epsilon_dot_hist = [0.5*Omega_hist[i]@epsilon_hist[i] for i in range(0,len(Omega_hist))]
    omega_dot_hist = [-I_inv@skewer(o)@I@o for o in omega_hist]
    euler_lin_hist = [-I_inv@(skewer(o)@I-skewer(I@o)) for o in omega_hist]

    constraints = []

    # Define the constraints
    for k in range(num_steps):

        constraints += [
            epsilon[k+1] == epsilon[k]+dt*(epsilon_dot_hist[k] + 0.5*Omega_hist[k]@(epsilon[k]-epsilon_hist[k])+0.5*Theta_hist[k]@(omega[k]-omega_hist[k])),
            omega[k+1] == omega[k]+dt*(omega_dot_hist[k] + euler_lin_hist[k]@(omega[k]-omega_hist[k])+I_inv@tau[k]),
        ]

    # Initial and final state constraints
    constraints += [
        epsilon[0] == a0[0:4],
        omega[0] == a0[4:7],
        epsilon[-1] == af[0:4],
        omega[-1] == af[4:7]
    ]

    # Solve the optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=False)

    #issue here is that if the optimisation fails, there is a none value returned
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
    tau_hist_updated = tau_opt
    omega_hist_updated = omega_opt
    epsilon_hist_updated = epsilon_opt

    return tau_hist_updated, np.hstack([epsilon_hist_updated,omega_hist_updated])

def seq_conv_opt(x0, xf, tf, x_hist, mu, a, e, nu, I, m, rs, bc: BoundaryConditions, sys_params: SystemParams,  num_iterations=10):
    delta = 1e-9
    num_steps = len(x_hist) - 1
    dt = tf / num_steps

    I_inv = np.linalg.inv(sys_params.I)

    for iteration in range(num_iterations):
        # Define CVXPY Variables
        U = [cp.Variable((num_steps, 3)) for _ in range(6)]  # Control forces from thrusters
        qs = [cp.Variable((num_steps+1, 3)) for _ in range(6)]
        force = cp.Variable((num_steps, 3))
        tau = cp.Variable((num_steps, 3))                    # Control torques
        omega = cp.Variable((num_steps + 1, 3))              # Angular velocities
        epsilon = cp.Variable((num_steps + 1, 4))            # Quaternions
        r = cp.Variable((num_steps + 1, 3))                  # Position
        v = cp.Variable((num_steps + 1, 3))                  # Velocity

        # Objective function to minimize the total thrust effort
        objective = cp.Minimize(sum(cp.sum_squares(U[i]) for i in range(6)))

        # Constraints list
        constraints = []

        # Initial and final conditions
        constraints.append(r[0] == bc.x0.r.copy())
        constraints.append(v[0] == bc.x0.v.copy())
        constraints.append(epsilon[0] == bc.x0.eps.copy())
        constraints.append(omega[0] == bc.x0.omega.copy())

        constraints.append(r[-1] == xf[:3])
        constraints.append(v[-1] == xf[3:6])
        constraints.append(epsilon[-1] == xf[6:10])
        constraints.append(omega[-1] == xf[10:13])
        for i in range(len(qs)):
            constraints.append(qs[i][0] == rs[i])

        # Populate constraints based on dynamics
        for k in range(num_steps):
            # Position and velocity update using linearized dynamics
            f, f_dot, f_ddot = orbital_params(mu, a, e, k * dt)
            A_pos = np.array([
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt]
            ])
            A_vel = np.array([
                [3 * f_dot**2, 0, 0, 1, 2 * f_dot, 0],
                [0, 0, 0, -2 * f_dot, 1, 0],
                [0, 0, -f_ddot, 0, 0, 1]
            ])
            B_vel = np.array([
                [1/m, 0, 0],
                [0, 1/m, 0],
                [0, 0, 1/m]
            ])

            # Dynamics constraints for position and velocity
            constraints.append(r[k + 1] == A_pos@cp.hstack([r[k],v[k]]))
            constraints.append(v[k + 1] == A_vel@cp.hstack([r[k],v[k]]) + dt * (B_vel @ force[k]))

            # Linearized attitude dynamics around previous iteration
            epsilon_hist = x_hist[:, 6:10]
            omega_hist = x_hist[:, 10:13]

            Theta_hist = [Theta(e) for e in epsilon_hist]
            Omega_hist = [Omega(o) for o in omega_hist]

            epsilon_dot_hist = [0.5 * Omega_hist[i] @ epsilon_hist[i] for i in range(len(Omega_hist))]
            omega_dot_hist = [-I_inv @ skewer(o) @ I @ o for o in omega_hist]
            euler_lin_hist = [-I_inv @ (skewer(o) @ I - skewer(I @ o)) for o in omega_hist]

            # Quaternion dynamics using linearized dynamics
            constraints.append(epsilon[k + 1] == epsilon[k] + dt * (epsilon_dot_hist[k] + 0.5 * Omega_hist[k] @ (epsilon[k] - epsilon_hist[k]) + 0.5 * Theta_hist[k] @ (omega[k] - omega_hist[k])))

            # Angular velocity dynamics using linearized dynamics
            constraints.append(omega[k + 1] == omega[k] + dt * (omega_dot_hist[k] + euler_lin_hist[k] @ (omega[k] - omega_hist[k]) + I_inv @ tau[k]))

            # Thruster constraints
            constraints.append(force[k] == cp.sum([U[i][k] for i in range(6)], axis=0))
            constraints.append(tau[k] == cp.sum([skewer(rs[i]) @ U[i][k] for i in range(6)], axis=0))


        for i in range(len(U)):
            for k in range(num_steps):
                #constraints.append(U[i][k]@qs[i][k]+2*np.cos(nu)*delta-np.cos(nu)*(cp.norm(U[i][k])*cp.norm(qs[i][k])+delta) >= 0)
                #Attachment vector dynamics
                constraints.append(qs[i][k+1] == skewer(omega_hist[k])@qs[i][k])

        constraints.append(cp.norm(cp.hstack([r,v,epsilon,omega])-x_hist) <= delta)

        # Define the optimization problem
        prob = cp.Problem(objective, constraints)

        # Solve the convex subproblem
        prob.solve(verbose=True)

        # Update x_hist for the next iteration using the current solution
        x_hist[:, :3] = r.value
        x_hist[:, 3:6] = v.value
        x_hist[:, 6:10] = epsilon.value
        x_hist[:, 10:13] = omega.value

    return r.value, v.value, epsilon.value, omega.value, qs.value, [U[i].value for i in range(6)], tau.value

def full_nlp(x0, xf, tf, mu, a, e, nu, I, m, rs, N, max_iters=100):
    num_steps = N  # Assuming N is defined
    num_agents = len(rs)
    dt = tf / num_steps
    delta = 1e-6
    U_guess = np.zeros((len(rs), N, 3))
    # Initial omega and epsilon propagation

    def objective(U_flat):
        U = U_flat.reshape((num_agents, num_steps, 3))

        # Example cost function: sum of norms of U
        loss = np.sum(np.linalg.norm(U, axis=1)**2)

        return loss

    # Define dynamics constraint functions
    def constraints(U_flat):
        U = U_flat.reshape((num_agents, num_steps, 3))
        Q = np.zeros((num_agents,num_steps+1,3))
        r = np.zeros((num_steps+1,3))
        v = np.zeros((num_steps+1,3))
        eps = np.zeros((num_steps+1,4))
        ome = np.zeros((num_steps+1,3))
        for i in range(len(rs)):
            Q[i][0] = rs[i]
        r[0] = x0[0:3]
        v[0] = x0[3:6]
        eps[0] = x0[6:10]
        ome[0] = x0[10:13]

        constraints = []

        for k in range(num_steps):
            torque_curr = np.sum([skewer(Q[i][k])@U[i][k] for i in range(0,len(rs))], axis=0)
            force_curr = np.sum([U[i][k] for i in range(0,len(rs))],axis=0)

            # Position and velocity update using linearized dynamics
            f, f_dot, f_ddot = orbital_params(mu, a, e, k * dt)
            A_pos = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            A_vel = np.array([
                [3 * f_dot**2, 0, 0, 1, 2 * f_dot, 0],
                [0, 0, 0, -2 * f_dot, 1, 0],
                [0, 0, -f_ddot, 0, 0, 1]
            ])
            B_vel = np.array([
                [1/m, 0, 0],
                [0, 1/m, 0],
                [0, 0, 1/m]
            ])

            phi = Phi(ome[k], I)

            # Dynamics constraints for position and velocity
            r[k+1] = r[k]+dt*A_pos@np.hstack([r[k],v[k]])
            v[k+1] = v[k] + dt*A_vel@np.hstack([r[k],v[k]]) + dt * (B_vel @ force_curr)
            eps[k+1] = eps[k]+(dt*phi@np.hstack([eps[k],ome[k]]))[0:4]
            ome[k+1] = ome[k]+(dt*phi@np.hstack([eps[k],ome[k]]))[4:7] + dt*(np.linalg.inv(I)@torque_curr)
            for i in range(0,num_agents):
                Q[i][k+1] = Q[i][k] + dt * skewer(ome[k]) @ Q[i][k]
                Q[i][k+1] = np.linalg.norm(rs[i])*Q[i][k+1] / np.linalg.norm(Q[i][k+1])

        constraints.append(r[num_steps] - xf[0:3])
        constraints.append(v[num_steps] - xf[3:6])
        constraints.append(eps[num_steps] - xf[6:10])
        constraints.append(ome[num_steps] - xf[10:13])

        return np.concatenate(constraints)

        # Define ineq constraint functions
    def constraints_ineq(U_flat):
        U = U_flat.reshape((num_agents, num_steps, 3))
        Q = np.zeros((num_agents,num_steps+1,3))
        r = np.zeros((num_steps+1,3))
        v = np.zeros((num_steps+1,3))
        eps = np.zeros((num_steps+1,4))
        ome = np.zeros((num_steps+1,3))
        for i in range(len(rs)):
            Q[i][0] = rs[i]
        r[0] = x0[0:3]
        v[0] = x0[3:6]
        eps[0] = x0[6:10]
        ome[0] = x0[10:13]

        constraints_ineq = []

        for k in range(num_steps):
            torque_curr = np.sum([skewer(Q[i][k])@U[i][k] for i in range(0,len(rs))], axis=0)
            force_curr = np.sum([U[i][k] for i in range(0,len(rs))],axis=0)

            # Position and velocity update using linearized dynamics
            f, f_dot, f_ddot = orbital_params(mu, a, e, k * dt)
            A_pos = np.array([
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            A_vel = np.array([
                [3 * f_dot**2, 0, 0, 1, 2 * f_dot, 0],
                [0, 0, 0, -2 * f_dot, 1, 0],
                [0, 0, -f_ddot, 0, 0, 1]
            ])
            B_vel = np.array([
                [1/m, 0, 0],
                [0, 1/m, 0],
                [0, 0, 1/m]
            ])

            phi = Phi(ome[k], I)

            # Dynamics constraints for position and velocity
            r[k+1] = r[k]+dt*A_pos@np.hstack([r[k],v[k]])
            v[k+1] = v[k] + dt*A_vel@np.hstack([r[k],v[k]]) + dt * (B_vel @ force_curr)
            eps[k+1] = eps[k]+(dt*phi@np.hstack([eps[k],ome[k]]))[0:4]
            ome[k+1] = ome[k]+(dt*phi@np.hstack([eps[k],ome[k]]))[4:7] + dt*(np.linalg.inv(I)@torque_curr)
            for i in range(0,num_agents):
                Q[i][k+1] = Q[i][k] + dt * skewer(ome[k]) @ Q[i][k]
                Q[i][k+1] = np.linalg.norm(rs[i])*Q[i][k+1] / np.linalg.norm(Q[i][k+1])
                constraints_ineq.append(U[i][k] @ Q[i][k] + 2 * np.cos(nu) * delta - np.cos(nu) * (np.linalg.norm(U[i][k]) * np.linalg.norm(Q[i][k]) + delta))

        return constraints_ineq

    # Initial guess for U
    U0 = U_guess
    U0_flat = U0.flatten()

    # Define the constraints dictionary for scipy minimize
    cons = {'type': 'eq', 'fun': constraints}
    cons_ineq = {'type': 'ineq', 'fun': constraints_ineq}

    # Define equality constraint
    eq_constraint = NonlinearConstraint(constraints, 0, 0)  # lb=ub=0 for equality

    ineq_constraint = NonlinearConstraint(constraints_ineq, 0, np.inf)  # lb=0, ub=np.inf for inequality

    # Solve the optimization problem
    #result = minimize(objective, U0_flat, method='SLSQP', constraints=[cons, cons_ineq],
    #                  options={'maxiter': max_iters, 'verbosity': 2})
    result = minimize(objective, U0_flat, method='trust-constr', constraints=[eq_constraint],
                      options={'maxiter': max_iters, 'verbose': 2})
    U_opt = result.x.reshape((num_agents, num_steps, 3))

    r_fin = np.zeros((num_steps+1,3))
    v_fin = np.zeros((num_steps+1,3))
    eps_fin = np.zeros((num_steps+1,4))
    ome_fin = np.zeros((num_steps+1,3))
    Q_fin = np.zeros((num_agents,num_steps+1,3))
    for i in range(len(rs)):
        Q_fin[i][0] = rs[i]
    r_fin[0] = x0[0:3]
    v_fin[0] = x0[3:6]
    eps_fin[0] = x0[6:10]
    ome_fin[0] = x0[10:13]

    for k in range(num_steps):
        torque_curr = np.sum([skewer(Q_fin[i][k])@U_opt[i][k] for i in range(0,len(rs))], axis=0)
        force_curr = np.sum([U_opt[i][k] for i in range(0,len(rs))],axis=0)

        # Position and velocity update using linearized dynamics
        f, f_dot, f_ddot = orbital_params(mu, a, e, k * dt)
        A_pos = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        A_vel = np.array([
            [3 * f_dot**2, 0, 0, 1, 2 * f_dot, 0],
            [0, 0, 0, -2 * f_dot, 1, 0],
            [0, 0, -f_ddot, 0, 0, 1]
        ])
        B_vel = np.array([
            [1/m, 0, 0],
            [0, 1/m, 0],
            [0, 0, 1/m]
        ])

        phi = Phi(ome_fin[k], I)

        # Dynamics constraints for position and velocity
        r_fin[k+1] = r_fin[k]+dt*A_pos@np.hstack([r_fin[k],v_fin[k]])
        v_fin[k+1] = v_fin[k] + dt*A_vel@np.hstack([r_fin[k],v_fin[k]]) + dt * (B_vel @ force_curr)
        eps_fin[k+1] = eps_fin[k]+(dt*phi@np.hstack([eps_fin[k],ome_fin[k]]))[0:4]
        ome_fin[k+1] = ome_fin[k]+(dt*phi@np.hstack([eps_fin[k],ome_fin[k]]))[4:7] + dt*(np.linalg.inv(I)@torque_curr)
        for i in range(0,num_agents):
            Q_fin[i][k+1] = Q_fin[i][k] + dt * skewer(ome_fin[k]) @ Q_fin[i][k]
            Q_fin[i][k+1] = np.linalg.norm(rs[i])*Q_fin[i][k+1] / np.linalg.norm(Q_fin[i][k+1])

    X_opt = np.hstack([r_fin,v_fin,eps_fin,ome_fin])

    # Retrieve optimized U and reshape
    return U_opt, X_opt, Q_fin

def x_init(x0, xf, tf, I, mu, a, e, m, num_steps=100):
    dt = tf / num_steps

    def objective(con):
        con = con.reshape((num_steps, 6))
        # Minimize the distance to zero (L2 norm)
        return np.sum(np.abs(con))

    def constraints(con):
        state = np.zeros((num_steps + 1, 13))
        con = con.reshape((num_steps, 6))
        state[0] = x0

        constr_list = []
        constr_list.append(state[0] - x0)  # Initial state should match x0

        for k in range(num_steps):
            omega = state[k, 10:13]
            # Using ZOH matrices for discrete dynamics
            # NOTE: There is a change made ehre, might trigger issue
            [A_d, B_d] = discretize_matrices_zoh(dt, k * dt, mu, a, e, I, omega, m)

            state[k+1] = A_d @ state[k] + B_d @ con[k]

        constr_list.append(state[num_steps] - xf)

        return np.concatenate(constr_list)

    # Initial guess for state vector x (start close to zero)
    initial_guess = np.zeros((num_steps, 6)).flatten()

    # Define the constraints dictionary for scipy minimize
    cons = {'type': 'eq', 'fun': constraints}

    # Solve the optimization problem
    result = minimize(objective, initial_guess, method='trust-constr', constraints=[cons],options={'maxiter': 5, 'verbose': 2})

    if not result.success:
        print("Optimization did not converge:", result.message)

    state_opt = np.zeros((num_steps + 1, 13))
    state_opt[0] = x0
    con_opt = result.x.reshape((num_steps,6))

    for k in range(num_steps):
        omega = state_opt[k, 10:13]
        # Using ZOH matrices for discrete dynamics
        #NOTE: There is a change made ehre, might trigger issue
        [A_d, B_d] = discretize_matrices_zoh(dt, k * dt, mu, a, e, I, omega, m)
        state_opt[k+1] = A_d @ state_opt[k] + B_d @ con_opt[k]

    return state_opt

def gradient_descent_loop(alpha, num_iterations, x0, xf, tf, tau_hist, mu, a, e, nu, I, m, att_hist = None):

    cost_history = []  # List to store the cost at each iteration

    if att_hist is not None:
      att_hist_updated = att_hist

    rel_tol = 1e-8

    # Outer progress bar for the full function
    with tqdm(total=num_iterations, desc="Total Progress", unit="iter") as outer_pbar:
        for i in range(num_iterations):
                # Step 1: Solve for the state and control history given the current tau
                start_time = time.time()
                X_opt, U_opt, grad, qs_opt, cost = opt_given_tau(tau_hist, x0, xf, tf, mu, a, e, nu, I, m)
                opt_given_tau_time = time.time() - start_time
                print(f"opt_given_tau took: {opt_given_tau_time:.2f} seconds")

                print("Iter Cost:" + str(cost))
                print("Grad Norm:" + str(np.linalg.norm(grad)))
                print("X_opt NAN?:" + str(np.isnan(X_opt).any()))

                # Store the cost for this iteration
                cost_history.append(cost)

                # Step 2: Project the gradient to update tau
                start_time = time.time()
                if att_hist is not None:
                  tau_hist, att_hist = tau_proj_lin(alpha, grad, x0[6:13], xf[6:13], tf, tau_hist, att_hist, I)
                  print("After tau_proj_lin:", type(tau_hist), tau_hist is None)
                else:
                  tau_hist, att_hist = tau_proj_nonlin(alpha, grad, x0[6:13], xf[6:13], tf, tau_hist, I)
                  print("After tau_proj_nonlin:", type(tau_hist), tau_hist is None)

                tau_proj_time = time.time() - start_time
                print(f"tau_proj took: {tau_proj_time:.2f} seconds")

                # Optionally: Plot results after each iteration
                if (i + 1) % 10 == 0 or i == num_iterations - 1:  # Plot every 10 iterations and the last one
                    plot_att_vs_time(X_opt.T[6:13].T, tf)
                    plot_pos_vel_vs_time(X_opt.T[0:6].T, tf)
                    plot_control_forces_vs_time(U_opt, tf)
                    plot_cost_vs_iteration(cost_history)
                    plt.show()

                if len(cost_history) > 1 and np.abs(cost_history[-1]-cost_history[-2])/np.abs(cost_history[-1]) < rel_tol:
                    return tau_hist, X_opt, U_opt, qs_opt, cost_history

                # Update outer progress bar
                outer_pbar.update(1)

    return tau_hist, X_opt, U_opt, qs_opt, cost_history

def momentum_gradient_descent_ema(alpha, beta, num_iterations, x0, xf, tf, tau_hist, mu, a, e, nu, I, m):
    """
    Gradient descent with momentum using an Exponential Moving Average (EMA).

    Parameters:#
    alpha (float): Learning rate.
    beta (float): Momentum constant (0 < beta < 1).
    num_iterations (int): Number of iterations to run.
    x0 (np.ndarray): Initial state vector.
    xf (np.ndarray): Final state vector.
    tf (float): Final time.
    tau_hist (np.ndarray): Initial guess for the control history.
    mu, a, e, nu, I, m: Various physical parameters for the optimization problem.

    Returns:
    tuple: Optimized control history, state trajectory, control inputs, auxiliary variables, and cost history.
    """
    cost_history = []  # List to store the cost at each iteration
    momentum = np.zeros_like(tau_hist)  # Initialize momentum term to zeros

    rel_tol = 1e-8

    # Outer progress bar for the full function
    with tqdm(total=num_iterations, desc="Total Progress", unit="iter") as outer_pbar:
        for i in range(num_iterations):
            # Step 1: Solve for the state and control history given the current tau
            start_time = time.time()
            X_opt, U_opt, grad, qs_opt, cost = opt_given_tau(tau_hist, x0, xf, tf, mu, a, e, nu, I, m)
            opt_given_tau_time = time.time() - start_time
            print(f"opt_given_tau took: {opt_given_tau_time:.2f} seconds")

            print("Iter Cost:" + str(cost))
            print("Grad Norm:" + str(np.linalg.norm(grad)))
            print("X_opt NAN?:" + str(np.isnan(X_opt).any()))

            # Store the cost for this iteration
            cost_history.append(cost)

            # Step 2: Update tau using momentum-based gradient descent with EMA
            momentum = beta * momentum + (1 - beta) * grad  # EMA of gradients
            tau_hist = tau_hist - alpha * momentum  # Update tau_hist using EMA of gradients

            # Apply projection using nonlinear projection method
            tau_hist, att_hist_updated = tau_proj_nonlin(alpha, grad, x0[6:13], xf[6:13], tf, tau_hist, I)
            print(f"tau_proj_nonlin took: {time.time() - start_time:.2f} seconds")

            # Optionally: Plot results after each iteration
            if (i + 1) % 10 == 0 or i == num_iterations - 1:  # Plot every 10 iterations and the last one
                plot_att_vs_time(X_opt.T[6:13].T, tf)
                plot_pos_vel_vs_time(X_opt.T[0:6].T, tf)
                plot_control_forces_vs_time(U_opt, tf)
                plot_cost_vs_iteration(cost_history)
                plt.show()

            # Check for convergence
            if len(cost_history) > 1 and np.abs(cost_history[-1] - cost_history[-2]) / np.abs(cost_history[-1]) < rel_tol:
                return tau_hist, X_opt, U_opt, qs_opt, cost_history

            # Update outer progress bar
            outer_pbar.update(1)

    return tau_hist, X_opt, U_opt, qs_opt, cost_history


def mimd_descent_loop(alpha_init, x0, xf, tf, tau_hist, mu, a, e, nu, I, m, att_hist=None, num_iterations=100):
    X_opt, U_opt, grad, qs_opt, cost = opt_given_tau(tau_hist, x0, xf, tf, mu, a, e, nu, I, m)
    tau_hist_prev = tau_hist
    att_hist_prev = att_hist

    cost_history = [cost]
    minima = []

    if att_hist is not None:
        att_hist_updated = att_hist

    NEAR_MIN = True
    alpha = alpha_init
    alpha_upper_lim = 1e0
    alpha_lower_lim = 1e-8
    i = 1

    # Outer progress bar for the full function
    with tqdm(total=num_iterations, desc="Total Progress", unit="iter") as outer_pbar:
        while i <= num_iterations:
            if NEAR_MIN:
                print("Near Minimum, searching...")
                while alpha > alpha_lower_lim and i <= num_iterations:
                    i+=1
                    outer_pbar.update(1)
                    start_time = time.time()
                    grad_prev = grad
                    X_opt, U_opt, grad, qs_opt, cost = opt_given_tau(tau_hist, x0, xf, tf, mu, a, e, nu, I, m)
                    opt_given_tau_time = time.time() - start_time
                    print(f"opt_given_tau took: {opt_given_tau_time:.2f} seconds")

                    # Store the cost for this iteration
                    if cost >= cost_history[-1]:
                        alpha = 0.25 * alpha
                        grad = grad_prev
                        tau_hist = tau_hist_prev
                        att_hist_updated = att_hist_prev
                    elif cost < cost_history[-1] and alpha < alpha_upper_lim:
                        alpha = 2*alpha
                        cost_history.append(cost)
                    else:
                        cost_history.append(cost)

                    print("Iter Cost:" + str(cost))
                    print("Alpha:" + str(alpha))
                    print("Grad Norm:" + str(np.linalg.norm(grad)))

                    # Step 2: Project the gradient to update tau
                    start_time = time.time()
                    if att_hist is not None:
                        tau_hist_prev, att_hist_prev = tau_hist, att_hist_updated
                        tau_hist, att_hist_updated = tau_proj_lin(alpha, grad, x0[6:13], xf[6:13], tf, tau_hist, att_hist_updated, I)
                    else:
                        tau_hist_prev, att_hist_prev = tau_hist, att_hist_updated
                        tau_hist, att_hist_updated = tau_proj_nonlin(alpha, grad, x0[6:13], xf[6:13], tf, tau_hist, I)

                    tau_proj_time = time.time() - start_time
                    print(f"tau_proj took: {tau_proj_time:.2f} seconds")

                    # Optionally: Plot results after each iteration
                    if (i + 1) % 10 == 0:  # Plot every 10 iterations and the last one
                        plot_att_vs_time(X_opt.T[6:13].T, tf)
                        plot_pos_vel_vs_time(X_opt.T[0:6].T, tf)
                        plot_control_forces_vs_time(U_opt, tf)
                        plot_cost_vs_iteration(cost_history)
                        plt.show()

                if att_hist is not None:
                    tau_hist_prev, att_hist_prev = tau_hist, att_hist_updated
                    tau_hist, att_hist_updated = tau_proj_lin(0, grad, x0[6:13], xf[6:13], tf, tau_hist, att_hist_updated, I)
                else:
                    tau_hist_prev, att_hist_prev = tau_hist, att_hist_updated
                    tau_hist, att_hist_updated = tau_proj_nonlin(0, grad, x0[6:13], xf[6:13], tf, tau_hist, I)


                minima.append([tau_hist, cost_history[-1]])
                print("Found Local Min:" + str(cost_history[-1]))
                NEAR_MIN = False

            if not NEAR_MIN:
                print("Searching for Next Min Region...")
                alpha = alpha_upper_lim
                while not NEAR_MIN and i < num_iterations:
                    i+=1
                    outer_pbar.update(1)
                    start_time = time.time()
                    X_opt, U_opt, grad, qs_opt, cost = opt_given_tau(tau_hist, x0, xf, tf, mu, a, e, nu, I, m)
                    opt_given_tau_time = time.time() - start_time
                    print(f"opt_given_tau took: {opt_given_tau_time:.2f} seconds")

                    print("Iter Cost:" + str(cost))
                    print("Alpha:" + str(alpha))
                    print("Grad Norm:" + str(np.linalg.norm(grad)))

                    if cost < cost_history[-1]:
                        NEAR_MIN = True
                        break  # don't want to take another maximum size step if near min

                    cost_history.append(cost)

                    # Step 2: Project the gradient to update tau
                    start_time = time.time()
                    if att_hist is not None:
                        tau_hist, att_hist_updated = tau_proj_lin(alpha, grad, x0[6:13], xf[6:13], tf, tau_hist, att_hist_updated, I)
                    else:
                        tau_hist, att_hist_updated = tau_proj_nonlin(alpha, grad, x0[6:13], xf[6:13], tf, tau_hist, I)

                    # Optionally: Plot results after each iteration
                    if (i + 1) % 10 == 0:  # Plot every 10 iterations and the last one
                        plot_att_vs_time(X_opt.T[6:13].T, tf)
                        plot_pos_vel_vs_time(X_opt.T[0:6].T, tf)
                        plot_control_forces_vs_time(U_opt, tf)
                        plot_cost_vs_iteration(cost_history)
                        plt.show()

    best_tau = [minimum[0] for minimum in minima][np.argmin([minimum[1] for minimum in minima])]
    return best_tau, X_opt, U_opt, qs_opt, cost_history, minima

def tau_proj_nonlin(tau_hist, a0,af, tf, N, I, epsilon, num_iter=None):
    num_steps = N  # Number of steps
    dt = tf / num_steps

    # Convert constants to CasADi types
    I_casadi = ca.DM(I)
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

                # Smoothness constraint: |tau[k+1] - tau[k]|^2 < epsilon
        #if k < num_steps-1:
        #    tau_diff = tau[k+1] - tau[k]
        #    smoothness_constraint = ca.sumsqr(tau_diff)  # (tau[k+1] - tau[k])^2
        #    constraints.append(smoothness_constraint)
        #    lbg.append(0)  # Lower bound for the smoothness constraint
        #    ubg.append(1)  # Upper bound for the smoothness constraint

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
                'print_level': 4,  # Solver verbosity level
            }
        }
    else:
        opts = {
            'ipopt': {
                'max_iter': num_iter,
                'print_level': 4,  # Solver verbosity level
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


def opt_given_tau_ipopt(tau, N, rs, tf, I, m, mu, a, e, epsilon, nu, x0, xf,  num_iter=None):
    num_steps = N
    num_agents = len(rs)
    dt = tf / num_steps

    U_guess = np.zeros((num_agents, num_steps, 3))

    # Convert constants to CasADi types
    I_casadi = ca.DM(I)
    m_casadi = ca.DM(m)
    dt_casadi = dt  # dt is already a scalar, so no need to convert
    epsilon_casadi = epsilon  # Assuming epsilon is a scalar
    nu_casadi = nu  # nu in radians

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
        get_r(0) - x0[0:3],
        get_v(0) - x0[3:6],
    ])
    constraints_lb.extend([0] * 3 + [0] * 3)
    constraints_ub.extend([0] * 3 + [0] * 3)

    ome = [x0[10:13] for _ in range(num_steps+1)]
    eps = [x0[6:10] for _ in range(num_steps+1)]

    for k in range(num_steps):
        rot_dot = rotational_derivative(np.hstack([eps[k], ome[k]]), tau[k],I)

        eps[k+1] = eps[k] + dt*rot_dot[0:4]
        eps[k+1] = eps[k+1] / np.linalg.norm(eps[k+1])

        ome[k+1] = ome[k] + dt*rot_dot[4:7]

    qs = [np.zeros((num_steps, 3)) for _ in range(len(rs))]

    for i in range(len(qs)):
        qs[i][0] = rs[i]
        for k in range(num_steps-1):
            qs[i][k+1] = qs[i][k] + dt * skewer(ome[k]) @ qs[i][k]
            qs[i][k+1] = np.linalg.norm(rs[i])*qs[i][k+1] / np.linalg.norm(qs[i][k+1])

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
        f, f_dot, f_ddot = orbital_params(mu, a, e, k * dt)

        A_vel = ca.SX.zeros(3, 6)
        A_vel[0, :] = ca.horzcat(3 * f_dot ** 2, 0, 0, 0, 2 * f_dot, 0)
        A_vel[1, :] = ca.horzcat(0, 0, 0, -2 * f_dot, 0, 0)
        A_vel[2, :] = ca.horzcat(0, 0, -f_ddot, 0, 0, 0)
        B_vel = (1 / m) * ca.SX_eye(3)

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
        get_r(num_steps) - xf[0:3],
        get_v(num_steps) - xf[3:6],
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
        np.tile(x0[0:3], num_steps + 1),
        np.tile(x0[3:6], num_steps + 1),
    ])

    # Define NLP problem
    nlp = {'x': opt_vars, 'f': cost, 'g': g}

    if num_iter is None:
        opts = {'ipopt': {'print_level': 5}}
    else:
        opts = {'ipopt': {'max_iter': num_iter, 'print_level': 5}}

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

    X_opt = np.hstack([r_opt, v_opt, eps_opt, ome_opt])

    return X_opt, U_opt, Q_opt, np.sum(np.square(U_opt))