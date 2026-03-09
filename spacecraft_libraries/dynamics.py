#NOTE TO SELF: This library has been modified to use the data structures class

import numpy as np
from scipy.linalg import expm
from .orbital_helpers import orbital_params
from .data_structures import StateVector, SystemParams
import casadi as ca

def skewer(v):
    """
    Returns the skew-symmetric matrix
    :param v: 3x1 np array vector
    :return: 3x3 np array skew-symmetric matrix
    """
    v=np.asarray(v)
    return np.array([[0,-v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def Theta(e):
    """
    used to return jacobian with respect to angular velocity
    :param e: orbital eccentricity
    :return: quaternion of what? idk
    """
    return np.array([
        [e[3], -e[2], e[1]],
        [e[2], e[3], -e[0]],
        [-e[1], e[0], e[3]],
        [-e[0], -e[1], -e[2]]
    ])

def Omega(o):
    """
    Quaternion multiplication
    :param o:
    :return:
    """
    return np.array([
        [0, o[2], -o[1], o[0]],
        [-o[2], 0, o[0], o[1]],
        [o[1], -o[0], 0, o[2]],
        [-o[0], -o[1], -o[2], 0]
    ])

def Phi(o,I):
    """
    Full state transition Jacobian
    :param o: angular velocity
    :param I: Inertia matrix
    :return: Jacobian
    """
    I_inv = np.linalg.inv(I)
    return np.block([[0.5*Omega(o),np.zeros((4,3))],[np.zeros((3,4)),-I_inv@skewer(o)@I]])

def zoh_Phi(dt, o, I):
    """
    Discrete version with zero order hold
    :param dt: time steps
    :param o: angular velocity
    :param I: inertia matrix
    :return: discrete Jacobian
    """
    phi = Phi(o, I)
    n = phi.shape[0]
    M = np.block([
        [phi, np.zeros((n, n))],
        [np.zeros((n, n)), np.zeros((n, n))]
    ])

    # Compute the matrix exponential
    M_d = expm(M * dt)

    # Extract the discrete-time Phi matrix
    Phi_d = M_d[:n, :n]

    return Phi_d

def A_matrix_3D(t, sys_params: SystemParams, sv: StateVector):
    """
    Makes the A matrix needed for each agent
    :param t: time steps
    :param sys_params: System Parameters
    :param sv: System State Vector
    :return: A matrix
    """
    f, f_dot_val, f_double_dot_val = orbital_params(sys_params.mu, sys_params.a, sys_params.e, t)
    I_inv = np.linalg.inv(sys_params.I)

    C = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [(f_dot_val**2 + 2 * sys_params.mu / sys_params.a**3), f_double_dot_val, 0, 0, 2 * f_dot_val, 0],
        [-f_double_dot_val, (f_dot_val**2 - sys_params.mu / sys_params.a**3), 0, -2 * f_dot_val, 0, 0],
        [0, 0, -sys_params.mu/sys_params.a**3, 0, 0, 0]
    ])

    #NOTE: assumption here is that o is angular velocity
    o_matrix = Omega(sv.omega)

    R_matrix = np.block([[0.5*o_matrix, np.zeros((4,3))],[np.zeros((3,4)),-I_inv@skewer(sv.omega)@sys_params.I]])

    A = np.block([
        [C, np.zeros((6, 7))],
        [np.zeros((7, 6)), R_matrix]
    ])
    return A

def B_matrix_3D(t, sys_params: SystemParams):
    """
    Makes the B matrix needed for each agent
    :param sys_params: System Parameters
    :param t: time steps
    :return: B matrix
    """
    zero_3x3 = np.zeros((3, 3))
    zero_4x3 = np.zeros((4, 3))
    identity_3x3 = np.eye(3)
    I_inv = np.linalg.inv(sys_params.I)

    # Construct the B matrix
    B = np.block([
        [zero_3x3, zero_3x3],
        [1/sys_params.m * identity_3x3, zero_3x3],
        [zero_4x3, zero_4x3],
        [zero_3x3, I_inv]
    ])

    return B

def discretize_matrices_zoh(dt, t, sys_params: SystemParams, sv: StateVector):
    """
    Returns both discrete versions of A and B matrices
    :param sv: StateVector
    :param sys_params: System Parameters
    :param dt: time steps
    :param t: time point
    :return: Discrete A and B matrices
    """
    A = A_matrix_3D(t, sys_params, sv)
    B = B_matrix_3D(t, sys_params)

    # Calculate the matrix exponential
    M = np.block([
        [A, B],
        [np.zeros((B.shape[1], A.shape[0] + B.shape[1]))]
    ])
    M_d = expm(M * dt)

    A_d = M_d[:A.shape[0], :A.shape[1]]
    B_d = M_d[:A.shape[0], A.shape[1]:]

    return A_d, B_d

def state_derivative(tau, sv: StateVector, sys_params: SystemParams):
    """
    Takes in the state
    :param sys_params: System Parameters
    :param sv: System State Vector
    :param tau: Control torque matrix
    :return:
    """
    state = np.hstack([sv.eps, sv.omega])
    phi = Phi(sv.omega, sys_params.I) # gets the discrete jacobian
    return phi @ state + np.vstack((np.zeros((4, 3)), np.linalg.inv(sys_params.I))) @ tau # computes an equation todo: write it down in the notes bit

    # Define the state derivative function

def another_state_derivative(tau,sv: StateVector, sys_params: SystemParams):
    """
    Takes in the
    :param sys_params: System Parameters
    :param sv: System State Vector
    :param tau: Torque Control Matrix
    :return: epsilon and omega derivatives
    """
    I_inv = np.linalg.inv(sys_params.I)
    eps_dot = 0.5 * Omega(sv.omega) @ sv.eps
    ome_dot = -I_inv @ skewer(sv.omega) @ sys_params.I @ sv.omega + I_inv @ tau
    return eps_dot, ome_dot

#original ones still here if needed
def state_derivative_og(state, tau, I):
    omega = state[4:]
    phi = Phi(omega, I)
    return phi @ state + np.vstack((np.zeros((4, 3)), np.linalg.inv(I))) @ tau

    # Define the state derivative function

def another_state_derivative_og(eps, ome, tau, I):
    I_inv = np.linalg.inv(I)
    eps_dot = 0.5 * Omega(ome) @ eps
    ome_dot = -I_inv @ skewer(ome) @ I @ ome + I_inv @ tau
    return eps_dot, ome_dot

def skew_casadi(vector):
    """Returns the skew-symmetric matrix of a vector using CasADi."""
    return ca.vertcat(
        ca.horzcat(0, -vector[2], vector[1]),
        ca.horzcat(vector[2], 0, -vector[0]),
        ca.horzcat(-vector[1], vector[0], 0)
    )

def quat_mult_casadi(q1, q2):
    """Quaternion multiplication using CasADi."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return ca.vertcat(w, x, y, z)

def quat_rotate_casadi(q, v):
    """Rotate vector v by quaternion q using CasADi."""
    q_conj = ca.vertcat(q[0], -q[1], -q[2], -q[3])
    v_quat = ca.vertcat(ca.SX(0), v[0], v[1], v[2])
    v_rot = quat_mult_casadi(quat_mult_casadi(q, v_quat), q_conj)
    return v_rot[1:4]

def quat_mult_numpy(q1, q2):
    """Quaternion multiplication using NumPy."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_rotate_numpy(q, v):
    """Rotate vector v by quaternion q using NumPy."""
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    v_quat = np.concatenate(([0], v))
    v_rot = quat_mult_numpy(quat_mult_numpy(q, v_quat), q_conj)
    return v_rot[1:4]

def Phi_casadi(o, I):
    I_inv = ca.inv(I)
    top_left = 0.5 * Omega(o)
    top_right = ca.SX.zeros(4, 3)
    bottom_left = ca.SX.zeros(3, 4)
    bottom_right = -I_inv @ skew_casadi(o) @ I

    # Build the block matrix using vertcat and horzcat
    return ca.vertcat(
        ca.horzcat(top_left, top_right),
        ca.horzcat(bottom_left, bottom_right)
    )

# Define the state derivative function using CasADi
def rotational_derivative_casadi(state, tau, I):
    quat = state[0:4]
    omega = state[4:7]

    # Quaternion derivative using quaternion multiplication
    omega_quat = ca.vertcat(0, omega[0], omega[1], omega[2])
    quat_dot = 0.5 * quat_mult_casadi(quat, omega_quat)

    # Angular velocity derivative
    omega_dot = ca.solve(I, tau - ca.cross(omega, I @ omega))

    return ca.vertcat(quat_dot, omega_dot)

def smooth_norm(vec, delta):
    return ca.sqrt(ca.dot(vec, vec) + delta**2)

def rotational_derivative(state, tau, I):
    omega = state[4:]
    phi = Phi(omega, I)
    return phi @ state + np.vstack((np.zeros((4, 3)), np.linalg.inv(I))) @ tau

def forward_pass_dynamics(sys_params, bc, ctrl_hist): #where ctrl_hist has ctrl_hist.U => control inpults
    num_steps = sys_params.N
    num_agents = len(sys_params.rs)
    dt = bc.tf / num_steps

    U = np.array(ctrl_hist.U)
    print(f"U shape: {np.shape(ctrl_hist.U)}")

    # Initialize state vectors
    r_fin = np.zeros((num_steps + 1, 3))
    v_fin = np.zeros((num_steps + 1, 3))
    eps_fin = np.zeros((num_steps + 1, 4))
    ome_fin = np.zeros((num_steps + 1, 3))
    Q_fin = np.zeros((num_agents, num_steps + 1, 3))

    # Set initial conditions
    for i in range(num_agents):
        Q_fin[i][0] = sys_params.rs[i]
    r_fin[0] = bc.x0.r
    v_fin[0] = bc.x0.v
    eps_fin[0] = bc.x0.eps / np.linalg.norm(bc.x0.eps)  # Ensure initial quaternion is normalized
    ome_fin[0] = bc.x0.omega

    # Forward pass through the dynamics using provided U
    for k in range(num_steps):
        # Compute current torque and force
        torque_curr = np.sum([skewer(Q_fin[i][k]) @ U[i][k] for i in range(num_agents)], axis=0)
        force_curr = np.sum([U[i][k] for i in range(num_agents)], axis=0)

        # Position and velocity update using linearized dynamics
        f, f_dot, f_ddot = orbital_params(sys_params.mu, sys_params.a, sys_params.e, k * dt)
        A_pos = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        A_vel = np.array([
            [3 * f_dot**2, 0, 0, 0, 2 * f_dot, 0],
            [0, 0, 0, -2 * f_dot, 0, 0],
            [0, 0, -f_ddot, 0, 0, 0]
        ])
        B_vel = np.array([
            [1/sys_params.m, 0, 0],
            [0, 1/sys_params.m, 0],
            [0, 0, 1/sys_params.m]
        ])

        # Dynamics update for position and velocity
        r_fin[k + 1] = r_fin[k] + dt * A_pos @ np.hstack([r_fin[k], v_fin[k]])
        v_fin[k + 1] = v_fin[k] + dt * A_vel @ np.hstack([r_fin[k], v_fin[k]]) + dt * (B_vel @ force_curr)

        # Update orientation (eps) and angular velocity (ome)
        phi = Phi(ome_fin[k], sys_params.I)
        rotational_update = phi @ np.hstack([eps_fin[k], ome_fin[k]])
        eps_next = eps_fin[k] + dt * rotational_update[0:4]
        ome_next = ome_fin[k] + dt * rotational_update[4:] + dt * (np.linalg.inv(sys_params.I) @ torque_curr)

        # Normalize the quaternion part of eps_next
        eps_next /= np.linalg.norm(eps_next)

        # Store the updated quaternion and angular velocity
        eps_fin[k + 1] = eps_next
        ome_fin[k + 1] = ome_next

        # Update Q for each agent based on angular velocity
        for i in range(num_agents):
            Q_ik = Q_fin[i][k]
            Q_next = Q_ik + dt * skewer(ome_fin[k]) @ Q_ik
            Q_next = Q_next / np.linalg.norm(Q_next) * np.linalg.norm(sys_params.rs[i])  # Normalize to original magnitude
            Q_fin[i][k + 1] = Q_next

    # Return the final states (position, velocity, orientation, angular velocity) and Q values
    #X_opt = np.hstack([r_fin, v_fin, eps_fin, ome_fin])
    X_opt = StateVector(r=r_fin, v=v_fin, eps=eps_fin, omega=ome_fin)

    return X_opt, Q_fin
