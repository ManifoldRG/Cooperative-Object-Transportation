import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


def plot_att_vs_time(state_opt, tf):
    if isinstance(state_opt, list):
        num_steps = len(state_opt) - 1
    else:
        num_steps = state_opt.shape[0] - 1

    time = np.linspace(0, tf, num_steps + 1)

    plt.figure(figsize=(12, 8))

    # Plot quaternion components (first 4 state variables)
    plt.subplot(2, 1, 1)
    plt.plot(time, state_opt[:, 0], label=r'$\epsilon_1$')
    plt.plot(time, state_opt[:, 1], label=r'$\epsilon_2$')
    plt.plot(time, state_opt[:, 2], label=r'$\epsilon_3$')
    plt.plot(time, state_opt[:, 3], label=r'$\epsilon_4$')
    plt.title('Quaternion Components vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Quaternion Components')
    plt.legend()

    # Plot angular velocities (last 3 state variables)
    plt.subplot(2, 1, 2)
    plt.plot(time, state_opt[:, 4], label=r'$\omega_1$')
    plt.plot(time, state_opt[:, 5], label=r'$\omega_2$')
    plt.plot(time, state_opt[:, 6], label=r'$\omega_3$')
    plt.title('Angular Velocities vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity [rad/s]')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_qs_vs_time(qs_hist_updated, tf):
    num_steps = len(qs_hist_updated[0])
    time = np.linspace(0, tf, num_steps)

    qs_hist_updated = np.array(qs_hist_updated)  # Convert list of qs to a numpy array

    plt.figure(figsize=(12, 8))

    # Plot each component of q against time
    for i in range(len(qs_hist_updated)):
        plt.plot(time, qs_hist_updated[i], label=f'q{i+1}')

    plt.title('q Components vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('q Components')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pos_vel_vs_time(state_opt, tf):
    if isinstance(state_opt, list):
        num_steps = len(state_opt) - 1
    else:
        num_steps = state_opt.shape[0] - 1

    time = np.linspace(0, tf, num_steps + 1)

    plt.figure(figsize=(12, 8))

    # Plot position components (first 3 state variables: x, y, z)
    plt.subplot(2, 1, 1)
    plt.plot(time, state_opt[:, 0], label='x')
    plt.plot(time, state_opt[:, 1], label='y')
    plt.plot(time, state_opt[:, 2], label='z')
    plt.title('Position Components vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.legend()

    # Plot velocity components (next 3 state variables: vx, vy, vz)
    plt.subplot(2, 1, 2)
    plt.plot(time, state_opt[:, 3], label='vx')
    plt.plot(time, state_opt[:, 4], label='vy')
    plt.plot(time, state_opt[:, 5], label='vz')
    plt.title('Velocity Components vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [m/s]')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_control_forces_vs_time(U_opt, tf):
    num_agents = len(U_opt)
    num_steps = U_opt[0].shape[0]

    time = np.linspace(0, tf, num_steps)

    plt.figure(figsize=(15, 12))

    # Plot control forces in the x direction for all agents
    plt.subplot(3, 1, 1)
    for i in range(num_agents):
        plt.plot(time, U_opt[i][:, 0], label=f'Agent {i+1}')
    plt.title('Control Force in X Direction vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Force in X [N]')
    plt.legend()
    plt.grid(True)

    # Plot control forces in the y direction for all agents
    plt.subplot(3, 1, 2)
    for i in range(num_agents):
        plt.plot(time, U_opt[i][:, 1], label=f'Agent {i+1}')
    plt.title('Control Force in Y Direction vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Force in Y [N]')
    plt.legend()
    plt.grid(True)

    # Plot control forces in the z direction for all agents
    plt.subplot(3, 1, 3)
    for i in range(num_agents):
        plt.plot(time, U_opt[i][:, 2], label=f'Agent {i+1}')
    plt.title('Control Force in Z Direction vs Time')
    plt.xlabel('Time [s]')
    plt.ylabel('Control Force in Z [N]')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_cost_vs_iteration(cost_history, window_size=5, show_trend_line=True):
    """
    Plots the cost vs. iteration with optional moving average trend line.

    Parameters:
    cost_history (list): List of cost values from each iteration.
    window_size (int): Window size for moving average. Default is 5.
    show_trend_line (bool): Whether to show a linear trend line. Default is True.
    """
    iterations = np.arange(1, len(cost_history) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, cost_history, '-o', color='b', label='Cost')

    # Plot moving average trend line
    if window_size > 1:
        moving_avg = uniform_filter1d(cost_history, size=window_size)
        plt.plot(iterations, moving_avg, color='r', linestyle='--', label=f'Moving Avg (window={window_size})')

    # Plot linear regression trend line
    if show_trend_line:
        z = np.polyfit(iterations, cost_history, 1)
        p = np.poly1d(z)
        plt.plot(iterations, p(iterations), color='g', linestyle=':', label='Linear Trend Line')

    plt.title('Cost vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.legend()
    plt.show()