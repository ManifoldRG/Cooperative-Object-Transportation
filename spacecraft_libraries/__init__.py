#spacecraft libraries init file

from .orbital_helpers import (
    kepler_equation_solver,
    true_anomaly_f,
    f_dot,
    f_double_dot,
    orbital_params
)

from .dynamics import (
    skewer,
    Theta,
    Omega,
    Phi,
    zoh_Phi,
    A_matrix_3D,
    B_matrix_3D,
    discretize_matrices_zoh,
    state_derivative,
    another_state_derivative,
    state_derivative_og,
    another_state_derivative_og,
    skew_casadi,
    quat_mult_casadi,
    quat_rotate_casadi,
    quat_mult_numpy,
    quat_rotate_numpy,
    Phi_casadi,
    rotational_derivative_casadi,
    smooth_norm,
    rotational_derivative,
    forward_pass_dynamics
)

from .optimisers import (
    opt_given_tau_clean,
    tau_projection_linear_clean,
    tau_proj_nonlin_clean,
    build_initial_Data,
    gradient_descent_loop_clean
)

from .plotters import (
    plot_att_vs_time,
    plot_qs_vs_time,
    plot_pos_vel_vs_time,
    plot_control_forces_vs_time,
    plot_cost_vs_iteration
)

from . import animator

from .data_structures import (
    SystemParams,
    StateVector,
    Trajectory,
    ControlHistory,
    BoundaryConditions
)

from .og_opts import (
    opt_given_tau,
    tau_proj_lin,
    seq_conv_opt,
    full_nlp,
    x_init,
    gradient_descent_loop,
    mimd_descent_loop,
    momentum_gradient_descent_ema,
    tau_proj_nonlin,
    opt_given_tau_ipopt
)

from .genetic_code import (
    multiple_shooting_optimization_new,
    multiple_shooting_optimization,
    pop_gen_new,
    pop_gen,
    fitness_func
)

from .new_opts import(
    tau_proj_nonlin_new,
    opt_given_tau_ipopt_new,
    opt_given_tau_cvx_new
)

