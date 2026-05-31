"""Top-level package exports for :mod:`spacecraft_libraries`.

This module intentionally uses lazy imports so users can access only the
submodules they need without importing every optional dependency up front.
"""

from importlib import import_module
from typing import Dict, Tuple

# Export map: public symbol -> (module_name, attribute_name)
_EXPORTS: Dict[str, Tuple[str, str]] = {
    # orbital_helpers
    "kepler_equation_solver": ("orbital_helpers", "kepler_equation_solver"),
    "true_anomaly_f": ("orbital_helpers", "true_anomaly_f"),
    "f_dot": ("orbital_helpers", "f_dot"),
    "f_double_dot": ("orbital_helpers", "f_double_dot"),
    "orbital_params": ("orbital_helpers", "orbital_params"),
    # dynamics
    "skewer": ("dynamics", "skewer"),
    "Theta": ("dynamics", "Theta"),
    "Omega": ("dynamics", "Omega"),
    "Phi": ("dynamics", "Phi"),
    "zoh_Phi": ("dynamics", "zoh_Phi"),
    "A_matrix_3D": ("dynamics", "A_matrix_3D"),
    "B_matrix_3D": ("dynamics", "B_matrix_3D"),
    "discretize_matrices_zoh": ("dynamics", "discretize_matrices_zoh"),
    "state_derivative": ("dynamics", "state_derivative"),
    "another_state_derivative": ("dynamics", "another_state_derivative"),
    "state_derivative_og": ("dynamics", "state_derivative_og"),
    "another_state_derivative_og": ("dynamics", "another_state_derivative_og"),
    "skew_casadi": ("dynamics", "skew_casadi"),
    "quat_mult_casadi": ("dynamics", "quat_mult_casadi"),
    "quat_rotate_casadi": ("dynamics", "quat_rotate_casadi"),
    "quat_mult_numpy": ("dynamics", "quat_mult_numpy"),
    "quat_rotate_numpy": ("dynamics", "quat_rotate_numpy"),
    "Phi_casadi": ("dynamics", "Phi_casadi"),
    "rotational_derivative_casadi": ("dynamics", "rotational_derivative_casadi"),
    "smooth_norm": ("dynamics", "smooth_norm"),
    "rotational_derivative": ("dynamics", "rotational_derivative"),
    "forward_pass_dynamics": ("dynamics", "forward_pass_dynamics"),
    # optimisers
    "opt_given_tau_clean": ("optimisers", "opt_given_tau_clean"),
    "tau_projection_linear_clean": ("optimisers", "tau_projection_linear_clean"),
    "tau_proj_nonlin_clean": ("optimisers", "tau_proj_nonlin_clean"),
    "build_initial_Data": ("optimisers", "build_initial_Data"),
    "gradient_descent_loop_clean": ("optimisers", "gradient_descent_loop_clean"),
    # plotters
    "plot_att_vs_time": ("plotters", "plot_att_vs_time"),
    "plot_qs_vs_time": ("plotters", "plot_qs_vs_time"),
    "plot_pos_vel_vs_time": ("plotters", "plot_pos_vel_vs_time"),
    "plot_control_forces_vs_time": ("plotters", "plot_control_forces_vs_time"),
    "plot_cost_vs_iteration": ("plotters", "plot_cost_vs_iteration"),
    # data_structures
    "SystemParams": ("data_structures", "SystemParams"),
    "StateVector": ("data_structures", "StateVector"),
    "Trajectory": ("data_structures", "Trajectory"),
    "ControlHistory": ("data_structures", "ControlHistory"),
    "BoundaryConditions": ("data_structures", "BoundaryConditions"),
    # og_opts
    "opt_given_tau": ("og_opts", "opt_given_tau"),
    "tau_proj_lin": ("og_opts", "tau_proj_lin"),
    "seq_conv_opt": ("og_opts", "seq_conv_opt"),
    "full_nlp": ("og_opts", "full_nlp"),
    "x_init": ("og_opts", "x_init"),
    "gradient_descent_loop": ("og_opts", "gradient_descent_loop"),
    "mimd_descent_loop": ("og_opts", "mimd_descent_loop"),
    "momentum_gradient_descent_ema": ("og_opts", "momentum_gradient_descent_ema"),
    "tau_proj_nonlin": ("og_opts", "tau_proj_nonlin"),
    "opt_given_tau_ipopt": ("og_opts", "opt_given_tau_ipopt"),
    # genetic_code
    "multiple_shooting_optimization_new": ("genetic_code", "multiple_shooting_optimization_new"),
    "multiple_shooting_optimization": ("genetic_code", "multiple_shooting_optimization"),
    "pop_gen_new": ("genetic_code", "pop_gen_new"),
    "pop_gen": ("genetic_code", "pop_gen"),
    "fitness_func": ("genetic_code", "fitness_func"),
    # new_opts
    "tau_proj_nonlin_new": ("new_opts", "tau_proj_nonlin_new"),
    "opt_given_tau_ipopt_new": ("new_opts", "opt_given_tau_ipopt_new"),
    "opt_given_tau_cvx_new": ("new_opts", "opt_given_tau_cvx_new"),

    # graph
    "SpaceAgent": ("graph.agent", "SpaceAgent"),
    "GraphManager": ("graph.graph_manager", "GraphManager"),
    # solvers
    "solve_centralized_nlp": ("solvers.centralized_nlp", "solve_centralized_nlp"),
    "solve_centralized_ga": ("solvers.centralized_ga", "solve_centralized_ga"),
    "solve_decentralized_island_ga": ("solvers.decentralized_island_ga", "solve_decentralized_island_ga"),
    # evaluation
    "default_scenario": ("evaluation.comparison", "default_scenario"),
    "get_scenario": ("evaluation.comparison", "get_scenario"),
    "scenario_1": ("evaluation.comparison", "scenario_1"),
    "scenario_2": ("evaluation.comparison", "scenario_2"),
    "scenario_3": ("evaluation.comparison", "scenario_3"),
    "run_method_comparison": ("evaluation.comparison", "run_method_comparison"),
    # module export
    "animator": ("animator", None),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """Resolve public attributes lazily from their source module."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(f".{module_name}", __name__)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
