import numpy as np
from spacecraft_libraries.genetic_code import multiple_shooting_optimization_new
from spacecraft_libraries.new_opts import opt_given_tau_ipopt_new
import casadi as ca


def solve(sys_params, bc, epsilon, num_iter: int = 500):
    dt = bc.tf / sys_params.N
    tau_seed, _ = multiple_shooting_optimization_new(
        bc,
        sys_params.N,
        ca.DM(dt),
        ca.DM(sys_params.I),
        ca.DM(epsilon),
        num_iter=200,
    )
    traj, ctrl, _, cost = opt_given_tau_ipopt_new(tau_seed, sys_params.N, epsilon, sys_params, bc, num_iter=num_iter)
    return tau_seed, traj, ctrl, float(cost)
