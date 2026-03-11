import time
from cot.evaluation.metrics import terminal_violation
from cot.solvers.centralized_nlp import solve as solve_nlp
from cot.solvers.centralized_ga import solve as solve_ga
from cot.solvers.decentralized_island_ga import solve as solve_island


def compare_methods(sys_params, bc, epsilon):
    rows = []
    for name, fn in [
        ("centralized_nlp", lambda: solve_nlp(sys_params, bc, epsilon)),
        ("centralized_ga", lambda: solve_ga(sys_params, bc, epsilon)),
        ("decentralized_island_ga", lambda: solve_island(sys_params, bc, epsilon)),
    ]:
        t0 = time.perf_counter()
        tau, traj, ctrl, cost = fn()
        runtime = time.perf_counter() - t0
        rows.append(
            {
                "method": name,
                "cost": float(cost),
                "terminal_violation": terminal_violation(traj, bc),
                "runtime_s": runtime,
                "tau_shape": tuple(tau.shape),
            }
        )
    return rows
