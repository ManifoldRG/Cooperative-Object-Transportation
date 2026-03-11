import numpy as np
from spacecraft_libraries.new_opts import tau_proj_nonlin_new, opt_given_tau_ipopt_new


def _mutate(parent: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    return parent + np.random.normal(scale=sigma, size=parent.shape)


def solve(sys_params, bc, epsilon, pop_size: int = 6, generations: int = 4):
    n = sys_params.N
    population = [np.random.randn(n, 3) * 0.1 for _ in range(pop_size)]
    best = None
    best_cost = np.inf
    best_out = None

    for _ in range(generations):
        scored = []
        for tau in population:
            tau_proj, _ = tau_proj_nonlin_new(tau, n, epsilon, sys_params, bc, num_iter=200)
            traj, ctrl, _, cost = opt_given_tau_ipopt_new(tau_proj, n, epsilon, sys_params, bc, num_iter=300)
            scored.append((float(cost), tau_proj, traj, ctrl))
        scored.sort(key=lambda x: x[0])
        if scored[0][0] < best_cost:
            best_cost = scored[0][0]
            best = scored[0][1]
            best_out = scored[0][2], scored[0][3]

        elites = [scored[i][1] for i in range(max(1, pop_size // 3))]
        new_pop = elites.copy()
        while len(new_pop) < pop_size:
            parent = elites[np.random.randint(len(elites))]
            new_pop.append(_mutate(parent))
        population = new_pop

    traj, ctrl = best_out
    return best, traj, ctrl, best_cost
