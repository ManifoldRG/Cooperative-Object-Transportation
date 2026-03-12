from __future__ import annotations

import time

import numpy as np
import pygad

from .. import genetic_code, new_opts
from ..data_structures import BoundaryConditions, SystemParams


def solve_centralized_ga(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    pop_size: int = 20,
    generations: int = 10,
    max_runtime_s: float | None = None,
):
    def fitness_wrapper(ga_instance, solution, solution_idx):
        return genetic_code.fitness_func(
            ga_instance,
            sys_params.N,
            epsilon,
            sys_params,
            bc,
            solution,
            solution_idx,
        )

    init_pop = genetic_code.pop_gen_new(bc, sys_params, sys_params.N, epsilon, pop_size)
    init_pop = [candidate.flatten() for candidate in init_pop]

    ga = pygad.GA(
        num_generations=1,
        num_parents_mating=max(2, pop_size // 2),
        initial_population=init_pop,
        fitness_func=fitness_wrapper,
        num_genes=sys_params.N * 3,
        mutation_percent_genes=10,
        crossover_type="two_points",
        mutation_type="random",
        parent_selection_type="sss",
        keep_parents=-1,
        keep_elitism=max(1, pop_size // 2),
        allow_duplicate_genes=True,
    )

    start = time.perf_counter()
    generations_completed = 0
    while generations_completed < generations:
        ga.run()
        generations_completed += 1
        ga.initial_population = np.array(ga.population, copy=True)
        if max_runtime_s is not None and (time.perf_counter() - start) >= max_runtime_s:
            break
    runtime = time.perf_counter() - start

    best_solution, _, _ = ga.best_solution()
    tau = best_solution.reshape((sys_params.N, 3))
    tau = new_opts.tau_proj_nonlin_new(tau, sys_params.N, epsilon, sys_params, bc)[0]
    traj, ctrl, q, cost = new_opts.opt_given_tau_ipopt_new(tau, sys_params.N, epsilon, sys_params, bc, num_iter=3000)

    return {
        "method": "centralized_ga",
        "tau": tau,
        "trajectory": traj,
        "control": ctrl,
        "attachment": q,
        "cost": float(cost),
        "runtime": runtime,
    }
