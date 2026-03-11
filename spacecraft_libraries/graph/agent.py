from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pygad

from .. import genetic_code, new_opts
from ..data_structures import BoundaryConditions, ControlHistory, SystemParams, Trajectory


@dataclass
class SpaceAgent:
    position: np.ndarray
    agent_id: int
    sys_params: SystemParams
    bc: BoundaryConditions
    epsilon: float
    best_solution: Optional[np.ndarray] = None
    best_solution_fitness: float = -np.inf
    population: Optional[np.ndarray] = None
    population_fitness: Optional[np.ndarray] = None
    migrants_received: Dict[int, np.ndarray] = field(default_factory=dict)
    final_traj: Optional[Trajectory] = None
    final_ctrl: Optional[ControlHistory] = None
    final_cost: Optional[float] = None
    final_tau: Optional[np.ndarray] = None

    def initialise_island(self, pop_size: int) -> None:
        population = genetic_code.pop_gen_new(self.bc, self.sys_params, self.sys_params.N, self.epsilon, pop_size)
        self.population = np.array([p.flatten() for p in population])

        def fitness_wrapper(ga_instance, solution, solution_idx):
            return genetic_code.fitness_func(
                ga_instance,
                self.sys_params.N,
                self.epsilon,
                self.sys_params,
                self.bc,
                solution,
                solution_idx,
            )

        self._ga = pygad.GA(
            num_generations=1,
            num_parents_mating=max(2, pop_size // 2),
            initial_population=self.population,
            sol_per_pop=pop_size,
            fitness_func=fitness_wrapper,
            num_genes=self.sys_params.N * 3,
            mutation_percent_genes=10,
            crossover_type="two_points",
            mutation_type="random",
            parent_selection_type="sss",
            keep_parents=-1,
            keep_elitism=max(1, pop_size // 4),
            allow_duplicate_genes=True,
        )

    def step_generation(self) -> None:
        if self.population is None:
            raise RuntimeError("Call initialise_island() before stepping generations.")
        self._ga.initial_population = np.array(self.population, copy=True)
        self._ga.num_generations = 1
        self._ga.run()
        self.population = np.array(self._ga.population, copy=True)
        self.population_fitness = np.array(getattr(self._ga, "last_generation_fitness"))
        self.best_solution, self.best_solution_fitness, _ = self._ga.best_solution()

    def inject_migrants_replace_worst(self) -> None:
        if not self.migrants_received:
            return
        if self.population_fitness is None:
            return
        migrants = np.vstack(list(self.migrants_received.values()))
        worst = np.argsort(self.population_fitness)[: migrants.shape[0]]
        self.population[worst] = migrants
        self.population_fitness = None

    def finalize_from_tau(self, tau: np.ndarray) -> None:
        projected_tau = new_opts.tau_proj_nonlin_new(tau, self.sys_params.N, self.epsilon, self.sys_params, self.bc)[0]
        traj, ctrl, _, cost = new_opts.opt_given_tau_ipopt_new(
            projected_tau,
            self.sys_params.N,
            self.epsilon,
            self.sys_params,
            self.bc,
            num_iter=3000,
        )
        self.final_tau = projected_tau
        self.final_traj = traj
        self.final_ctrl = ctrl
        self.final_cost = cost
