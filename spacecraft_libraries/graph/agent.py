from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pygad

from spacecraft_libraries.data_structures import BoundaryConditions, SystemParams
from spacecraft_libraries.genetic_code import fitness_func, pop_gen_new


@dataclass
class SpaceAgent:
    position: np.ndarray
    agent_id: int
    sys_params: SystemParams
    bc: BoundaryConditions
    epsilon: float
    population: np.ndarray | None = None
    population_fitness: np.ndarray | None = None
    best_solution: np.ndarray | None = None
    best_solution_fitness: float = -np.inf
    migrants_received: dict[int, np.ndarray] = field(default_factory=dict)
    ga_instance: pygad.GA | None = None

    def initialize(self, pop_size: int) -> None:
        initial_population = pop_gen_new(
            self.bc,
            self.sys_params,
            self.sys_params.N,
            self.epsilon,
            pop_size=pop_size,
        )
        flattened = [x.flatten() for x in initial_population]
        self.population = np.asarray(flattened)

        def fitness_wrapper(ga_instance, solution, solution_idx):
            return fitness_func(
                ga_instance,
                self.sys_params.N,
                self.epsilon,
                self.sys_params,
                self.bc,
                solution,
                solution_idx,
            )

        self.ga_instance = pygad.GA(
            num_generations=1,
            num_parents_mating=max(2, pop_size // 2),
            initial_population=self.population,
            sol_per_pop=pop_size,
            fitness_func=fitness_wrapper,
            num_genes=self.sys_params.N * 3,
            mutation_percent_genes=10,
            crossover_type="two_points",
            mutation_type="random",
            mutation_by_replacement=False,
            parent_selection_type="sss",
            keep_parents=-1,
            keep_elitism=max(1, pop_size // 4),
            allow_duplicate_genes=True,
        )

    def evolve_one_generation(self) -> None:
        if self.ga_instance is None:
            raise RuntimeError("Agent GA instance is not initialized")
        self.ga_instance.initial_population = np.array(self.population, copy=True)
        self.ga_instance.num_generations = 1
        self.ga_instance.run()
        self.population = np.array(self.ga_instance.population, copy=True)
        fit = None
        for attr in ("last_generation_fitness", "last_population_fitness", "population_fitness"):
            if hasattr(self.ga_instance, attr):
                fit = np.array(getattr(self.ga_instance, attr), copy=True)
                break
        self.population_fitness = fit
        self.best_solution, self.best_solution_fitness, _ = self.ga_instance.best_solution()

    def inject_migrants_replace_worst(self) -> None:
        if not self.migrants_received:
            return
        if self.population is None or self.population_fitness is None:
            return
        migrants = np.vstack(list(self.migrants_received.values()))
        m = migrants.shape[0]
        worst_idx = np.argsort(np.asarray(self.population_fitness))[:m]
        self.population[worst_idx] = migrants
        self.population_fitness = None
        self.migrants_received.clear()
