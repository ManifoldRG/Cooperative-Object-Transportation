import numpy as np
from cot.solvers.centralized_ga import solve as centralized_ga_solve


class SpaceAgent:
    def __init__(self, agent_id, sys_params, bc, epsilon):
        self.id = agent_id
        self.sys_params = sys_params
        self.bc = bc
        self.epsilon = epsilon
        self.best_tau = None
        self.best_cost = np.inf

    def evolve(self, pop_size: int, generations: int):
        tau, _, _, cost = centralized_ga_solve(
            self.sys_params,
            self.bc,
            self.epsilon,
            pop_size=pop_size,
            generations=generations,
        )
        self.best_tau = tau
        self.best_cost = cost
        return tau, cost
