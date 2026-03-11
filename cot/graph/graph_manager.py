import networkx as nx
import numpy as np
from cot.graph.agent import SpaceAgent
from spacecraft_libraries.new_opts import opt_given_tau_ipopt_new


class IslandGraphManager:
    def __init__(self, sys_params, bc, epsilon):
        self.sys_params = sys_params
        self.bc = bc
        self.epsilon = epsilon
        self.graph = nx.complete_graph(len(sys_params.rs))
        self.agents = [SpaceAgent(i, sys_params, bc, epsilon) for i in range(len(sys_params.rs))]

    def run(self, pop_size: int = 4, local_generations: int = 2, migration_rounds: int = 3):
        for _ in range(migration_rounds):
            for agent in self.agents:
                agent.evolve(pop_size=pop_size, generations=local_generations)
            best_tau = min(self.agents, key=lambda a: a.best_cost).best_tau
            for agent in self.agents:
                agent.best_tau = np.array(best_tau, copy=True)

        winner = min(self.agents, key=lambda a: a.best_cost)
        traj, ctrl, _, cost = opt_given_tau_ipopt_new(winner.best_tau, self.sys_params.N, self.epsilon, self.sys_params, self.bc, num_iter=400)
        return winner.best_tau, traj, ctrl, float(cost)
