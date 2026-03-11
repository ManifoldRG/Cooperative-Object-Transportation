from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from ..data_structures import BoundaryConditions, SystemParams
from .agent import SpaceAgent


@dataclass
class GraphManager:
    attach_vecs: np.ndarray
    sys_params: SystemParams
    bc: BoundaryConditions
    epsilon: float
    line_of_sight_limit: float = 100.0
    migration_rounds: int = 10

    def __post_init__(self):
        self.num_agents = self.attach_vecs.shape[0]
        self.agents = [
            SpaceAgent(self.attach_vecs[i], i, self.sys_params, self.bc, self.epsilon)
            for i in range(self.num_agents)
        ]
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_agents))

    def build_line_of_sight_graph(self) -> None:
        distances = cdist(self.attach_vecs, self.attach_vecs)
        for i in range(self.num_agents):
            neighbors = np.where((distances[i] < self.line_of_sight_limit) & (distances[i] > 0))[0]
            self.graph.add_edges_from((i, int(j)) for j in neighbors)

    def run_island_evolution(self, pop_size: int) -> None:
        for agent in self.agents:
            agent.initialise_island(pop_size)
        for _ in range(self.migration_rounds):
            for agent in self.agents:
                agent.step_generation()
            self._communicate_best()
            for agent in self.agents:
                agent.inject_migrants_replace_worst()

    def _communicate_best(self) -> None:
        best = {agent.agent_id: np.array(agent.best_solution, copy=True) for agent in self.agents}
        for agent in self.agents:
            agent.migrants_received.clear()
            for n in self.graph.neighbors(agent.agent_id):
                agent.migrants_received[n] = np.array(best[n], copy=True)

    def run_consensus(self, max_iters: int = 100):
        for agent in self.agents:
            agent.finalize_from_tau(agent.best_solution.reshape((self.sys_params.N, 3)))

        state = {agent.agent_id: (1.0 / float(agent.final_cost), agent.final_tau) for agent in self.agents}
        nodes = list(self.graph.nodes)
        for _ in range(max_iters):
            updated = {
                i: max([state[i]] + [state[j] for j in self.graph.neighbors(i)], key=lambda t: t[0])
                for i in nodes
            }
            if updated == state:
                break
            state = updated

        best_fit, best_tau = next(iter(state.values()))
        for agent in self.agents:
            agent.finalize_from_tau(best_tau)
        return {
            "fitness": best_fit,
            "tau": best_tau,
            "trajectory": self.agents[0].final_traj,
            "control": self.agents[0].final_ctrl,
            "cost": self.agents[0].final_cost,
        }
