from __future__ import annotations

import time

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from spacecraft_libraries.data_structures import BoundaryConditions, SystemParams
from spacecraft_libraries.graph.agent import SpaceAgent
from spacecraft_libraries.new_opts import opt_given_tau_ipopt_new, tau_proj_nonlin_new
from spacecraft_libraries.solvers.centralized_nlp import SolverRun


class GraphManager:
    def __init__(
        self,
        attach_vecs: np.ndarray,
        sys_params: SystemParams,
        bc: BoundaryConditions,
        epsilon: float,
        line_of_sight_limit: float = 100.0,
    ):
        self.sys_params = sys_params
        self.bc = bc
        self.epsilon = epsilon
        self.graph = nx.Graph()
        self.agents = [
            SpaceAgent(position=attach_vecs[i], agent_id=i, sys_params=sys_params, bc=bc, epsilon=epsilon)
            for i in range(len(attach_vecs))
        ]
        self.graph.add_nodes_from(range(len(self.agents)))
        distances = cdist([a.position for a in self.agents], [a.position for a in self.agents])
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if 0 < distances[i, j] < line_of_sight_limit:
                    self.graph.add_edge(i, j)

    def run_island_ga(self, pop_size: int = 8, migration_iterations: int = 4) -> SolverRun:
        t0 = time.perf_counter()
        for agent in self.agents:
            agent.initialize(pop_size=pop_size)

        for _ in range(migration_iterations):
            for agent in self.agents:
                agent.evolve_one_generation()

            best_by_agent = {agent.agent_id: np.array(agent.best_solution, copy=True) for agent in self.agents}
            for agent in self.agents:
                for n in self.graph.neighbors(agent.agent_id):
                    agent.migrants_received[n] = np.array(best_by_agent[n], copy=True)
            for agent in self.agents:
                agent.inject_migrants_replace_worst()

        best_tau = None
        best_cost = np.inf
        best_data = None
        for agent in self.agents:
            tau = agent.best_solution.reshape((self.sys_params.N, 3))
            tau_proj, _ = tau_proj_nonlin_new(tau, self.sys_params.N, self.epsilon, self.sys_params, self.bc)
            traj, ctrl, q_hist, cost = opt_given_tau_ipopt_new(
                tau_proj,
                self.sys_params.N,
                self.epsilon,
                self.sys_params,
                self.bc,
            )
            if cost < best_cost:
                best_tau = tau_proj
                best_cost = float(cost)
                best_data = (traj, ctrl, q_hist)

        if best_tau is None or best_data is None:
            raise RuntimeError("Decentralized island GA did not produce a solution")

        traj, ctrl, q_hist = best_data
        runtime_s = time.perf_counter() - t0
        return SolverRun(
            method="decentralized_island_ga",
            trajectory=traj.as_array(),
            control=ctrl.U,
            q_history=np.asarray(q_hist),
            cost=best_cost,
            runtime_s=runtime_s,
        )
