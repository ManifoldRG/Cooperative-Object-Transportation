from __future__ import annotations

import numpy as np

from spacecraft_libraries.data_structures import BoundaryConditions, SystemParams
from spacecraft_libraries.graph.graph_manager import GraphManager
from spacecraft_libraries.solvers.centralized_nlp import SolverRun


def solve_decentralized_island_ga(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float = 1e-5,
    pop_size: int = 8,
    migration_iterations: int = 4,
) -> SolverRun:
    manager = GraphManager(np.asarray(sys_params.rs), sys_params, bc, epsilon=epsilon)
    return manager.run_island_ga(pop_size=pop_size, migration_iterations=migration_iterations)
