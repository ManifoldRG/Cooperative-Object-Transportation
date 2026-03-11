from __future__ import annotations

import time

import numpy as np

from ..data_structures import BoundaryConditions, SystemParams
from ..graph.graph_manager import GraphManager


def solve_decentralized_island_ga(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    epsilon: float,
    pop_size: int = 10,
    migration_rounds: int = 5,
    max_runtime_s: float | None = None,
):
    manager = GraphManager(
        attach_vecs=np.array(sys_params.rs),
        sys_params=sys_params,
        bc=bc,
        epsilon=epsilon,
        migration_rounds=migration_rounds,
    )
    manager.build_line_of_sight_graph()
    start = time.perf_counter()
    manager.run_island_evolution(pop_size=pop_size, max_runtime_s=max_runtime_s)
    result = manager.run_consensus()
    runtime = time.perf_counter() - start
    result.update({"method": "decentralized_island_ga", "runtime": runtime})
    return result
