from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from spacecraft_libraries.data_structures import BoundaryConditions, SystemParams
from spacecraft_libraries.evaluation.metrics import terminal_constraint_violation
from spacecraft_libraries.solvers.centralized_ga import solve_centralized_ga
from spacecraft_libraries.solvers.centralized_nlp import solve_centralized_nlp
from spacecraft_libraries.solvers.decentralized_island_ga import solve_decentralized_island_ga


@dataclass
class ComparisonConfig:
    epsilon: float = 1e-5
    ga_pop_size: int = 8
    island_pop_size: int = 8
    island_migration_iterations: int = 4
    nlp_max_iters: int = 5000


def compare_methods(
    sys_params: SystemParams,
    bc: BoundaryConditions,
    config: ComparisonConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ComparisonConfig()
    runs = [
        solve_centralized_nlp(sys_params, bc, max_iters=cfg.nlp_max_iters),
        solve_centralized_ga(sys_params, bc, epsilon=cfg.epsilon, pop_size=cfg.ga_pop_size),
        solve_decentralized_island_ga(
            sys_params,
            bc,
            epsilon=cfg.epsilon,
            pop_size=cfg.island_pop_size,
            migration_iterations=cfg.island_migration_iterations,
        ),
    ]
    rows = []
    for run in runs:
        rows.append(
            {
                "method": run.method,
                "cost": run.cost,
                "terminal_constraint_violation": terminal_constraint_violation(run.trajectory, bc),
                "runtime_s": run.runtime_s,
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
