from cot.graph.graph_manager import IslandGraphManager


def solve(sys_params, bc, epsilon, pop_size: int = 4, local_generations: int = 2, migration_rounds: int = 3):
    return IslandGraphManager(sys_params, bc, epsilon).run(
        pop_size=pop_size,
        local_generations=local_generations,
        migration_rounds=migration_rounds,
    )
