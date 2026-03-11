from cot.evaluation.comparison import compare_methods
from cot.scenarios import scenario_two


def test_compare_methods_runs_all_paths():
    sys_params, bc, epsilon = scenario_two(num_steps=6)
    rows = compare_methods(sys_params, bc, epsilon)
    names = {row['method'] for row in rows}
    assert names == {"centralized_nlp", "centralized_ga", "decentralized_island_ga"}
    for row in rows:
        assert row['cost'] >= 0
        assert row['terminal_violation'] >= 0
