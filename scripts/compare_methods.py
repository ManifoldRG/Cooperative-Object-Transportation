from spacecraft_libraries.evaluation import ComparisonConfig, compare_methods, default_scenario


def main() -> None:
    sys_params, bc = default_scenario()
    config = ComparisonConfig(epsilon=1e-5, ga_pop_size=6, island_pop_size=6, island_migration_iterations=3, nlp_max_iters=3000)
    result = compare_methods(sys_params, bc, config)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
