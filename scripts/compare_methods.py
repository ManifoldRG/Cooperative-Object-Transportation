from pathlib import Path
import argparse
import sys
import csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacecraft_libraries.evaluation import get_scenario, run_method_comparison

TIME_LIMITS_S = [60.0, 300.0, 600.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare optimization methods across predefined scenarios and time limits."
    )
    parser.add_argument(
        "--scenario",
        type=int,
        nargs="*",
        choices=(1, 2, 3),
        default=[1, 2, 3],
        help="Scenario numbers to run. Default: 1 2 3",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("method_comparison_results.csv"),
        help="CSV output path.",
    )
    return parser.parse_args()


def print_rows(rows: list[dict]) -> None:
    print("method,cost,terminal_violation,runtime_s")
    for row in rows:
        print(
            f"{row['method']},"
            f"{row['cost']:.3e},"
            f"{row['terminal_violation']:.3e},"
            f"{row['runtime_s']:.1f}"
        )


def main() -> None:
    args = parse_args()
    all_rows: list[dict] = []

    for scenario in args.scenario:
        sys_params, bc, epsilon = get_scenario(scenario)

        for time_limit_s in TIME_LIMITS_S:
            print(f"\nRunning scenario {scenario} with time limit {int(time_limit_s)} s")
            rows = run_method_comparison(
                sys_params,
                bc,
                epsilon,
                max_runtime_s=time_limit_s,
            )

            for row in rows:
                row["scenario"] = scenario
                row["time_limit_s"] = time_limit_s

            print_rows(rows)
            all_rows.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scenario",
        "time_limit_s",
        "method",
        "cost",
        "terminal_violation",
        "runtime_s",
    ]

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()