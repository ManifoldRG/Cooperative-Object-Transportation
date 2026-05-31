"""
compare_methods.py
------------------
Run all solvers across the 3 engineered scenarios and a configurable set of
time limits. Writes one CSV row per (scenario, time_limit, method) and flushes
after every solver call so a crash loses at most one result.

Usage
-----
    python compare_methods.py
    python compare_methods.py --scenario 1 2 --time-limits 30 60
    python compare_methods.py --no-mppi
"""

from pathlib import Path
import argparse
import csv
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacecraft_libraries.evaluation import get_scenario, run_method_comparison

TIME_LIMITS_S: list[float] = [6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare all solvers across predefined scenarios and time limits."
    )
    parser.add_argument(
        "--scenario", type=int, nargs="*", choices=(1, 2, 3), default=[1, 2, 3],
        help="Scenario numbers to run. Default: 1 2 3",
    )
    parser.add_argument(
        "--time-limits", type=float, nargs="*", default=TIME_LIMITS_S,
        help=f"Per-solver time limits in seconds. Default: {TIME_LIMITS_S}",
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    parser.add_argument(
        "--output", type=Path,
        default=Path(f"data/comparison_results/comparison_{timestamp}.csv"),
        help="CSV output path.",
    )
    parser.add_argument("--no-mppi", action="store_true", help="Skip MPPI methods.")
    parser.add_argument("--mppi-iterations", type=int,   default=5)
    parser.add_argument("--mppi-samples",    type=int,   default=10)
    parser.add_argument("--mppi-sigma",      type=float, default=1e-1)
    parser.add_argument("--mppi-lambda",     type=float, default=1.0)
    parser.add_argument("--mppi-base-seed",  type=int,   default=42)
    return parser.parse_args()


def print_rows(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"  {row['method']:<30}"
            f"  cost={row['cost']:.3e}"
            f"  violation={row['terminal_violation']:.3e}"
            f"  runtime={row['runtime_s']:.1f}s"
        )


def main() -> None:
    args = parse_args()
    fieldnames = ["scenario", "time_limit_s", "method", "cost", "terminal_violation", "runtime_s"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scenario in args.scenario:
            sys_params, bc, epsilon = get_scenario(scenario)

            for time_limit_s in args.time_limits:
                print(f"\nScenario {scenario} | time limit {int(time_limit_s)}s")
                rows = run_method_comparison(
                    sys_params, bc, epsilon,
                    max_runtime_s=time_limit_s,
                    show_progress=True,
                    silence_solver_output=True,
                    include_mppi=not args.no_mppi,
                    mppi_iterations=args.mppi_iterations,
                    mppi_samples=args.mppi_samples,
                    mppi_sigma=args.mppi_sigma,
                    mppi_lambda=args.mppi_lambda,
                    mppi_base_seed=args.mppi_base_seed,
                )
                for row in rows:
                    row["scenario"] = scenario
                    row["time_limit_s"] = time_limit_s

                print_rows(rows)
                writer.writerows(rows)
                f.flush()

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()