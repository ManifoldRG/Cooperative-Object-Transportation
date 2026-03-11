from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacecraft_libraries.evaluation import get_scenario, run_method_comparison


def _print_progress(current: int, total: int, method_name: str, width: int = 30) -> None:
    filled = int(width * current / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    percent = int(100 * current / total) if total else 100
    print(f"\r[{bar}] {percent:3d}% ({current}/{total}) {method_name}", end="", flush=True)
    if current == total:
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare optimization methods across predefined scenarios.")
    parser.add_argument(
        "--scenario",
        type=int,
        default=1,
        choices=(1, 2, 3),
        help="Scenario number to run (1, 2, or 3).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sys_params, bc, epsilon = get_scenario(args.scenario)
    print(f"Running method comparison for scenario {args.scenario}")
    rows = run_method_comparison(sys_params, bc, epsilon, progress_callback=_print_progress)
    print("method,cost,terminal_violation,runtime_s")
    for row in rows:
        print(f"{row['method']},{row['cost']:.6f},{row['terminal_violation']:.6f},{row['runtime_s']:.3f}")


if __name__ == "__main__":
    main()
