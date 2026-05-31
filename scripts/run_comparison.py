from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spacecraft_libraries.evaluation import default_scenario, run_method_comparison

# Set to None to disable runtime caps.
TOTAL_RUNTIME_CAP_S = 30.0


def main():
    sys_params, bc, epsilon = default_scenario()
    rows = run_method_comparison(
        sys_params,
        bc,
        epsilon,
        max_runtime_s=TOTAL_RUNTIME_CAP_S,
        show_progress=True,
    )
    print("method,cost,terminal_violation,runtime_s")
    for row in rows:
        print(f"{row['method']},{row['cost']:.6f},{row['terminal_violation']:.6f},{row['runtime_s']:.3f}")


if __name__ == "__main__":
    main()
