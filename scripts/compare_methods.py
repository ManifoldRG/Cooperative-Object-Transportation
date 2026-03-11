from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cot.evaluation.comparison import compare_methods
from cot.scenarios import scenario_two


def main():
    sys_params, bc, epsilon = scenario_two(num_steps=8)
    rows = compare_methods(sys_params, bc, epsilon)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
