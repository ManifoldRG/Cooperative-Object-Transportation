"""Run ONE benchmark task (JSON spec) and write its result row atomically.

Usage: py worker.py <task.json> <result.json>
Exits with os._exit to skip CasADi's interpreter-teardown segfault.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    task = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    from multi_payload.harness import run_task
    row = run_task(task)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, out)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
