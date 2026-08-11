#!/usr/bin/env python3
"""Local (non-SLURM) driver for scripts/csf/run_sensitivity.py.

Enumerates the same scenario x sweep x method combo grid the CSF array job
uses (task-ids identical), filters to the requested methods, and runs the
surviving tasks through a process pool. Each task writes task_XXXX.csv into
--output-dir, same format the CSF merge consumes.

Example (centralized only, 300s deadline-driven, 6 workers):
    python scripts/run_sensitivity_local.py \
        --scenarios results/local_scenarios_5.json \
        --output-dir results/local_sensitivity_300s \
        --methods centralized_mppi --time-limit 300 --workers 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import the grid definition so task-ids can never drift from the CSF script.
from scripts.csf.run_sensitivity import METHODS, SWEEPS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=["centralized_mppi"])
    p.add_argument("--sweeps", nargs="+", default=None,
                   help="Restrict to these varied params (e.g. sigma tau); default all")
    p.add_argument("--time-limit", type=float, default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    bad = set(args.methods) - set(METHODS)
    if bad:
        sys.exit(f"unknown methods: {bad}")

    with open(args.scenarios) as f:
        scenarios = json.load(f)

    # Mirror run_sensitivity.build_combos enumeration order exactly.
    tasks = []  # (task_id, scenario_id, varied_param, key, value, method)
    tid = 0
    for sc, (param_name, values), method in product(scenarios, SWEEPS, METHODS):
        for key, val in values:
            tid += 1
            if method in args.methods and (args.sweeps is None or param_name in args.sweeps):
                tasks.append((tid, sc["scenario_id"], param_name, key, val, method))

    print(f"{tid} total combos; {len(tasks)} selected for {args.methods}")
    if args.dry_run:
        for t in tasks:
            print("  task %4d  sc=%-3s sweep=%-8s %s=%-12s %s" %
                  (t[0], t[1], t[2], t[3], t[4], t[5]))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    def run_task(task):
        task_id, sc_id, param, key, val, method = task
        out_csv = args.output_dir / f"task_{task_id:04d}.csv"
        if out_csv.exists():
            return (task_id, "skipped (exists)", 0.0)
        cmd = [sys.executable, str(ROOT / "scripts" / "csf" / "run_sensitivity.py"),
               "--task-id", str(task_id),
               "--scenarios", str(args.scenarios),
               "--output-dir", str(args.output_dir)]
        if args.time_limit is not None:
            cmd += ["--time-limit", str(args.time_limit)]
        t0 = time.perf_counter()
        with open(log_dir / f"task_{task_id:04d}.log", "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        return (task_id, f"rc={rc}", time.perf_counter() - t0)

    t_start = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            task_id, sc_id, param, key, val, method = futures[fut]
            _, status, dt = fut.result()
            done += 1
            wall = (time.perf_counter() - t_start) / 60
            print(f"[{done}/{len(tasks)}] task {task_id:4d} sc={sc_id:<3} "
                  f"{param:<8} {key}={val!s:<12} -> {status} in {dt/60:.1f} min "
                  f"(wall {wall:.0f} min)", flush=True)

    print("all done.")


if __name__ == "__main__":
    main()
