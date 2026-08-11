#!/usr/bin/env python3
"""Local (non-SLURM) driver for scripts/csf/baseline_comparison.py.

Enumerates the same scenario x method x time-limit combo grid the CSF array
job uses, filters to the requested methods/limits, and runs the surviving
task-ids through a process pool. Each task writes its own task_XXXX.csv into
--output-dir (same format the CSF merge step consumes).

Example (sampling methods only, two budgets, 6 workers):
    python scripts/run_baseline_local.py \
        --scenarios results/local_scenarios_set2.json \
        --output-dir results/local_baseline_warm \
        --methods centralized_ga centralized_mppi decentralized_mppi \
        --time-limits 60 1200 --workers 6
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

# Must mirror scripts/csf/baseline_comparison.py exactly.
METHODS = [
    "centralized_nlp",
    "centralized_nlp_warm",
    "centralized_ga",
    "centralized_mppi",
    "decentralized_mppi",
]
TIME_LIMITS = [60.0, 300.0, 600.0, 1200.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", nargs="+", default=["centralized_ga", "centralized_mppi", "decentralized_mppi"])
    p.add_argument("--time-limits", nargs="+", type=float, default=TIME_LIMITS)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tau-init-std", type=float, default=0.1)
    p.add_argument("--extra-args", type=str, default="",
                   help="Extra flags passed verbatim to baseline_comparison.py, "
                        "e.g. \"--noise-mode smooth --mppi-sigma 0.03\"")
    p.add_argument("--dry-run", action="store_true", help="print selected tasks and exit")
    return p.parse_args()


def main():
    args = parse_args()
    bad = set(args.methods) - set(METHODS)
    if bad:
        sys.exit(f"unknown methods: {bad}")

    with open(args.scenarios) as f:
        scenarios = json.load(f)

    tasks = []  # (task_id, scenario_id, method, limit)
    for tid, (sc, method, tl) in enumerate(product(scenarios, METHODS, TIME_LIMITS), start=1):
        if method in args.methods and tl in args.time_limits:
            tasks.append((tid, sc["scenario_id"], method, tl))

    print(f"{len(tasks)} tasks selected "
          f"({len(scenarios)} scenarios x {len(args.methods)} methods x {len(args.time_limits)} limits)")
    if args.dry_run:
        for t in tasks:
            print("  task %4d  sc=%-3s %-22s %6.0fs" % t)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    def run_task(task):
        tid, sc_id, method, tl = task
        out_csv = args.output_dir / f"task_{tid:04d}.csv"
        if out_csv.exists():
            return (tid, "skipped (exists)", 0.0)
        cmd = [sys.executable, str(ROOT / "scripts" / "csf" / "baseline_comparison.py"),
               "--task-id", str(tid),
               "--scenarios", str(args.scenarios),
               "--output-dir", str(args.output_dir),
               "--tau-init-std", str(args.tau_init_std)]
        if args.extra_args:
            cmd += args.extra_args.split()
        t0 = time.perf_counter()
        with open(log_dir / f"task_{tid:04d}.log", "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        dt = time.perf_counter() - t0
        return (tid, f"rc={rc}", dt)

    t_start = time.perf_counter()
    done = 0
    # Threads are fine as dispatchers: each task is its own python subprocess.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            tid, sc_id, method, tl = futures[fut]
            res_tid, status, dt = fut.result()
            done += 1
            elapsed = (time.perf_counter() - t_start) / 60
            print(f"[{done}/{len(tasks)}] task {tid:4d} sc={sc_id:<3} {method:<22} "
                  f"{tl:6.0f}s -> {status} in {dt/60:.1f} min (wall {elapsed:.0f} min)",
                  flush=True)

    print("all done.")


if __name__ == "__main__":
    main()
