"""Benchmark driver: grid of scenarios x oracles, one subprocess per task.

Resume-safe (tasks with a result JSON are skipped), parallel, with a hard
per-task timeout recorded as status "timeout". Merges all rows into
<out>/results.csv at the end; summarize with analyze.py.

Example (stress sweep of the waypoint-repair oracle):
  py multi_payload/benchmark/driver.py --out multi_payload/results/bench_stress \
     --M 3 4 5 6 --layouts crossing random --nus 15 60 --seeds 0 1 \
     --agents-per-payload 3 --oracles waypoint --budget 30 --workers 4
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--M", type=int, nargs="+", default=[3])
    p.add_argument("--layouts", nargs="+", default=["crossing"])
    p.add_argument("--nus", type=float, nargs="+", default=[60.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--points", type=int, default=6)
    p.add_argument("--agents-per-payload", type=float, default=3.0,
                   help="swarm size = round(ratio * M)")
    p.add_argument("--oracles", nargs="+", default=["waypoint"])
    p.add_argument("--allocation", choices=("greedy", "misocp"), default="greedy")
    p.add_argument("--objective", choices=("fuel", "energy"), default="fuel")
    p.add_argument("--budget", type=float, default=30.0, help="per-payload solver budget [s]")
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--tf", type=float, default=None)
    p.add_argument("--ring", type=float, default=500.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=None,
                   help="per-task wall cap [s] (default 15*budget*M + 600)")
    a = p.parse_args()

    tasks_dir, res_dir = a.out / "tasks", a.out / "rows"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    grid = []
    for M, layout, nu, seed, orc in itertools.product(a.M, a.layouts, a.nus, a.seeds, a.oracles):
        task = dict(layout=layout, M=M, points=a.points,
                    agents=int(round(a.agents_per_payload * M)), nu=nu, seed=seed,
                    oracle=orc, allocation=a.allocation, objective=a.objective,
                    budget=a.budget, N=a.N, tf=a.tf, ring=a.ring)
        name = f"{orc}_{layout}_M{M}_nu{nu:g}_s{seed}"
        grid.append((name, task))

    todo = [(n, t) for n, t in grid if not (res_dir / f"{n}.json").exists()]
    print(f"{len(grid)} tasks, {len(grid) - len(todo)} done, {len(todo)} to run", flush=True)
    running = {}
    t_start = time.time()
    while todo or running:
        while todo and len(running) < a.workers:
            name, task = todo.pop(0)
            tf = tasks_dir / f"{name}.json"
            tf.write_text(json.dumps(task))
            err = open(tasks_dir / f"{name}.err", "w")
            proc = subprocess.Popen([sys.executable, str(HERE / "worker.py"), str(tf),
                                     str(res_dir / f"{name}.json")],
                                    stdout=subprocess.DEVNULL, stderr=err)
            cap = a.timeout or 15 * a.budget * task["M"] + 600
            running[name] = (proc, time.time(), cap, task, err)
        time.sleep(2)
        for name in list(running):
            proc, t0, cap, task, err = running[name]
            if proc.poll() is not None:
                err.close()
                del running[name]
                if not (res_dir / f"{name}.json").exists():
                    (res_dir / f"{name}.json").write_text(json.dumps(dict(
                        task, status="crash", success=False, returncode=proc.returncode)))
                print(f"[{time.time() - t_start:6.0f}s] done {name}", flush=True)
            elif time.time() - t0 > cap:
                proc.kill()
                err.close()
                del running[name]
                (res_dir / f"{name}.json").write_text(json.dumps(dict(
                    task, status="timeout", success=False, wall=time.time() - t0)))
                print(f"[{time.time() - t_start:6.0f}s] TIMEOUT {name}", flush=True)
    merge(res_dir, a.out / "results.csv")


def merge(res_dir, csv_path):
    import pandas as pd
    rows = [json.loads(f.read_text()) for f in sorted(res_dir.glob("*.json"))]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"merged {len(rows)} rows -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
