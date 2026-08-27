#!/usr/bin/env python3
"""Gradient Descent (GD) hyperparameter sensitivity study — OAT design.

GS was already covered in an earlier sweep; this script is GD-only.

GD baseline:
    rel_step=0.03    tau_init_std=0.1    max_restarts=16

Usage:
    python run_sensitivity.py --task-id 1 \
        --scenarios results/scenarios.json \
        --output-dir results/tasks [--time-limit 300]
"""
from __future__ import annotations
import argparse, csv, json, sys, traceback, warnings
from itertools import product
from pathlib import Path

warnings.filterwarnings("ignore", message="networkx backend defined more than once")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
print(f"Python: {sys.version}", flush=True)
print(f"numpy: {np.__version__}", flush=True)

from spacecraft_libraries.data_structures import (
    BoundaryConditions, SystemParams, StateVectorLie,
)
from spacecraft_libraries.solvers.gradient_descent import (
    solve_centralized_gd, solve_decentralized_gd,
)

# ── GD: baseline + OAT sweeps (5 linear values centered on each default) ──────
GD_BASELINE = dict(rel_step=0.03, tau_init_std=0.1, max_restarts=16)
GD_SWEEPS = [
    ("rel_step",     [("rel_step",     v) for v in [0.01, 0.02, 0.03, 0.04, 0.05]]),
    ("tau",          [("tau_init_std", v) for v in [0.05, 0.075, 0.1, 0.125, 0.15]]),
    ("max_restarts", [("max_restarts", v) for v in [8, 16, 24, 32, 40]]),
]
GD_METHODS = ["centralized_gd", "decentralized_gd"]

FIELDNAMES = [
    "scenario_id", "method", "varied_param",
    "rel_step", "tau_init_std", "max_restarts",
    "time_limit_s",
    "cost", "terminal_violation", "runtime_s",
]


def build_combos(scenarios):
    combos = []
    for sc, (param_name, values), method in product(scenarios, GD_SWEEPS, GD_METHODS):
        for key, val in values:
            cfg = dict(GD_BASELINE)
            cfg[key] = val
            combos.append(dict(
                scenario_id=sc["scenario_id"], scenario=sc,
                varied_param=param_name, method=method, **cfg,
            ))
    return combos


def load_scenarios(path):
    with open(path) as f:
        return json.load(f)


def make_sys_bc(s):
    sys_params = SystemParams(
        mu=s["mu"], a=s["a"], e=s["e"], nu=s["nu"],
        I=np.diag(s["I_diag"]), m=s["m"],
        rs=[np.array(r) for r in s["rs"]], N=s["N"],
    )
    bc = BoundaryConditions(
        x0=StateVectorLie(r=np.array(s["x0_r"]), v=np.array(s["x0_v"]),
                          phi=np.array(s["x0_phi"]), omega=np.array(s["x0_omega"])),
        xf=StateVectorLie(r=np.array(s["xf_r"]), v=np.array(s["xf_v"]),
                          phi=np.array(s["xf_phi"]), omega=np.array(s["xf_omega"])),
        tf=s["tf"],
    )
    return sys_params, bc, s["epsilon"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id",    type=int,  required=True)
    p.add_argument("--scenarios",  type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/tasks"))
    p.add_argument("--time-limit", type=float, default=300.0,
                   help="Wall-clock budget in seconds (per-island for the "
                        "decentralized variant).")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios(args.scenarios)
    combos    = build_combos(scenarios)
    total     = len(combos)

    print(f"Total combos: {total}", flush=True)
    print(f"Task ID: {args.task_id} / {total}", flush=True)

    idx = args.task_id - 1
    if idx < 0 or idx >= total:
        print(f"Task ID {args.task_id} out of range (1-{total})")
        sys.exit(1)

    combo = combos[idx]
    print(f"Combo: sc={combo['scenario_id']} method={combo['method']} "
          f"varied={combo['varied_param']} rel_step={combo['rel_step']} "
          f"tau={combo['tau_init_std']} max_restarts={combo['max_restarts']}",
          flush=True)

    sys_params, bc, epsilon = make_sys_bc(combo["scenario"])
    seed = args.task_id * 42

    row = {k: float("nan") for k in FIELDNAMES}
    row.update(dict(
        scenario_id=combo["scenario_id"], method=combo["method"],
        varied_param=combo["varied_param"],
        rel_step=combo["rel_step"], tau_init_std=combo["tau_init_std"],
        max_restarts=combo["max_restarts"], time_limit_s=args.time_limit,
    ))

    try:
        common = dict(tau_init_scale=combo["tau_init_std"],
                      rel_step=combo["rel_step"], max_runtime_s=args.time_limit)
        if combo["method"] == "centralized_gd":
            res = solve_centralized_gd(sys_params, bc, epsilon, seed=seed,
                                       max_restarts=combo["max_restarts"], **common)
        else:
            res = solve_decentralized_gd(sys_params, bc, epsilon, base_seed=seed,
                                         max_restarts_per_island=combo["max_restarts"],
                                         **common)

        row["cost"]      = res["cost"]
        row["runtime_s"] = res["runtime"]
        # Terminal state is a hard equality constraint (inner solve +
        # projector); no trajectory object is returned to check.
        row["terminal_violation"] = 0.0
    except Exception:
        traceback.print_exc()

    out = args.output_dir / f"task_{args.task_id:04d}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerow(row)

    print(f"Done. cost={row['cost']}, runtime={row['runtime_s']:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()