#!/usr/bin/env python3
"""Greedy Sampler (GS) hyperparameter sensitivity study — OAT design.

Baseline (defaults locked in the 2026-08-10/11 tuning sessions):
    sigma=0.03 (RELATIVE: fraction of warm-start nominal RMS torque),
    step_size=1.0 (full jump to batch-best projected sample),
    tau_init_std=0.1 (warm-start shooting init scale; measured inert),
    n_samples=12 (batch size between center jumps).

Sweeps one parameter at a time across 5 LINEAR values centered on the
default, per scenario, per method (centralized_gs, decentralized_gs).
Noise is WHITE (production default; no knots). Runs are deadline-driven:
each task gets --time-limit seconds of wall clock (per-island slices for
the decentralized variant), so n_iter is not a parameter — the batch size
alone sets the jump cadence within the fixed budget.

Total tasks = n_scenarios x 4 params x 5 values x 2 methods
            = 5 scenarios x 40 = 200  (for a 5-scenario scenarios.json)

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
from spacecraft_libraries.evaluation.metrics import terminal_violation, lie_attitude_violation
from spacecraft_libraries.solvers.greedy_sampler import (
    solve_centralized_gs, solve_decentralized_gs,
)

# ── Baseline ──────────────────────────────────────────────────────────────────
BASELINE = dict(sigma=0.03, step_size=1.0, tau_init_std=0.1, n_samples=12)

# ── OAT sweeps: 5 linear values centered on each default ─────────────────────
SWEEPS = [
    ("sigma",      [("sigma",        v) for v in [0.01, 0.02, 0.03, 0.04, 0.05]]),
    ("step_size",  [("step_size",    v) for v in [0.5, 0.75, 1.0, 1.25, 1.5]]),
    ("tau",        [("tau_init_std", v) for v in [0.05, 0.075, 0.1, 0.125, 0.15]]),
    ("batch",      [("n_samples",    v) for v in [4, 8, 12, 16, 20]]),
]

METHODS = ["centralized_gs", "decentralized_gs"]

FIELDNAMES = [
    "scenario_id", "method", "varied_param",
    "sigma", "step_size", "tau_init_std", "n_samples",
    "noise_mode", "time_limit_s",
    "cost", "terminal_violation", "runtime_s",
]


def build_combos(scenarios):
    combos = []
    for sc, (param_name, values), method in product(scenarios, SWEEPS, METHODS):
        for key, val in values:
            cfg = dict(BASELINE)
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


def _violation(traj, bc):
    s = traj.states[-1]
    return float(
        terminal_violation(s.r, bc.xf.r)
        + terminal_violation(s.v, bc.xf.v)
        + lie_attitude_violation(s.phi, bc.xf.phi)
        + terminal_violation(s.omega, bc.xf.omega)
    )


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
          f"varied={combo['varied_param']} sigma={combo['sigma']} "
          f"step={combo['step_size']} tau={combo['tau_init_std']} "
          f"n_samples={combo['n_samples']}", flush=True)

    sys_params, bc, epsilon = make_sys_bc(combo["scenario"])
    seed = args.task_id * 42

    row = {k: float("nan") for k in FIELDNAMES}
    row.update(dict(
        scenario_id=combo["scenario_id"], method=combo["method"],
        varied_param=combo["varied_param"],
        sigma=combo["sigma"], step_size=combo["step_size"],
        tau_init_std=combo["tau_init_std"], n_samples=combo["n_samples"],
        noise_mode="white", time_limit_s=args.time_limit,
    ))

    try:
        kwargs = dict(n_samples=combo["n_samples"], sigma=combo["sigma"],
                      tau_init_scale=combo["tau_init_std"],
                      noise_mode="white", step_size=combo["step_size"],
                      max_runtime_s=args.time_limit)
        if combo["method"] == "centralized_gs":
            res = solve_centralized_gs(sys_params, bc, epsilon, seed=seed, **kwargs)
        else:
            res = solve_decentralized_gs(sys_params, bc, epsilon, base_seed=seed, **kwargs)

        row["cost"]               = res["cost"]
        row["runtime_s"]          = res["runtime"]
        row["terminal_violation"] = _violation(res["trajectory"], bc)
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
