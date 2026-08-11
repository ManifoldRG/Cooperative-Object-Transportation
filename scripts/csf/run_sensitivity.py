#!/usr/bin/env python3
"""MPPI hyperparameter sensitivity study — OAT (one-at-a-time) design.

Baseline: sigma=0.25, tau=0.1, lambda=1.0, n_iter=10, n_samples=10
Sweeps one parameter at a time across 5 values, per scenario, per method.

Nominal tau comes from the multiple-shooting warm start (make_nominal_tau) —
the same initialization family the GA population uses — NOT a Gaussian cold
start. tau_init_std is the scale of the uniform random tau guess fed to the
shooting IPOPT (GA default 1e-1).

With --time-limit T, runs are DEADLINE-DRIVEN like baseline_comparison.py:
n_iter is effectively unbounded and run_mppi stops at the wall clock (per-
island slices of T each for decentralized). The gamma sweep's n_iter values
are inert in this mode; its n_samples values still set the batch size.
Without --time-limit, behavior is the original untimed n_iter-driven loop.

Total tasks = n_scenarios x 4 params x 5 values x 2 methods
            = 5 scenarios x 20 x 2 = 200  (for a 5-scenario scenarios.json)

Usage:
    python run_sensitivity.py --task-id 1 \
        --scenarios results/scenarios.json \
        --output-dir results/tasks
"""
from __future__ import annotations
import argparse, csv, json, sys, time, traceback, warnings
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
from spacecraft_libraries.solvers.mppi_core import make_nominal_tau, run_mppi
from spacecraft_libraries import new_opts

# ── Baseline ──────────────────────────────────────────────────────────────────
# sigma=0.1 confirmed optimal on the warm-start protocol (local 5-point probe
# at 300s, 2026-08-10: 0.7 never improves the nominal, 0.01 is sub-noise-floor,
# 0.1 beats 0.05 and 0.25).
BASELINE = dict(sigma=0.03, lambda_=1.0, tau_init_std=0.1, n_iter=10, n_samples=10)

# ── OAT sweeps ────────────────────────────────────────────────────────────────
SWEEPS = [
    # RELATIVE sigma (fraction of warm-start nominal RMS torque; see
    # run_mppi(relative_sigma=True)). Log grid spans both observed regimes:
    # polishing (sigma_opt/RMS ~ 0.02, tight warm starts) and escaping
    # (~0.7, loose warm starts). Absolute-sigma sweeps proved scenario-
    # dependent and unrecommendable (2026-08-10/11 local + CSF runs).
    ("sigma",    [("sigma",      v) for v in [0.01, 0.03, 0.1, 0.3, 1.0]]),
    # log-spaced: 1e-3 ~ island-collapse regime, 0.1 = GA-matched default,
    # 5.0 ~ noise-dominated shooting init
    ("tau",      [("tau_init_std", v) for v in [0.001, 0.01, 0.1, 1.0, 5.0]]),
    ("lambda_",  [("lambda_",    v) for v in [0.5, 0.8, 0.9, 1.0, 1.5]]),
    ("gamma",    [("gamma",      v) for v in
                  [(5,10), (10,10), (10,5), (20,10), (10,20)]]),
]

METHODS = ["centralized_mppi", "decentralized_mppi"]

FIELDNAMES = [
    "scenario_id", "method", "varied_param",
    "sigma", "sigma_mode", "lambda_", "n_iter", "n_samples", "tau_init_std",
    "time_limit_s",
    "cost", "terminal_violation", "runtime_s",
    "failed_sample_fraction", "n_iter_completed",
]

# n_iter used when --time-limit makes the loop deadline-driven
DEADLINE_ITERS = 1_000_000


def build_combos(scenarios):
    combos = []
    for sc, (param_name, values), method in product(scenarios, SWEEPS, METHODS):
        for key, val in values:
            cfg = dict(BASELINE)
            if key == "gamma":
                cfg["n_iter"], cfg["n_samples"] = val
                cfg["gamma_label"] = f"{val[0]}iter_{val[1]}samp"
            else:
                cfg[key] = val
                cfg["gamma_label"] = f"{cfg['n_iter']}iter_{cfg['n_samples']}samp"
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


def _run_centralized(sys_params, bc, epsilon, nominal, rng, cfg, time_limit_s=None,
                     relative_sigma=True):
    t0 = time.perf_counter()
    n_iter = cfg["n_iter"] if time_limit_s is None else DEADLINE_ITERS
    best_tau, best_cost, history = run_mppi(
        sys_params, bc, epsilon, nominal,
        n_iter=n_iter, n_samples=cfg["n_samples"],
        sigma=cfg["sigma"], lambda_=cfg["lambda_"], rng=rng,
        deadline_s=time_limit_s, start_time=t0,
        relative_sigma=relative_sigma,
    )
    traj, _, _, cost = new_opts.opt_given_tau_ipopt_new(
        best_tau, sys_params.N, epsilon, sys_params, bc, num_iter=1000
    )
    return traj, float(cost), time.perf_counter() - t0, history.as_dict()


def _run_decentralized(sys_params, bc, epsilon, rng, cfg, time_limit_s=None,
                       relative_sigma=True):
    from scipy.spatial.distance import cdist
    import networkx as nx

    attach_vecs = np.asarray(sys_params.rs)
    num_agents  = attach_vecs.shape[0]
    N           = sys_params.N
    t0          = time.perf_counter()

    distances = cdist(attach_vecs, attach_vecs)
    g = nx.Graph()
    g.add_nodes_from(range(num_agents))
    for i in range(num_agents):
        neighbors = np.where((distances[i] < 100.0) & (distances[i] > 0))[0]
        g.add_edges_from((i, int(j)) for j in neighbors)

    island_costs, island_taus, island_histories = [], [], []
    for i in range(num_agents):
        island_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        nominal = make_nominal_tau(sys_params, bc, epsilon, island_rng,
                                   tau_init_scale=cfg["tau_init_std"])
        # Deadline mode: per-island slice, absolute deadline (i+1)*T from t0
        # (mirrors baseline_comparison.run_decentralized_mppi_warm_start).
        n_iter = cfg["n_iter"] if time_limit_s is None else DEADLINE_ITERS
        island_deadline = None if time_limit_s is None else (i + 1) * time_limit_s
        best_tau, best_cost, history = run_mppi(
            sys_params, bc, epsilon, nominal,
            n_iter=n_iter, n_samples=cfg["n_samples"],
            sigma=cfg["sigma"], lambda_=cfg["lambda_"], rng=island_rng,
            deadline_s=island_deadline, start_time=t0,
            relative_sigma=relative_sigma,
        )
        island_taus.append(best_tau)
        island_costs.append(best_cost)
        island_histories.append(history.as_dict())

    state = {
        i: (1.0 / float(island_costs[i])
            if np.isfinite(island_costs[i]) and island_costs[i] > 0 else 0.0, i)
        for i in range(num_agents)
    }
    for _ in range(100):
        updated = {
            i: max([state[i]] + [state[j] for j in g.neighbors(i)],
                   key=lambda t: t[0])
            for i in range(num_agents)
        }
        if updated == state:
            break
        state = updated
    _, winner_id = next(iter(state.values()))

    winner_tau_proj = new_opts.tau_proj_nonlin_new(
        island_taus[winner_id], N, epsilon, sys_params, bc
    )[0]
    winner_tau_proj = np.asarray(winner_tau_proj, dtype=float).reshape(N, 3)
    traj, _, _, cost = new_opts.opt_given_tau_ipopt_new(
        winner_tau_proj, N, epsilon, sys_params, bc, num_iter=1000
    )
    return traj, float(cost), time.perf_counter() - t0, island_histories[winner_id]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id",    type=int,  required=True)
    p.add_argument("--scenarios",  type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results/tasks"))
    p.add_argument("--time-limit", type=float, default=None,
                   help="Wall-clock budget in seconds; makes runs deadline-"
                        "driven like baseline_comparison.py (per-island for "
                        "decentralized). Omit for the original untimed loop.")
    p.add_argument("--sigma-mode", choices=["absolute", "relative"], default="relative",
                   help="'relative' (default): sigma is a fraction of the warm-"
                        "start nominal's RMS torque — the sigma sweep grid is "
                        "designed for this mode. 'absolute': legacy raw torque "
                        "units.")
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
          f"tau={combo['tau_init_std']} lambda={combo['lambda_']} "
          f"n_iter={combo['n_iter']} n_samples={combo['n_samples']}", flush=True)

    sys_params, bc, epsilon = make_sys_bc(combo["scenario"])
    N   = sys_params.N
    rng = np.random.default_rng(args.task_id * 42)

    row = {k: float("nan") for k in FIELDNAMES}
    row.update(dict(
        scenario_id=combo["scenario_id"], method=combo["method"],
        varied_param=combo["varied_param"],
        sigma=combo["sigma"], lambda_=combo["lambda_"],
        n_iter=combo["n_iter"], n_samples=combo["n_samples"],
        tau_init_std=combo["tau_init_std"],
        time_limit_s=args.time_limit if args.time_limit is not None else "",
        sigma_mode=args.sigma_mode,
    ))

    relative = args.sigma_mode == "relative"
    try:
        nominal = make_nominal_tau(sys_params, bc, epsilon, rng,
                                   tau_init_scale=combo["tau_init_std"])
        if combo["method"] == "centralized_mppi":
            traj, cost, rt, mppi_info = _run_centralized(
                sys_params, bc, epsilon, nominal, rng, combo,
                time_limit_s=args.time_limit, relative_sigma=relative)
        else:
            traj, cost, rt, mppi_info = _run_decentralized(
                sys_params, bc, epsilon, rng, combo,
                time_limit_s=args.time_limit, relative_sigma=relative)

        row["cost"]               = cost
        row["runtime_s"]          = rt
        row["terminal_violation"] = _violation(traj, bc)
        row["failed_sample_fraction"] = float(
            mppi_info.get("failed_sample_fraction", float("nan")))
        n_iter_val = mppi_info.get("n_iter_completed", float("nan"))
        row["n_iter_completed"] = (
            int(n_iter_val) if not np.isnan(float(n_iter_val)) else float("nan"))
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