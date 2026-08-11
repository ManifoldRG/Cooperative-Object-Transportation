#!/usr/bin/env python3
"""Run the Greedy Sampler over a scenarios JSON, one CSV row per scenario,
mirroring the baseline harness format. Seeds follow derive_solver_seed
(scenario_seed*10007 + time_limit).

Usage:
    python scripts/run_gs_local.py --scenarios results/local_scenarios_5.json \
        --output-dir results/local_gs_white_s003 --time-limit 300 \
        --sigma 0.03 --noise-mode white
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spacecraft_libraries.data_structures import BoundaryConditions, StateVectorLie, SystemParams
from spacecraft_libraries.evaluation.metrics import lie_attitude_violation, terminal_violation
from spacecraft_libraries.solvers.greedy_sampler import solve_centralized_gs, solve_decentralized_gs

FIELDNAMES = [
    "scenario_id", "method", "time_limit_s",
    "cost", "terminal_violation", "runtime_s",
    "n_agents", "a", "e", "m", "tf", "epsilon_tol",
    "solver_seed", "tau_init_std", "mppi_sigma", "sigma_mode", "noise_mode", "noise_knots",
    "n_samples", "gs_iters",
]


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--time-limit", type=float, default=300.0)
    p.add_argument("--sigma", type=float, default=0.03)
    p.add_argument("--noise-mode", choices=["white", "smooth"], default="white")
    p.add_argument("--noise-knots", type=int, default=8)
    p.add_argument("--n-samples", type=int, default=12)
    p.add_argument("--step-size", type=float, default=1.0)
    p.add_argument("--tau-init-std", type=float, default=0.1)
    p.add_argument("--decentralized", action="store_true")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = json.load(open(args.scenarios))

    for s in scenarios:
        sc_id = s["scenario_id"]
        out = args.output_dir / f"gs_{sc_id:04d}.csv"
        if out.exists():
            print(f"sc {sc_id}: exists, skipping", flush=True)
            continue
        sys_params, bc, epsilon = make_sys_bc(s)
        seed = int(s.get("seed", 42)) * 10007 + int(args.time_limit)
        method = "decentralized_gs" if args.decentralized else "centralized_gs"
        row = {k: float("nan") for k in FIELDNAMES}
        row.update(scenario_id=sc_id, method=method, time_limit_s=args.time_limit,
                   n_agents=len(sys_params.rs), a=sys_params.a, e=sys_params.e,
                   m=sys_params.m, tf=bc.tf, epsilon_tol=epsilon, solver_seed=seed,
                   tau_init_std=args.tau_init_std, mppi_sigma=args.sigma,
                   sigma_mode="relative", noise_mode=args.noise_mode,
                   noise_knots=args.noise_knots, n_samples=args.n_samples)
        try:
            kwargs = dict(n_samples=args.n_samples, sigma=args.sigma,
                          tau_init_scale=args.tau_init_std,
                          noise_mode=args.noise_mode, noise_knots=args.noise_knots,
                          step_size=args.step_size,
                          max_runtime_s=args.time_limit)
            if args.decentralized:
                res = solve_decentralized_gs(sys_params, bc, epsilon,
                                             base_seed=seed, **kwargs)
                iters = ""
            else:
                res = solve_centralized_gs(sys_params, bc, epsilon,
                                           seed=seed, **kwargs)
                iters = res["gs"].get("n_iter_completed", "")
            st = res["trajectory"].states[-1]
            viol = (terminal_violation(st.r, bc.xf.r)
                    + terminal_violation(st.v, bc.xf.v)
                    + lie_attitude_violation(st.phi, bc.xf.phi)
                    + terminal_violation(st.omega, bc.xf.omega))
            row.update(cost=res["cost"], terminal_violation=float(viol),
                       runtime_s=res["runtime"], gs_iters=iters)
            print(f"sc {sc_id}: cost {res['cost']:.6f}  viol {viol:.2e}  "
                  f"rt {res['runtime']:.1f}s", flush=True)
        except Exception:
            traceback.print_exc()
            print(f"sc {sc_id}: FAILED", flush=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
