#!/usr/bin/env python3
"""Run the projected-gradient-descent solver over a scenarios JSON, one row
per scenario, mirroring baseline_comparison.py's CSV format (method
'centralized_gd'). Seeds follow derive_solver_seed (scenario_seed*10007+limit).

Usage:
    python scripts/run_gd_local.py --scenarios results/local_scenarios_5.json \
        --output-dir results/local_gd --time-limit 300
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

from spacecraft_libraries import new_opts
from spacecraft_libraries.data_structures import BoundaryConditions, StateVectorLie, SystemParams
from spacecraft_libraries.evaluation.metrics import lie_attitude_violation, terminal_violation
from spacecraft_libraries.solvers.gradient_descent import (
    solve_centralized_gd, solve_decentralized_gd,
)

FIELDNAMES = [
    "scenario_id", "method", "time_limit_s",
    "cost", "terminal_violation", "runtime_s",
    "n_agents", "a", "e", "m", "tf", "epsilon_tol",
    "solver_seed", "tau_init_std",
    "gd_iterations", "gd_inner_solves", "gd_final_grad_norm",
    "gd_restarts", "gd_restart_J",
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
    p.add_argument("--tau-init-std", type=float, default=0.1)
    p.add_argument("--rel-step", type=float, default=0.03)
    p.add_argument("--decentralized", action="store_true",
                   help="One GD island per agent + LoS max-consensus "
                        "(single start per island).")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = json.load(open(args.scenarios))

    for s in scenarios:
        sc_id = s["scenario_id"]
        out = args.output_dir / f"gd_{sc_id:04d}.csv"
        if out.exists():
            print(f"sc {sc_id}: exists, skipping", flush=True)
            continue
        sys_params, bc, epsilon = make_sys_bc(s)
        seed = int(s.get("seed", 42)) * 10007 + int(args.time_limit)
        method = "decentralized_gd" if args.decentralized else "centralized_gd"
        row = {k: float("nan") for k in FIELDNAMES}
        row.update(scenario_id=sc_id, method=method,
                   time_limit_s=args.time_limit, n_agents=len(sys_params.rs),
                   a=sys_params.a, e=sys_params.e, m=sys_params.m, tf=bc.tf,
                   epsilon_tol=epsilon, solver_seed=seed,
                   tau_init_std=args.tau_init_std)
        try:
            if args.decentralized:
                res = solve_decentralized_gd(
                    sys_params, bc, epsilon, base_seed=seed,
                    tau_init_scale=args.tau_init_std, rel_step=args.rel_step,
                    max_runtime_s=args.time_limit, max_restarts_per_island=1)
            else:
                res = solve_centralized_gd(
                    sys_params, bc, epsilon, seed=seed,
                    tau_init_scale=args.tau_init_std, rel_step=args.rel_step,
                    max_runtime_s=args.time_limit)
            # terminal violation: r/v from inner solution; attitude by DIRECT
            # numerical rollout of the final tau (no NLP — a re-projection can
            # fail on fragile scenarios and report spurious violations).
            dt = bc.tf / sys_params.N
            I_mat = np.asarray(sys_params.I)
            I_inv = np.linalg.inv(I_mat)
            ome_N = np.asarray(bc.x0.omega, dtype=float).copy()
            R_N = new_opts.so3_exp(new_opts.state_attitude_to_phi(bc.x0))
            for k in range(sys_params.N):
                R_N = R_N @ new_opts.so3_exp(dt * ome_N)
                ome_N = ome_N + dt * (I_inv @ (res["tau"][k] - np.cross(ome_N, I_mat @ ome_N)))
            phi_N = new_opts.so3_log(R_N)
            viol = (terminal_violation(res["r"][-1], bc.xf.r)
                    + terminal_violation(res["v"][-1], bc.xf.v)
                    + lie_attitude_violation(phi_N, bc.xf.phi)
                    + terminal_violation(ome_N, bc.xf.omega))
            row.update(cost=res["cost"], terminal_violation=float(viol),
                       runtime_s=res["runtime"],
                       gd_iterations=res["gd"]["iterations"],
                       gd_inner_solves=res["gd"]["inner_solves"],
                       gd_final_grad_norm=res["gd"]["final_grad_norm"],
                       gd_restarts=res["gd"]["restarts"],
                       gd_restart_J=";".join(f"{j:.6f}" for j in res["gd"]["restart_J"]))
            print(f"sc {sc_id}: cost {res['cost']:.6f}  viol {viol:.2e}  "
                  f"restarts {res['gd']['restarts']}  iters {res['gd']['iterations']}  "
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
