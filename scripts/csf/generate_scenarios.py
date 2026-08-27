#!/usr/bin/env python3
"""Generate and save Monte Carlo scenarios for reproducible sensitivity studies.
generating separately so all tasks submitted can use the same scenario for different hyper-param combinations

"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path

# scripts/csf/ -> scripts/ -> repo root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from spacecraft_libraries.evaluation.comparison import random_scenario_generator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int,  default=3,
                    help="Number of scenarios PER thrust angle.")
    p.add_argument("--thrust-angles-deg", type=float, nargs="+", default=[90.0],
                    help="One or more thrust-angle cone half-angles in "
                         "degrees (SystemParams.nu). --n scenarios are "
                         "generated at each. Default [90.0] preserves the "
                         "old unwired behavior (random_scenario_generator's "
                         "own default, pi/2 rad) — pass explicit values for "
                         "the ablation sweep, e.g. 1 2 2.5 5 15 60.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed",   type=int,  default=None)
    p.add_argument("--fixed-agents-num", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Path(f"results/scenarios_{ts}.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_total = args.n * len(args.thrust_angles_deg)
    print(f"Generating {args.n} scenarios x {len(args.thrust_angles_deg)} "
          f"thrust angles ({args.thrust_angles_deg} deg) = {n_total} total", flush=True)
    if args.seed is not None:
        print(f"Base seed: {args.seed} (scenario k -> seed {args.seed}+k, "
              f"k running continuously across all angles)", flush=True)
    if args.fixed_agents_num > 0:
        print(f"Fixed agent count: {args.fixed_agents_num}", flush=True)

    log = []
    k = 0  # continuous counter across all angles, so seeds/scenario_ids never collide
    for angle_deg in args.thrust_angles_deg:
        thrust_angle = np.radians(angle_deg)
        for _ in range(args.n):
            scenario_seed = None if args.seed is None else args.seed + k
            sys_p, bc, eps = random_scenario_generator(
                fixed_agents_num=args.fixed_agents_num, seed=scenario_seed,
                thrust_angle=thrust_angle)
            print(f"  Scenario {k+1} (nu={angle_deg:g} deg): N={sys_p.N}, "
                  f"n_agents={len(sys_p.rs)}, tf={bc.tf:.1f}s, a={sys_p.a:.3e}, "
                  f"e={sys_p.e:.3f}", flush=True)
            log.append({
                "scenario_id": k + 1,
                "seed": scenario_seed,
                "thrust_angle_deg": angle_deg,
                "mu": sys_p.mu, "a": sys_p.a, "e": sys_p.e, "nu": sys_p.nu,
                "m": sys_p.m, "I_diag": np.diag(sys_p.I).tolist(),
                "N": sys_p.N, "tf": bc.tf, "epsilon": eps,
                "rs": [r.tolist() for r in sys_p.rs],
                "n_agents": len(sys_p.rs),
                "x0_r":  bc.x0.r.tolist(),  "x0_v":  bc.x0.v.tolist(),
                "x0_phi": bc.x0.phi.tolist(), "x0_omega": bc.x0.omega.tolist(),
                "xf_r":  bc.xf.r.tolist(),  "xf_v":  bc.xf.v.tolist(),
                "xf_phi": bc.xf.phi.tolist(), "xf_omega": bc.xf.omega.tolist(),
            })
            k += 1

    with args.output.open("w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved {n_total} scenarios -> {args.output}")


if __name__ == "__main__":
    main()