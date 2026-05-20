"""Stress test: MPPI at N=100, 5 agents, 50 iterations x 5 samples per island.

Centralized MPPI runs once; decentralized MPPI runs one island per agent (5),
each with its own seed, then consensuses to pick the best. Writes a CSV with
per-island progress and per-method summary.
"""
from pathlib import Path
import csv
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from spacecraft_libraries.data_structures import (
    BoundaryConditions,
    StateVector,
    SystemParams,
)
from spacecraft_libraries.evaluation.metrics import (
    quaternion_aware_violation,
    terminal_violation,
)
from spacecraft_libraries.solvers import (
    solve_centralized_mppi,
    solve_decentralized_mppi,
)


def tough_scenario_n100_5agents():
    rs = [
        np.array([0.5, 1.0, 1.5]),
        np.array([0.0, 0.5, 2.0]),
        np.array([-0.5, 1.0, -1.5]),
        np.array([1.0, -0.5, 1.0]),
        np.array([-1.0, 0.75, 1.5]),
    ]
    sys_params = SystemParams(
        mu=3.98e14,
        a=8e6,
        e=0.2,
        nu=np.pi / 4,
        I=1000 * np.diag([1, 2, 3]),
        m=100,
        rs=rs,
        N=100,
    )
    bc = BoundaryConditions(
        x0=StateVector(
            r=np.array([0.0, 0.0, 0.0]),
            v=np.array([0.0, 0.0, 0.0]),
            eps=np.array([0.0, 0.0, 0.0, 1.0]),
            omega=np.array([0.0, 0.0, 0.0]),
        ),
        xf=StateVector(
            r=np.array([5.0, 5.0, 5.0]),
            v=np.array([0.0, 0.0, 0.0]),
            eps=np.array([0.5, 0.5, 0.5, 0.5]),
            omega=np.array([0.0, 0.0, 0.0]),
        ),
        tf=50.0,
    )
    return sys_params, bc, 1e-5


def violation_of(result, bc):
    s = result["trajectory"].states[-1]
    return float(
        terminal_violation(s.r, bc.xf.r)
        + terminal_violation(s.v, bc.xf.v)
        + quaternion_aware_violation(s.eps, bc.xf.eps)
        + terminal_violation(s.omega, bc.xf.omega)
    )


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"mppi_stress_n100_5agents_{stamp}.csv"
    json_path = out_dir / f"mppi_stress_n100_5agents_{stamp}.json"

    sys_params, bc, eps = tough_scenario_n100_5agents()
    print(f"[scenario] N={sys_params.N}  num_agents={len(sys_params.rs)}  tf={bc.tf}")
    print(f"[mppi] iterations=10  samples=5  sigma=1e-1  lambda=1.0")

    rows = []

    print("\n[1/2] centralized_mppi  (1 island x 10 iter x 5 samples = 51 inner-IPOPT calls)")
    t0 = time.perf_counter()
    cent = solve_centralized_mppi(
        sys_params, bc, eps,
        n_iter=10, n_samples=5, sigma=1e-1, lambda_=1.0, seed=42,
    )
    cent_time = time.perf_counter() - t0
    cent_viol = violation_of(cent, bc)
    print(f"        cost={cent['cost']:.4e}  term_viol={cent_viol:.3e}  runtime={cent_time:.1f}s")
    print(f"        best_cost_per_iter (first 5): {cent['mppi']['best_cost_per_iter'][:5]}")
    print(f"        best_cost_per_iter (last 5):  {cent['mppi']['best_cost_per_iter'][-5:]}")
    print(f"        failed_sample_fraction={cent['mppi']['failed_sample_fraction']:.3f}")
    rows.append({
        "method": "centralized_mppi",
        "cost": cent["cost"],
        "terminal_violation": cent_viol,
        "runtime_s": cent_time,
        "n_islands": 1,
    })

    print("\n[2/2] decentralized_mppi  (5 islands x 10 iter x 5 samples = 255 inner-IPOPT calls)")
    t0 = time.perf_counter()
    dec = solve_decentralized_mppi(
        sys_params, bc, eps,
        n_iter=10, n_samples=5, sigma=1e-1, lambda_=1.0, base_seed=42,
    )
    dec_time = time.perf_counter() - t0
    dec_viol = violation_of(dec, bc)
    print(f"        cost={dec['cost']:.4e}  term_viol={dec_viol:.3e}  runtime={dec_time:.1f}s")
    print(f"        per_island_best_costs: {dec['mppi']['per_island_best_costs']}")
    rows.append({
        "method": "decentralized_mppi",
        "cost": dec["cost"],
        "terminal_violation": dec_viol,
        "runtime_s": dec_time,
        "n_islands": 5,
    })

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "cost", "terminal_violation", "runtime_s", "n_islands"])
        writer.writeheader()
        writer.writerows(rows)

    diagnostics = {
        "scenario": {"N": sys_params.N, "num_agents": len(sys_params.rs), "tf": bc.tf},
        "mppi_config": {"iterations": 10, "samples": 5, "sigma": 1e-3, "lambda": 1.0},
        "centralized": {
            "cost": cent["cost"],
            "terminal_violation": cent_viol,
            "runtime_s": cent_time,
            "best_cost_per_iter": cent["mppi"]["best_cost_per_iter"],
            "failed_sample_fraction": cent["mppi"]["failed_sample_fraction"],
        },
        "decentralized": {
            "cost": dec["cost"],
            "terminal_violation": dec_viol,
            "runtime_s": dec_time,
            "per_island_best_costs": dec["mppi"]["per_island_best_costs"],
            "per_island_history": dec["mppi"]["per_island_history"],
        },
    }
    with json_path.open("w") as f:
        json.dump(diagnostics, f, indent=2, default=float)

    print(f"\nSaved CSV  -> {csv_path}")
    print(f"Saved JSON -> {json_path}")


if __name__ == "__main__":
    main()
