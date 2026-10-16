"""Run the multi-payload approach on one scenario and save results + figures.

Examples:
  py multi_payload/scripts/run_demo.py --layout crossing --M 3 --points 6 --agents 12 --nu 60
  py multi_payload/scripts/run_demo.py --layout random --M 4 --allocation misocp --step5 dgd
  py multi_payload/scripts/run_demo.py --scenario path/to/scenario.json

Writes <out>/scenario.json, summary.json and trajectories/separation/allocation PNGs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multi_payload import pipeline, plots, scenario  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=Path, help="load a saved scenario JSON instead of generating")
    p.add_argument("--layout", choices=("crossing", "random"), default="crossing")
    p.add_argument("--M", type=int, default=3, help="number of payloads")
    p.add_argument("--points", type=int, default=6, help="open attachment points per payload")
    p.add_argument("--agents", type=int, default=None, help="agents in the swarm (default 3M)")
    p.add_argument("--nu", type=float, default=60.0, help="thrust-cone half-angle [deg]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=None, help="time steps (default: paper rule tf/10)")
    p.add_argument("--tf", type=float, default=None, help="horizon [s] (default U(300,900))")
    p.add_argument("--ring", type=float, default=500.0, help="layout radius [m]")
    p.add_argument("--allocation", choices=("greedy", "misocp"), default="greedy")
    p.add_argument("--step5", choices=("cgd", "dgd"), default="cgd")
    p.add_argument("--objective", choices=("fuel", "energy"), default="fuel",
                   help="sum ||u|| (paper) or sum ||u||^2 (continuous thrust)")
    p.add_argument("--budget", type=float, default=60.0, help="COT budget per payload [s]")
    p.add_argument("--restarts", type=int, default=None, help="max GD starts (default: until budget)")
    p.add_argument("--eps", type=float, default=0.0, help="initial corridor radius [m]")
    p.add_argument("--rho", type=float, default=1e3, help="corridor slack penalty")
    p.add_argument("--out", type=Path, default=ROOT / "multi_payload" / "results" / "demo")
    a = p.parse_args()

    if a.scenario:
        sc = scenario.load(a.scenario)
    else:
        sc = scenario.generate(a.layout, M=a.M, n_points=a.points, n_agents=a.agents,
                               seed=a.seed, nu_deg=a.nu, ring_radius=a.ring, tf=a.tf, N=a.N)
    a.out.mkdir(parents=True, exist_ok=True)
    scenario.save(sc, a.out / "scenario.json")
    print(f"scenario: M={sc.M}, agents={sc.n_agents}, N={sc.N}, tf={sc.tf:.0f} s, "
          f"nu={a.nu} deg, delta={sc.delta:.1f} m")

    out = pipeline.run(sc, allocation=a.allocation, step5=a.step5, budget_s=a.budget,
                       eps0=a.eps, rho=a.rho, seed=a.seed, max_restarts=a.restarts,
                       objective=a.objective)
    summ = pipeline.summary(sc, out)
    tmp = a.out / ".summary.json.tmp"
    tmp.write_text(json.dumps(summ, indent=1, default=lambda o: o.tolist()
                              if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, a.out / "summary.json")
    if "verify" in out:
        plots.all_figures(sc, out, a.out)
    print(f"wrote {a.out}")
    sys.stdout.flush()
    os._exit(0)  # skip CasADi interpreter teardown (see scripts/baseline_fuel)


if __name__ == "__main__":
    main()
