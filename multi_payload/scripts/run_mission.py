"""Full mission: allocation + transport plan, docking, and animation.

  py multi_payload/scripts/run_mission.py --layout crossing --M 3 --agents 12 --allocation misocp
  py multi_payload/scripts/run_mission.py --out <dir> --render-only     (re-render from cache)

Writes to --out: scenario.json, summary.json, docking.json, mission.mp4, and
cache.pkl (pipeline + docking results, so the video can be re-rendered).
"""
import argparse
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multi_payload import animate, docking, pipeline, plots, scenario  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", type=Path)
    p.add_argument("--layout", choices=("crossing", "random"), default="crossing")
    p.add_argument("--M", type=int, default=3)
    p.add_argument("--points", type=int, default=6)
    p.add_argument("--agents", type=int, default=12)
    p.add_argument("--nu", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--N", type=int, default=None)
    p.add_argument("--tf", type=float, default=None)
    p.add_argument("--allocation", choices=("greedy", "misocp"), default="misocp")
    p.add_argument("--step5", choices=("cgd", "dgd"), default="cgd")
    p.add_argument("--objective", choices=("fuel", "energy"), default="fuel",
                   help="sum ||u|| (paper) or sum ||u||^2 (continuous thrust)")
    p.add_argument("--budget", type=float, default=60.0)
    p.add_argument("--dock-time", type=float, default=400.0, help="docking duration [s]")
    p.add_argument("--dock-steps", type=int, default=40)
    p.add_argument("--dock-noise", type=float, default=0.02, help="thrust execution error")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "multi_payload" / "results" / "mission")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "cache.pkl"

    if a.render_only:
        with open(cache, "rb") as fh:
            sc, out, starts, asg, dock, dock_kw = pickle.load(fh)
        dp = docking.DockingProblem(sc, starts, **dock_kw)
    else:
        sc = (scenario.load(a.scenario) if a.scenario else
              scenario.generate(a.layout, M=a.M, n_points=a.points, n_agents=a.agents,
                                seed=a.seed, nu_deg=a.nu, tf=a.tf, N=a.N))
        scenario.save(sc, a.out / "scenario.json")
        print(f"scenario: M={sc.M}, agents={sc.n_agents}, N={sc.N}, tf={sc.tf:.0f} s")
        out = pipeline.run(sc, allocation=a.allocation, step5=a.step5, budget_s=a.budget,
                           seed=a.seed, objective=a.objective)
        (a.out / "summary.json").write_text(json.dumps(
            pipeline.summary(sc, out), indent=1, default=lambda o: o.tolist()
            if hasattr(o, "tolist") else str(o)))
        plots.all_figures(sc, out, a.out)

        starts = docking.agent_starts(sc, seed=a.seed)
        dock_kw = dict(T_dock=a.dock_time, N_dock=a.dock_steps, objective=a.objective)
        dp = docking.DockingProblem(sc, starts, **dock_kw)
        asg, _ = docking.assign(dp, out["allocation"]["occupied"])
        dock = docking.simulate(dp, asg, noise=a.dock_noise, seed=a.seed)
        print(f"Docking: all captured {dock['all_captured']}, fuel {dock['fuel']:.2f}, "
              f"max pos err {max(x['pos_err'] for x in dock['agents']):.3f} m, "
              f"min agent-agent {dock['min_agent_agent']:.1f} m, "
              f"min agent-to-other-payload {dock['min_agent_other_payload']:.1f} m")
        (a.out / "docking.json").write_text(json.dumps(dict(
            assignment=[list(x) for x in asg], all_captured=dock["all_captured"],
            fuel=dock["fuel"], min_agent_agent=dock["min_agent_agent"],
            min_agent_other_payload=dock["min_agent_other_payload"],
            agents=[{k: v for k, v in x.items() if k not in ("r", "u")} for x in dock["agents"]],
            starts=starts.tolist(), **dock_kw), indent=1))
        with open(cache, "wb") as fh:
            pickle.dump((sc, out, starts, asg, dock, dock_kw), fh)

    animate.render(sc, out, dp, dock, asg, a.out / "mission.mp4", fps=a.fps)
    print(f"wrote {a.out / 'mission.mp4'}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
