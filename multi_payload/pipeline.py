"""End-to-end run of the proposed approach on one multi-payload scenario."""
from __future__ import annotations

import copy
import time

import numpy as np

from .allocation import build_allocators, greedy, misocp
from .attitude import attitude_reference, required_wrenches
from .conflicts import repair
from .oracle import run_oracle
from .translation import psi_list, solve_all
from .verify import verify


def run(sc, allocation="greedy", step5="cgd", budget_s=60.0, eps0=0.0, rho=1e3,
        seed=0, max_restarts=None, misocp_time=300.0, verbose=True):
    say = print if verbose else (lambda *a, **k: None)
    timing = {}
    t0 = time.perf_counter()
    psis = psi_list(sc)

    # Step 1: allocation-independent translational references and lower bound
    refs0 = solve_all(sc, psis=psis)
    if not all(r["ok"] for r in refs0):
        raise RuntimeError("translational reference infeasible")
    lower_bound = float(sum(r["fuel"] for r in refs0))
    say(f"Step 1: references, lower bound {lower_bound:.4f}")

    # Steps 2-3: conflict detection and waypoint repair (corridor radii eps0)
    eps = [np.full(sc.N + 1, float(eps0)) for _ in range(sc.M)]
    for e in eps:
        e[0] = e[-1] = 0.0
    refs = copy.deepcopy(refs0)
    waypoints = [dict() for _ in range(sc.M)]
    refs, waypoints, repair_log, ok = repair(sc, refs, waypoints, eps, psis=psis)
    if not ok:
        raise RuntimeError("waypoint repair did not converge")
    timing["references"] = time.perf_counter() - t0
    say(f"Steps 2-3: {len(repair_log)} repair(s); reference fuel "
        f"{sum(r['fuel'] for r in refs):.4f}")

    # Allocation: attitude references -> required wrenches -> surrogate
    t1 = time.perf_counter()
    att = [attitude_reference(sc, j, seed=seed) for j in range(sc.M)]
    W = [required_wrenches(refs[j], att[j]) for j in range(sc.M)]
    allocators = build_allocators(sc, W)
    alloc = (misocp(allocators, sc.n_agents, time_limit=misocp_time)
             if allocation == "misocp" else greedy(allocators, sc.n_agents))
    timing["allocation"] = time.perf_counter() - t1
    say(f"Allocation ({alloc['method']}): {alloc['occupied']}, case {alloc['case']}, "
        f"J_hat {alloc['J_hat']:.4f}, residuals {np.round(alloc['residual'], 4).tolist()}")

    # Steps 4-6: corridors and per-payload COT solves
    t2 = time.perf_counter()
    orc = run_oracle(sc, refs, waypoints, alloc["occupied"], [a["tau"] for a in att], eps,
                     rho=rho, method=step5, budget_s=budget_s, seed=seed,
                     max_restarts=max_restarts, psis=psis, verbose=verbose)
    timing["oracle"] = time.perf_counter() - t2
    if orc["status"] == "infeasible":
        say("Oracle: infeasible")
        return dict(status="infeasible", lower_bound=lower_bound, allocation=alloc,
                    repair_log=repair_log, oracle_log=orc["log"], timing=timing)

    ver = verify(sc, alloc["occupied"], orc, lower_bound, alloc["J_hat"])
    timing["total"] = time.perf_counter() - t0
    say(f"Verify: J_final {ver['J_final']:.4f}, min separation {ver['min_separation']:.2f} "
        f"(delta {sc.delta:.2f}), max V {max(p['V'] for p in ver['payloads']):.2e}, "
        f"certificate {ver['certificate_ok']}")
    return dict(status=orc["status"], lower_bound=lower_bound, allocation=alloc,
                refs0=refs0, refs=orc["refs"], waypoints=orc["waypoints"], eps=orc["eps"],
                attitude=att, wrenches=W, oracle=orc, verify=ver,
                repair_log=repair_log, timing=timing)


def summary(sc, out) -> dict:
    """JSON-serializable summary of a run."""
    a = out["allocation"]
    s = dict(status=out["status"], M=sc.M, n_agents=sc.n_agents, N=sc.N, tf=sc.tf,
             nu_deg=float(np.rad2deg(sc.nu)), delta=sc.delta, meta=sc.meta,
             lower_bound=out["lower_bound"], timing=out["timing"],
             repairs=len(out["repair_log"]),
             allocation=dict(method=a["method"], occupied=a["occupied"], case=a["case"],
                             J_hat=a["J_hat"], residual=a["residual"], gamma=a["gamma"],
                             **{k: a[k] for k in ("stage1_max_residual", "mip_case") if k in a}))
    if "verify" in out:
        v = out["verify"]
        s.update(J_final=v["J_final"], min_separation=v["min_separation"],
                 collision_free=v["collision_free"], all_terminal_ok=v["all_terminal_ok"],
                 certificate_ok=v["certificate_ok"], payloads=v["payloads"],
                 attitude_ref_terminal_error=[x["terminal_error"] for x in out["attitude"]],
                 oracle_log=out["oracle"]["log"])
    return s
