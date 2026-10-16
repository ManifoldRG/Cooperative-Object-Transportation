"""Side-by-side evaluation of Oracle methods on a shared allocation.

run_task(task) generates a scenario, computes ONE allocation (the notes'
allocation rule, independent of the oracle being tested), runs the requested
oracle, and scores it with the same independent verification for every
method. The returned row is flat and JSON-serializable.
"""
from __future__ import annotations

import time
import traceback

import numpy as np

from . import oracles, scenario
from .allocation import build_allocators, greedy, misocp
from .attitude import attitude_reference, required_wrenches
from .conflicts import repair
from .translation import psi_list, solve_all
from .verify import verify


def allocate(sc, rule="greedy", objective="fuel", seed=0, misocp_time=300.0):
    """Notes' allocation (references -> required wrenches -> surrogate).
    Returns dict(occupied, J_hat, case, tau_refs, ell, time)."""
    t0 = time.perf_counter()
    psis = psi_list(sc)
    refs0 = solve_all(sc, psis=psis, objective=objective)
    refs = [dict(r) for r in refs0]
    refs, _, _, ok = repair(sc, refs, [dict() for _ in range(sc.M)], psis=psis,
                            objective=objective)
    att = [attitude_reference(sc, j, seed=seed) for j in range(sc.M)]
    W = [required_wrenches(refs[j], att[j]) for j in range(sc.M)]
    al = build_allocators(sc, W, objective=objective)
    alloc = (misocp(al, sc.n_agents, time_limit=misocp_time) if rule == "misocp"
             else greedy(al, sc.n_agents))
    return dict(occupied=alloc["occupied"], J_hat=alloc["J_hat"], case=alloc["case"],
                residual=alloc["residual"], tau_refs=[a["tau"] for a in att],
                ell=[r["cost"] for r in refs0], repair_converged=ok,
                time=time.perf_counter() - t0)


def lower_bound(ell, occupied, objective):
    return float(sum(e if objective == "fuel" else e / max(len(o), 1)
                     for e, o in zip(ell, occupied)))


def run_task(task):
    """task keys: layout, M, points, agents, nu, seed, oracle, allocation,
    objective, budget, (optional) N, tf, ring, oracle_kwargs."""
    row = {k: task.get(k) for k in ("layout", "M", "points", "agents", "nu", "seed",
                                    "oracle", "allocation", "objective", "budget")}
    try:
        sc = scenario.generate(task["layout"], M=task["M"], n_points=task["points"],
                               n_agents=task["agents"], seed=task["seed"], nu_deg=task["nu"],
                               ring_radius=task.get("ring", 500.0), tf=task.get("tf"),
                               N=task.get("N"))
        row.update(N=sc.N, tf=sc.tf, delta=sc.delta)
        al = allocate(sc, task["allocation"], task["objective"], task["seed"])
        lb = lower_bound(al["ell"], al["occupied"], task["objective"])
        row.update(alloc_case=al["case"], J_hat=al["J_hat"], lower_bound=lb,
                   alloc_time=al["time"], occupied=al["occupied"],
                   max_residual=float(max(al["residual"])))
        oracle = oracles.make(task["oracle"], **task.get("oracle_kwargs", {}))
        prob = oracles.OracleProblem(sc, al["occupied"], task["objective"], task["budget"],
                                     task["seed"], al["tau_refs"])
        res = oracle.solve(prob)
        info = {k: v for k, v in res.info.items() if k != "refs"}
        row.update(status=res.status, oracle_time=res.runtime,
                   **{f"info_{k}": v for k, v in _flatten(info).items()})
        if res.status in ("ok", "slack") and all(p is not None for p in res.payloads):
            v = verify(sc, al["occupied"], dict(payloads=res.payloads), lb, al["J_hat"],
                       objective=task["objective"])
            row.update(J_final=v["J_final"], total_fuel=v["total_fuel"],
                       total_energy=v["total_energy"], min_separation=v["min_separation"],
                       collision_free=v["collision_free"], terminal_ok=v["all_terminal_ok"],
                       max_V=float(max(p["V"] for p in v["payloads"])),
                       max_slack=float(max(p["max_slack"] for p in v["payloads"])),
                       certificate_ok=v["certificate_ok"],
                       gap_ratio=v["J_final"] / lb if lb > 0 else None)
        row["success"] = bool(row.get("collision_free") and row.get("terminal_ok"))
    except Exception as exc:  # recorded, not raised: one bad task must not stop a sweep
        row.update(status="error", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc(limit=5), success=False)
    return row


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}_"))
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[f"{prefix}{k}"] = v
        elif isinstance(v, (np.floating, np.integer)):
            out[f"{prefix}{k}"] = v.item()
    return out
