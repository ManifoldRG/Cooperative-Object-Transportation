"""Common interface for Oracle-Problem methods, for side-by-side comparison.

An oracle receives a scenario and a FIXED allocation (which candidate points
each payload's agents occupy) and returns, for every payload, the body-frame
agent thrusts U (A_j, N, 3) and the body torque trajectory tau (N, 3). Every
method is then scored by the same independent verification (rollout through
the paper's dynamics, pairwise separation, certificate), so results are
comparable regardless of how a method plans.

To add a method (e.g. the fused GCS + gate-scheduling planner):

    class GCSOracle:
        name = "gcs"
        def solve(self, prob: OracleProblem) -> OracleResult: ...

    register("gcs", GCSOracle)

and run it with the benchmark driver (`--oracles waypoint gcs`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .attitude import attitude_reference
from .conflicts import repair, repair_stats
from .oracle import run_oracle
from .translation import psi_list, solve_all


@dataclass
class OracleProblem:
    sc: object                      # MultiPayloadScenario
    occupied: list                  # occupied point indices per payload
    objective: str = "fuel"         # "fuel" or "energy"
    budget_s: float = 60.0          # per-payload solver budget
    seed: int = 0
    tau_refs: list | None = None    # optional shared attitude references


@dataclass
class OracleResult:
    status: str                     # "ok" | "slack" | "infeasible" | "failed"
    payloads: list = field(default_factory=list)   # dicts with control, tau
    runtime: float = 0.0
    info: dict = field(default_factory=dict)


REGISTRY = {}


def register(name, factory):
    REGISTRY[name] = factory


def make(name, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown oracle {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


class WaypointRepairOracle:
    """Steps 1-6 of the notes: translation-only references, waypoint repair
    of closest-approach conflicts, ball corridors, per-payload COT solves,
    corridor relaxation."""

    name = "waypoint"

    def __init__(self, step5="cgd", eps0=0.0, rho=1e3, max_restarts=None,
                 repair_kwargs=None, oracle_kwargs=None, verbose=False):
        self.step5, self.eps0, self.rho = step5, eps0, rho
        self.max_restarts = max_restarts
        self.repair_kwargs = repair_kwargs or {}
        self.oracle_kwargs = oracle_kwargs or {}
        self.verbose = verbose

    def solve(self, prob: OracleProblem) -> OracleResult:
        t0 = time.perf_counter()
        sc = prob.sc
        psis = psi_list(sc)
        refs = solve_all(sc, psis=psis, objective=prob.objective)
        eps = [np.full(sc.N + 1, float(self.eps0)) for _ in range(sc.M)]
        for e in eps:
            e[0] = e[-1] = 0.0
        waypoints = [dict() for _ in range(sc.M)]
        refs, waypoints, rlog, ok = repair(sc, refs, waypoints, eps, psis=psis,
                                           objective=prob.objective, **self.repair_kwargs)
        t_rep = time.perf_counter() - t0
        info = dict(repair=repair_stats(rlog, ok), t_references=t_rep)
        if not ok:
            return OracleResult("failed", runtime=time.perf_counter() - t0,
                                info=dict(info, reason="repair did not converge"))
        tau_refs = prob.tau_refs or [attitude_reference(sc, j, seed=prob.seed)["tau"]
                                     for j in range(sc.M)]
        orc = run_oracle(sc, refs, waypoints, prob.occupied, tau_refs, eps, rho=self.rho,
                         method=self.step5, budget_s=prob.budget_s, seed=prob.seed,
                         max_restarts=self.max_restarts, psis=psis, verbose=self.verbose,
                         objective=prob.objective, **self.oracle_kwargs)
        attempts = [e for e in orc["log"] if "payload" in e]
        rerepairs = [e for e in orc["log"] if "repair" in e]
        info.update(
            corridor_expansions=int(sum(e["attempt"] > 0 for e in attempts)),
            re_repairs=len(rerepairs),
            re_repair_converged=all(e["converged"] for e in rerepairs),
            max_corridor_radius=float(max(np.max(e) for e in orc["eps"])),
            waypoints=int(sum(len(w) for w in orc["waypoints"])),
            refs=orc["refs"])
        if "reason" in orc:
            info["reason"] = orc["reason"]
        payloads = [None] * sc.M
        if orc["status"] != "infeasible":
            payloads = [dict(control=r["control"], tau=r["tau"], slack=r["slack"])
                        for r in orc["payloads"]]
        return OracleResult(orc["status"], payloads, time.perf_counter() - t0, info)


register("waypoint", WaypointRepairOracle)
