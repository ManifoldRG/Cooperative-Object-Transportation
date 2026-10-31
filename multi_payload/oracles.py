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


# --- translational deconfliction + segment-safe ball corridors ---------------
#
# "gauss" and "chains" deconflict the translational references, then hand each
# payload a ball corridor whose radii are half the smallest surplus over delta
# of the deconflicted relative SEGMENTS touching each step (safe between steps,
# not only at them), and solve every payload with the same COT corridor solver
# as "waypoint" (run_oracle). The deconfliction code lives in studies/ (the
# exact code behind the notes' benchmarks) and is imported from there.

def _studies_path():
    import sys
    from pathlib import Path
    p = str(Path(__file__).with_name("studies"))
    if p not in sys.path:
        sys.path.insert(0, p)


def _segment_radii(x, delta, cap=150.0):
    """eps[j][k] = half the smallest surplus over delta of the relative
    segments k-1 -> k and k -> k+1 against every other payload; pinned ends 0.
    Positions inside the balls keep every segment >= delta apart."""
    _studies_path()
    from expand_method import _seg_closest
    M, N = len(x), len(x[0]) - 1
    eps = [np.full(N + 1, cap) for _ in range(M)]
    for j in range(M):
        for l in range(j + 1, M):
            dist, _ = _seg_closest(x[j] - x[l])
            half = np.full(N + 1, np.inf)
            half[:-1] = np.minimum(half[:-1], 0.5 * (dist - delta))
            half[1:] = np.minimum(half[1:], 0.5 * (dist - delta))
            eps[j] = np.minimum(eps[j], half)
            eps[l] = np.minimum(eps[l], half)
    for e in eps:
        e[0] = e[-1] = 0.0
    return eps


class _DeconflictOracle:
    """Shared Steps 1, 3-5: references, segment-safe corridors, COT solves.
    Subclasses implement _deconflict(sc, psis, rbar) -> (positions, info)."""

    def __init__(self, step5="cgd", rho=1e3, max_restarts=None, oracle_kwargs=None,
                 verbose=False):
        self.step5, self.rho, self.max_restarts = step5, rho, max_restarts
        self.oracle_kwargs = oracle_kwargs or {}
        self.verbose = verbose

    def solve(self, prob: OracleProblem) -> OracleResult:
        t0 = time.perf_counter()
        sc = prob.sc
        psis = psi_list(sc)
        refs = solve_all(sc, psis=psis, objective=prob.objective)
        rbar = [np.asarray(r["r"], float) for r in refs]
        x, info = self._deconflict(sc, psis, rbar)
        info["t_deconflict"] = time.perf_counter() - t0
        if x is None:
            return OracleResult("failed", runtime=time.perf_counter() - t0, info=info)
        eps = _segment_radii(x, sc.delta)
        worst = min(float(e[1:-1].min()) for e in eps)
        info["min_corridor_radius"] = worst
        if worst < -1e-6:
            return OracleResult("failed", runtime=time.perf_counter() - t0,
                                info=dict(info, reason=f"deconflicted segment below delta "
                                                       f"({2 * worst:.2f} m)"))
        eps = [np.maximum(e, 0.0) for e in eps]
        new_refs = []
        for j, r in enumerate(refs):
            v = np.vstack([np.diff(x[j], axis=0) / sc.dt, r["v"][-1:]])
            new_refs.append(dict(r, r=np.asarray(x[j], float), v=v))
        tau_refs = prob.tau_refs or [attitude_reference(sc, j, seed=prob.seed)["tau"]
                                     for j in range(sc.M)]
        orc = run_oracle(sc, new_refs, [dict() for _ in range(sc.M)], prob.occupied,
                         tau_refs, eps, rho=self.rho, method=self.step5,
                         budget_s=prob.budget_s, seed=prob.seed,
                         max_restarts=self.max_restarts, psis=psis, verbose=self.verbose,
                         objective=prob.objective, **self.oracle_kwargs)
        info.update(max_corridor_radius=float(max(np.max(e) for e in orc["eps"])),
                    refs=orc["refs"])
        if "reason" in orc:
            info["reason"] = orc["reason"]
        payloads = [None] * sc.M
        if orc["status"] != "infeasible":
            payloads = [dict(control=r["control"], tau=r["tau"], slack=r["slack"])
                        for r in orc["payloads"]]
        return OracleResult(orc["status"], payloads, time.perf_counter() - t0, info)


class GaussOracle(_DeconflictOracle):
    """Gaussian inflation (studies/benchmark_methods.relax_iso): every
    overlapping same-step pair is pushed apart by its exact gap, the
    displacement field is Gaussian-smoothed after each sweep, until every pair
    clears delta + h at every step. h = 30 m is the best continuous-safe
    headroom on the benchmark family (it must cover the between-step dips)."""

    name = "gauss"

    def __init__(self, h=30.0, **kw):
        super().__init__(**kw)
        self.h = h

    def _deconflict(self, sc, psis, rbar):
        _studies_path()
        from benchmark_methods import relax_iso
        M, N = sc.M, sc.N
        pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
        tgt = {p: sc.delta + self.h for p in pairs}
        x, sweeps, _ = relax_iso(rbar, pairs, tgt, N, M)
        info = dict(method="gauss", h=self.h, sweeps=sweeps)
        if x is None:
            return None, dict(info, reason="inflation did not clear")
        return [np.asarray(v, float) for v in x], info


class ChainsOracle(_DeconflictOracle):
    """Method of chains (notes Part I; studies/expand_method.py with
    --seg-chain --split-planes). Step 2: monotone expansion of sphere chains
    with projected Gaussian smoothing, segment check, and cluster-dilation
    failsafe; every same-step pair reaches delta + h and every segment clears
    delta. Step 3: split-plane rounds (fixed normals, midpoint re-centering,
    one translational fuel SOCP per payload per round; feasible, safe between
    steps, fuel non-increasing). The last round's positions are the corridor
    centers.

    Differences from the study runs: delta is the scenario's uniform clearance
    (the studies used per-pair clearances); the Rigid Link limit is
    max(link_rho * rho, longest nominal link) so the run stays on the
    scenario's grid N; and grid refinement (T <- 2T on a jam) is not done here,
    since the 6-DoF solve and the verification use the scenario's N; a jam is
    reported as status 'failed' (rerun with a larger N)."""

    name = "chains"

    def __init__(self, h=1.0, alpha=5.0, link_rho=1.0, split_rounds=6, **kw):
        super().__init__(**kw)
        self.h, self.alpha, self.link_rho = h, alpha, link_rho
        self.split_rounds = split_rounds

    def _deconflict(self, sc, psis, rbar):
        _studies_path()
        from chain_method import step_targets
        from expand_method import deconflict, split_plane_rounds
        M, N = sc.M, sc.N
        pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
        deltas = [[sc.delta] * M for _ in range(M)]
        rho = (sc.delta + self.h) / 2
        L = [max(self.link_rho * rho,
                 float(np.max(np.linalg.norm(np.diff(rbar[j], axis=0), axis=1))))
             for j in range(M)]
        tg = step_targets(rbar, pairs, deltas, L, self.h)
        P, rounds, resid, st = deconflict(rbar, pairs, tg, L, self.alpha, h=self.h,
                                          smooth=True, post_smooth=100, failsafe=True,
                                          Dm=deltas)
        info = dict(method="chains", h=self.h, alpha=self.alpha, rounds=rounds,
                    residual=round(float(resid), 3), **st)
        if P is None:
            return None, dict(info, reason="chains jammed on this grid (no refinement "
                                           "inside the oracle; rerun with a larger N)")
        hist = split_plane_rounds(sc, psis, P, pairs, tg, deltas, N, M,
                                  rounds=self.split_rounds, D=None)
        if not hist:
            return None, dict(info, reason="split-plane round 1 infeasible")
        info["split_fuel_by_round"] = [round(f, 4) for f, _ in hist]
        return [np.asarray(r, float) for r in hist[-1][1]], info


register("gauss", GaussOracle)
register("chains", ChainsOracle)
