"""Steps 2-3: conflict detection and waypoint repair on the references.

Payloads j and l conflict at step k if
    ||rbar_j,k - rbar_l,k|| < delta + eps_j,k + eps_l,k + (dt/2) ||vbar_j,k - vbar_l,k||,
where eps are the corridor radii of Step 4 and the last term is a margin for
separation between time steps (both problems share the time grid).

Repair takes the worst conflict and displaces the conflicting positions apart
along their separation direction. Candidate displacements (move j, move l,
split) are scored by re-solving Step 1 with the displaced position as a
waypoint, which gives the exact cost increase: the first-order multiplier of a
waypoint placed on the optimal reference is zero, so it carries no
information. Only the affected payloads are re-solved.
"""
from __future__ import annotations

import numpy as np

from .translation import psi_list, solve_reference


def required_separation(sc, refs, j, l, eps):
    dv = np.linalg.norm(refs[j]["v"] - refs[l]["v"], axis=1)
    return sc.delta + eps[j] + eps[l] + 0.5 * sc.dt * dv


def detect(sc, refs, eps):
    """All conflicts as (gap, j, l, k, required), worst first."""
    out = []
    for j in range(sc.M):
        for l in range(j + 1, sc.M):
            d = np.linalg.norm(refs[j]["r"] - refs[l]["r"], axis=1)
            need = required_separation(sc, refs, j, l, eps)
            for k in np.nonzero(d < need)[0]:
                out.append((float(need[k] - d[k]), j, l, int(k), float(need[k])))
    return sorted(out, reverse=True)


def zero_eps(sc):
    return [np.zeros(sc.N + 1) for _ in range(sc.M)]


def repair(sc, refs, waypoints, eps=None, max_iter=100, push_margin=1.0,
           psis=None, log=None, objective=None):
    """Repair conflicts in place. Returns (refs, waypoints, log, converged)."""
    psis = psi_list(sc) if psis is None else psis
    eps = zero_eps(sc) if eps is None else eps
    objective = objective or refs[0].get("objective", "fuel")
    log = [] if log is None else log
    for _ in range(max_iter):
        cf = detect(sc, refs, eps)
        if not cf:
            return refs, waypoints, log, True
        gap, j, l, k, need = cf[0]
        if k in (0, sc.N):
            raise ValueError(f"payloads {j},{l} conflict at a boundary condition (k={k})")
        n = refs[j]["r"][k] - refs[l]["r"][k]
        if np.linalg.norm(n) < 1e-9:   # head-on: push perpendicular to the relative velocity
            dv = refs[j]["v"][k] - refs[l]["v"][k]
            n = np.cross(dv, [0.0, 0.0, 1.0])
            if np.linalg.norm(n) < 1e-9:
                n = np.cross(dv, [0.0, 1.0, 0.0])
        n = n / np.linalg.norm(n)
        push = gap + push_margin
        options = {f"move {j}": {j: push, l: 0.0},
                   f"move {l}": {j: 0.0, l: push},
                   "split": {j: push / 2, l: push / 2}}
        scored = []
        for label, amt in options.items():
            new, cost, ok = {}, 0.0, True
            for p, sgn in ((j, 1.0), (l, -1.0)):
                if amt[p] == 0.0:
                    continue
                w = dict(waypoints[p])
                w[k] = refs[p]["r"][k] + sgn * amt[p] * n
                sol = solve_reference(sc, p, w, psis, objective=objective)
                if not sol["ok"]:
                    ok = False
                    break
                new[p] = sol
                cost += sol["cost"] - refs[p]["cost"]
            if ok:
                scored.append((cost, label, new))
        if not scored:
            raise RuntimeError(f"no feasible repair for conflict {j},{l} at k={k}")
        cost, label, new = min(scored, key=lambda s: s[0])
        for p, sol in new.items():
            refs[p] = sol
            waypoints[p] = sol["waypoints"]
        log.append(dict(pair=(j, l), k=k, gap=gap, choice=label, cost_increase=cost,
                        options={lab: c for c, lab, _ in scored}))
    return refs, waypoints, log, not detect(sc, refs, eps)
