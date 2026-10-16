"""Steps 2-3: conflict detection and waypoint repair on the references.

Payloads j and l conflict at step k if
    ||rbar_j,k - rbar_l,k|| < delta + eps_j,k + eps_l,k + (dt/2) ||vbar_j,k - vbar_l,k||,
where eps are the corridor radii of Step 4 and the last term is a margin for
separation between time steps (both problems share the time grid).

Repair takes the worst conflict and displaces the conflicting positions apart.
Candidates combine a push direction (the separation axis and versions tilted
45 degrees toward four perpendicular directions) with a split (move j, move l,
or half each), each pushed exactly far enough to clear the required
separation. They are scored by re-solving Step 1 with the displaced position
as a waypoint, which gives the exact cost increase (the first-order multiplier
of a waypoint placed on the optimal reference is zero, so it carries no
information), and the cheapest is applied. Only the affected payloads are
re-solved. A conflict that recurs is pushed with a geometrically growing
margin, so the loop cannot cycle at a fixed displacement; non-convergence
within max_iter is reported, with recurrences, by repair_stats.
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


def _separation_axis(refs, j, l, k):
    """Unit vector from l to j at step k (perpendicular to the relative
    velocity if the two references coincide)."""
    n = refs[j]["r"][k] - refs[l]["r"][k]
    if np.linalg.norm(n) < 1e-9:
        dv = refs[j]["v"][k] - refs[l]["v"][k]
        n = np.cross(dv, [0.0, 0.0, 1.0])
        if np.linalg.norm(n) < 1e-9:
            n = np.cross(dv, [0.0, 1.0, 0.0])
    return n / np.linalg.norm(n)


def push_directions(n, n_tilt=4, tilt_deg=45.0):
    """Candidate relative push directions: the separation axis n and n tilted
    by tilt_deg toward n_tilt evenly spaced perpendicular directions."""
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    b1 = np.cross(n, a)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1)
    th = np.deg2rad(tilt_deg)
    dirs = [n]
    for i in range(n_tilt):
        phi = 2 * np.pi * i / n_tilt
        e = np.cos(phi) * b1 + np.sin(phi) * b2
        dirs.append(np.cos(th) * n + np.sin(th) * e)
    return dirs


def push_length(D, d, need):
    """Smallest alpha >= 0 with ||D + alpha d|| >= need (d unit)."""
    Dd = float(D @ d)
    disc = Dd ** 2 - float(D @ D) + need ** 2
    return max(0.0, -Dd + np.sqrt(max(disc, 0.0)))


def repair(sc, refs, waypoints, eps=None, max_iter=100, push_margin=1.0,
           psis=None, log=None, objective=None, n_tilt=4, tilt_deg=45.0,
           escalate=2.0):
    """Repair conflicts in place. Returns (refs, waypoints, log, converged).

    Each iteration takes the worst conflict (j, l, k) and scores every
    combination of a push direction (separation axis and tilted variants,
    n_tilt=0 for the axis only) with a split (move j, move l, both halves),
    each pushed exactly far enough to clear the required separation, by
    re-solving Step 1. The cheapest is applied.

    Oscillation guard: if the same conflict (j, l, k) comes back, its push
    margin is multiplied by `escalate` each time it recurs, so repeated
    conflicts are pushed progressively farther apart instead of cycling.
    Stop after max_iter iterations; repair_stats(log, converged) summarizes.
    """
    psis = psi_list(sc) if psis is None else psis
    eps = zero_eps(sc) if eps is None else eps
    objective = objective or refs[0].get("objective", "fuel")
    log = [] if log is None else log
    seen = {}
    for _ in range(max_iter):
        cf = detect(sc, refs, eps)
        if not cf:
            return refs, waypoints, log, True
        gap, j, l, k, need = cf[0]
        if k in (0, sc.N):
            raise ValueError(f"payloads {j},{l} conflict at a boundary condition (k={k})")
        key = (j, l, k)
        seen[key] = seen.get(key, 0) + 1
        margin = push_margin * escalate ** (seen[key] - 1)
        D = refs[j]["r"][k] - refs[l]["r"][k]
        n = _separation_axis(refs, j, l, k)
        scored = []
        for di, d in enumerate(push_directions(n, n_tilt, tilt_deg)):
            alpha = push_length(D, d, need + margin)
            for label, share in ((f"move {j}", {j: 1.0, l: 0.0}),
                                 (f"move {l}", {j: 0.0, l: 1.0}),
                                 ("split", {j: 0.5, l: 0.5})):
                new, cost, ok = {}, 0.0, True
                for p, sgn in ((j, 1.0), (l, -1.0)):
                    if share[p] == 0.0:
                        continue
                    w = dict(waypoints[p])
                    w[k] = refs[p]["r"][k] + sgn * share[p] * alpha * d
                    sol = solve_reference(sc, p, w, psis, objective=objective)
                    if not sol["ok"]:
                        ok = False
                        break
                    new[p] = sol
                    cost += sol["cost"] - refs[p]["cost"]
                if ok:
                    scored.append((cost, f"{label}, dir {di}", new, alpha))
        if not scored:
            raise RuntimeError(f"no feasible repair for conflict {j},{l} at k={k}")
        cost, label, new, alpha = min(scored, key=lambda s: s[0])
        for p, sol in new.items():
            refs[p] = sol
            waypoints[p] = sol["waypoints"]
        log.append(dict(pair=(j, l), k=k, gap=gap, choice=label, push=alpha,
                        cost_increase=cost, recurrence=seen[key], margin=margin,
                        n_options=len(scored)))
    return refs, waypoints, log, not detect(sc, refs, eps)


def repair_stats(log, converged):
    """Iterations, recurrences and an oscillation flag for a repair log."""
    rec = [e.get("recurrence", 1) for e in log if "pair" in e]
    return dict(iterations=len(rec), converged=bool(converged),
                recurring_conflicts=int(sum(r > 1 for r in rec)),
                max_recurrence=int(max(rec, default=0)),
                oscillation=bool(not converged and any(r > 1 for r in rec)),
                cost_increase=float(sum(e.get("cost_increase", 0.0) for e in log if "pair" in e)))
