"""Sphere-inflation relaxation reference generator (user's method).

Every trajectory point of payload j carries a protective sphere of radius
rho_j = extent_j + 5 m, so disjoint same-step spheres == per-pair separation
delta_jl = ext_j + ext_l + 10. The spheres are inflated gradually
(continuation s: 0 -> 1, target clearance s*(delta_jl + headroom)); at each
level, position-based-dynamics sweeps resolve overlaps: each overlapping
same-step pair is pushed apart symmetrically along the gap direction, and the
displacement field from the nominal is smoothed along the path (endpoints
pinned) so pushes propagate over the whole trajectory instead of kinking
single waypoints.

Hand-offs measured on the relaxed references:
  tubes:   per-step balls eps_jk = min over partners (gap - delta_jl)/2,
           one convex SOCP per payload (decoupled);
  planes:  one-shot fixed normals from the relaxed gap directions, one joint
           convex solve (full timing freedom).
Benchmarks (seed 3): LB 235.0, per-pair alternation 265.2, corridors 339.5,
warp tubes 1456 / warp planes 432.

Usage: py inflate_oracle.py [--seed 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parent))
import route2_multi as r2  # noqa: E402
from corridor_oracle import alternate_pp, axis_prices  # noqa: E402

import cvxpy as cp  # noqa: E402


def relax(rbar, pairs, deltas_t, N, M, Winv2, max_sweeps=5000,
          alpha=0.25, sigma=1.5, tol=0.1):
    """While-loop PBD at FULL targets (event-driven, no ladder). Each
    violated same-step pair is resolved by the CHEAPEST displacement gaining
    the needed Euclidean gap: direction W^-2 g (fuel-metric push), scaled so
    each payload contributes need/2 of true gap. Termination = the exact
    Euclidean clearance check. Phase 1 smooths the displacement field from
    the nominal (endpoints pinned); phase 2 is projection-only polish."""
    x = [r.copy() for r in rbar]

    def sweep(smooth):
        worst = 0.0
        for (j, l) in pairs:
            d = x[j] - x[l]
            dist = np.linalg.norm(d, axis=1)
            for k in range(1, N):
                need = deltas_t[(j, l)] - dist[k]
                if need > 0:
                    worst = max(worst, need)
                    g = d[k] / dist[k] if dist[k] > 1e-9 else np.array([1.0, 0, 0])
                    pj = Winv2[j] * g
                    pl = Winv2[l] * g
                    x[j][k] += 0.5 * need * pj / float(g @ pj)
                    x[l][k] -= 0.5 * need * pl / float(g @ pl)
        if smooth:
            for j in range(M):
                u = x[j] - rbar[j]
                us = gaussian_filter1d(u, sigma=sigma, axis=0)
                u = (1 - alpha) * u + alpha * us
                u[0] = 0.0
                u[N] = 0.0
                x[j] = rbar[j] + u
        return worst

    sweeps = 0
    worst = np.inf
    while sweeps < 800 and worst >= 1.0:
        worst = sweep(True)
        sweeps += 1
    while sweeps < max_sweeps and worst >= tol:
        worst = sweep(False)
        sweeps += 1
    info = dict(sweeps=sweeps, resid=round(float(worst), 3))
    return (x if worst < tol else None), info


def run(seed, N, tf, M, headroom=60.0, iso=False):
    sc, psis, refs, pairs, gaps, _ = r2.crossing(seed, N, tf, M, n_points=8)
    LB = sum(r["fuel"] for r in refs)
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc.payloads]
    margin = sc.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    deltas_t = {(j, l): deltas[j][l] + headroom for (j, l) in pairs}
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]
    Ws = ([np.ones(3)] * M if iso
          else [axis_prices(sc, psis, refs, j) for j in range(M)])
    Winv2 = [1.0 / w ** 2 for w in Ws]
    out = dict(seed=seed, M=M, LB=round(LB, 2), headroom=headroom,
               iso=iso, axis_prices=[[round(v, 2) for v in w] for w in Ws])

    t0 = time.time()
    x, hist = relax(rbar, pairs, deltas_t, N, M, Winv2)
    out["relax_time"] = round(time.time() - t0, 1)
    out["relax_info"] = hist
    if x is None:
        out["status"] = "relaxation stalled"
        print(json.dumps(out), flush=True)
        return out
    mm = min(np.linalg.norm(x[j][k] - x[l][k]) - deltas[j][l]
             for (j, l) in pairs for k in range(N + 1))
    spd = max(float(np.max(np.linalg.norm(np.diff(x[j], axis=0), axis=1)))
              for j in range(M)) / sc.dt
    dev = max(float(np.max(np.linalg.norm(x[j] - rbar[j], axis=1)))
              for j in range(M))
    out["ref_min_margin"] = round(mm, 1)
    out["ref_max_speed"] = round(spd, 2)
    out["ref_max_deviation"] = round(dev, 1)
    print(json.dumps(out), flush=True)

    # hand-off A: per-step tubes, decoupled SOCPs
    CAP = 150.0
    t0 = time.time()
    fuel_tubes, hist_t, ok = 0.0, [], True
    for j in range(M):
        e = np.full(N + 1, CAP)
        for l in range(M):
            if l == j:
                continue
            gjl = np.linalg.norm(x[j] - x[l], axis=1) - deltas[j][l]
            e = np.minimum(e, 0.5 * gjl)
        r = cp.Variable((N + 1, 3))
        v = cp.Variable((N + 1, 3))
        F = cp.Variable((N, 3))
        cons = r2._dyn(sc, psis, r, v, F, j)
        for k in range(1, N):
            if e[k] < CAP:
                cons.append(cp.norm(r[k] - x[j][k]) <= e[k])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            ok = False
            break
        fuel_tubes += float(prob.value)
        hist_t.append(np.asarray(r.value))
    if ok:
        worst = min(min(np.linalg.norm(hist_t[j][k] - hist_t[l][k])
                        - deltas[j][l] for k in range(N + 1))
                    for (j, l) in pairs)
        out["J_tubes"] = round(fuel_tubes, 2)
        out["tubes_min_margin"] = round(worst, 1)
    else:
        out["J_tubes"] = None
    out["tubes_time"] = round(time.time() - t0, 1)

    # hand-off B: one-shot planes from relaxed gap directions
    t0 = time.time()
    oneshot = alternate_pp(sc, psis, pairs, [dict(r=x[j]) for j in range(M)],
                           deltas, iters=1)
    out["J_planes_oneshot"] = round(oneshot, 2) if oneshot else None
    out["planes_time"] = round(time.time() - t0, 1)

    t0 = time.time()
    alt = alternate_pp(sc, psis, pairs, refs, deltas)
    out["alternation_perpair"] = round(alt, 2) if alt else None
    out["alt_time"] = round(time.time() - t0, 1)
    out["status"] = "ok"
    print(json.dumps({k: out.get(k) for k in
                      ("status", "J_tubes", "tubes_min_margin",
                       "J_planes_oneshot", "alternation_perpair",
                       "ref_max_speed", "ref_max_deviation")}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[3])
    ap.add_argument("--iso", action="store_true")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--headroom", type=float, default=60.0)
    a = ap.parse_args()
    rows = [run(s, a.N, a.tf, a.M, a.headroom, iso=a.iso) for s in a.seeds]
    tag = "_iso" if a.iso else "_aniso"
    p = Path(__file__).with_name("results") / f"inflate_oracle{tag}.json"
    p.write_text(json.dumps(rows, indent=1))
    print(f"wrote {p}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
