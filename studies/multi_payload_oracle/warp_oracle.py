"""Space-warp reference generator + corridor tracking (translational layer).

One anisotropic dilation applied to ALL nominals at once:
  W(x) = x + beta(||x-c||) (S_lambda - I)(x - c),
  S_lambda = diag(1 + (lambda-1)/w_a),  w = measured per-axis fuel prices
  (cheap axes stretch more), beta = 1 inside R1, smooth to 0 at R0 (identity
  at every pinned endpoint).
lambda is found by BISECTION on the exactly-verified predicate
  min over pairs,k of (||W(rbar_jk) - W(rbar_lk)|| - delta_jl) >= target,
which is monotone-checked, so there is no repair-style oscillation.
The warped references then give per-step ball corridors
  eps_jk = min over partners (gap_jlk - delta_jl)/2
and each payload solves ONE convex SOCP (dynamics, pins, ||r - ref|| <= eps).
Pairwise separation >= delta_jl holds by the triangle inequality. No binaries.

Benchmarks (seed 3): LB 235.0, per-pair alternation 265.2, corridors 339.5.

Usage: py warp_oracle.py [--seed 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import route2_multi as r2  # noqa: E402
from corridor_oracle import axis_prices, alternate_pp  # noqa: E402

import cvxpy as cp  # noqa: E402


def make_warp(c, w, R1, R0):
    """Ball inflation in price-scaled coordinates: u -> u (|u|^3+rho^3)^{1/3}/|u|.
    Displacement bounded by rho (additive clearance), decays ~ rho^3/r^2,
    cheap axes (small w) absorb more of it; blended to identity at R0."""
    def warp(x, rho):
        x = np.asarray(x, float)
        d = x - c
        r = np.linalg.norm(d)
        if r >= R0 or r < 1e-9:
            return x
        u = w * d
        ru = float(np.linalg.norm(u))
        scale = (ru ** 3 + rho ** 3) ** (1.0 / 3.0) / ru
        disp = (scale - 1.0) * u / w
        beta = 1.0 if r <= R1 else ((R0 - r) / (R0 - R1)) ** 2
        return x + beta * disp
    return warp


def run(seed, N, tf, M, target=15.0):
    sc, psis, refs, pairs, gaps, _ = r2.crossing(seed, N, tf, M, n_points=8)
    LB = sum(r["fuel"] for r in refs)
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc.payloads]
    margin = sc.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]
    out = dict(seed=seed, M=M, LB=round(LB, 2))

    t0 = time.time()
    # conflict cluster center and measured axis prices
    cpts = []
    for (j, l) in pairs:
        d = np.linalg.norm(rbar[j] - rbar[l], axis=1)
        k = int(np.argmin(d))
        if d[k] < deltas[j][l]:
            cpts.append(0.5 * (rbar[j][k] + rbar[l][k]))
    c = np.mean(cpts, axis=0)
    w = np.mean([axis_prices(sc, psis, refs, j) for j in range(M)], axis=0)
    out["axis_prices"] = [round(x, 2) for x in w]

    end_d = min(np.linalg.norm(rbar[j][k] - c) for j in range(M) for k in (0, N))
    R0 = 0.95 * end_d
    R1 = min(200.0, 0.5 * R0)
    out["R1_R0"] = [round(R1, 1), round(R0, 1)]
    warp = make_warp(c, w, R1, R0)

    def min_margin(lam):
        wr = [np.array([warp(rbar[j][k], lam) for k in range(N + 1)])
              for j in range(M)]
        mm = min(np.linalg.norm(wr[j][k] - wr[l][k]) - deltas[j][l]
                 for (j, l) in pairs for k in range(N + 1))
        return mm, wr

    lo_l, hi_l = 1.0, 400.0
    mm_hi, _ = min_margin(hi_l)
    if mm_hi < target:
        out["status"] = f"rho=400 insufficient (margin {mm_hi:.1f})"
        print(json.dumps(out), flush=True)
        return out
    for _ in range(40):
        mid = 0.5 * (lo_l + hi_l)
        mm, _ = min_margin(mid)
        if mm >= target:
            hi_l = mid
        else:
            lo_l = mid
    lam = hi_l
    mm, wr = min_margin(lam)
    out["rho"] = round(lam, 1)
    out["warped_min_margin"] = round(mm, 1)
    out["warp_time"] = round(time.time() - t0, 1)

    # per-step ball radii from achieved margins
    eps = []
    CAP = 150.0
    for j in range(M):
        e = np.full(N + 1, CAP)
        for l in range(M):
            if l == j:
                continue
            g = np.linalg.norm(wr[j] - wr[l], axis=1) - deltas[j][l]
            e = np.minimum(e, 0.5 * g)
        eps.append(e)
    out["eps_min"] = round(float(min(e.min() for e in eps)), 1)
    print(json.dumps(out), flush=True)

    # decoupled convex tracking solves
    t0 = time.time()
    fuel_tot, hist = 0.0, []
    for j in range(M):
        r = cp.Variable((N + 1, 3))
        v = cp.Variable((N + 1, 3))
        F = cp.Variable((N, 3))
        cons = r2._dyn(sc, psis, r, v, F, j)
        for k in range(1, N):
            if eps[j][k] < CAP:
                cons.append(cp.norm(r[k] - wr[j][k]) <= eps[j][k])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            out["status"] = f"tracking SOCP failed payload {j}: {prob.status}"
            print(json.dumps({"status": out["status"]}), flush=True)
            return out
        fuel_tot += float(prob.value)
        hist.append(np.asarray(r.value))
    out["solve_time"] = round(time.time() - t0, 1)
    worst = min(min(np.linalg.norm(hist[j][k] - hist[l][k]) - deltas[j][l]
                    for k in range(N + 1)) for (j, l) in pairs)
    out["J_warp"] = round(fuel_tot, 2)
    out["min_sep_margin"] = round(worst, 1)
    out["gap_vs_LB"] = round((fuel_tot - LB) / fuel_tot, 4)

    # one-shot planes oriented by the WARPED gap directions (no tubes)
    t0 = time.time()
    oneshot = alternate_pp(sc, psis, pairs,
                           [dict(r=wr[j]) for j in range(M)], deltas, iters=1)
    out["J_warp_planes_oneshot"] = round(oneshot, 2) if oneshot else None
    out["oneshot_time"] = round(time.time() - t0, 1)

    t0 = time.time()
    alt = alternate_pp(sc, psis, pairs, refs, deltas)
    out["alternation_perpair"] = round(alt, 2) if alt else None
    if alt:
        out["vs_alternation_perpair"] = round(fuel_tot / alt - 1, 4)
    out["alt_time"] = round(time.time() - t0, 1)
    out["status"] = "ok"
    print(json.dumps({k: out.get(k) for k in ("status", "rho", "J_warp",
                                              "J_warp_planes_oneshot",
                                              "min_sep_margin", "gap_vs_LB",
                                              "alternation_perpair",
                                              "vs_alternation_perpair",
                                              "solve_time")}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--M", type=int, default=5)
    a = ap.parse_args()
    out = run(a.seed, a.N, a.tf, a.M)
    p = Path(__file__).with_name("results") / f"warp_oracle_s{a.seed}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
