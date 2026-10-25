"""Occupancy MISOCP, cube-grid version (translational layer).

Uniform cube grid (side s = delta) over the contested region only; ball-touch
occupancy with rho = delta/2 (a cube just encompasses the ball):
  z[j,k,c] = 1 whenever ball(r_jk, rho) touches cube c; z = 0 is allowed only
  with an axis certificate |r_i - c_i| >= h + rho (6 one-sided options, big-M);
  z_j + z_l <= 1 per contested pair and shared cube.
Disjoint touched sets => balls disjoint => ||r_j - r_l|| >= delta.

Everything outside the contested region is covered by a theorem, not cells:
with ||r_jk - rbar_jk|| <= R enforced for all j,k, any pair whose nominals are
>= delta + 2R apart at step k is separated automatically. Cells therefore
exist only for (pair, k) with nominal distance < delta + 2R, and only on cubes
reachable by both members.

Benchmark (seed 3): LB 235.0, alternation (uncertified) 265.7.

Usage: py occupancy_cubes.py [--seed 3] [--rcand 120] [--mio-time 900]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import route2_multi as r2  # noqa: E402

import cvxpy as cp  # noqa: E402


def run(seed, N, tf, M, R, mio_time):
    sc, psis, refs, pairs, gaps, _ = r2.crossing(seed, N, tf, M, n_points=8)
    LB = sum(r["fuel"] for r in refs)
    delta = sc.delta
    rho = delta / 2.0
    s = delta            # cube side: just encompasses the rho-ball
    h = s / 2.0
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]
    out = dict(seed=seed, M=M, delta=round(delta, 1), cube=round(s, 1),
               R_cand=R, LB=round(LB, 2))

    t0 = time.time()
    # contested (pair, k): the R-ball theorem does not separate them
    contested = [(j, l, k) for (j, l) in pairs for k in range(N + 1)
                 if np.linalg.norm(rbar[j][k] - rbar[l][k]) < delta + 2 * R]
    out["contested_pair_steps"] = len(contested)
    if not contested:
        out["status"] = "nothing contested"
        print(json.dumps(out), flush=True)
        return out

    # cube lattice covering the contested reach
    reach = R + rho + h
    pts = np.array([rbar[j][k] for (j, l, k) in contested]
                   + [rbar[l][k] for (j, l, k) in contested])
    lo = pts.min(axis=0) - reach
    hi = pts.max(axis=0) + reach
    n_ax = np.maximum(1, np.ceil((hi - lo) / s).astype(int))
    centers = np.array([lo + s * (np.array(ix) + 0.5) for ix in
                        itertools.product(*[range(n) for n in n_ax])])
    out["grid_dims"] = n_ax.tolist()
    out["n_cubes"] = len(centers)

    def cand(j, k):
        d = np.abs(centers - rbar[j][k]).max(axis=1)
        return set(np.nonzero(d <= reach)[0].tolist())

    cand_cache = {}
    shared = {}            # (j,k) -> set of cubes needing z
    rows = []              # (j, l, k, cube)
    for (j, l, k) in contested:
        for key in ((j, k), (l, k)):
            if key not in cand_cache:
                cand_cache[key] = cand(*key)
        both = cand_cache[(j, k)] & cand_cache[(l, k)]
        for c in both:
            rows.append((j, l, k, c))
        shared.setdefault((j, k), set()).update(both)
        shared.setdefault((l, k), set()).update(both)

    cons = []
    rv = [(cp.Variable((N + 1, 3)), cp.Variable((N + 1, 3)), cp.Variable((N, 3)))
          for _ in range(M)]
    for j, (r, v, F) in enumerate(rv):
        cons += r2._dyn(sc, psis, r, v, F, j)
        for k in range(1, N):
            cons.append(cp.norm(r[k] - rbar[j][k]) <= R)

    zvar, n_z, n_y = {}, 0, 0
    for (j, k), cubes in sorted(shared.items()):
        cubes = sorted(cubes)
        z = cp.Variable(len(cubes), boolean=True)
        n_z += len(cubes)
        for ci, c in enumerate(cubes):
            zvar[(j, k, c)] = z[ci]
            certs = []
            for ax in range(3):
                for sign in (1.0, -1.0):
                    # certificate: sign*(r_ax - c_ax) >= h + rho
                    if sign * (rbar[j][k][ax] - centers[c][ax]) + R >= h + rho:
                        certs.append((ax, sign))
            y = cp.Variable(len(certs), boolean=True)
            n_y += len(certs)
            cons.append(cp.sum(y) >= 1 - z[ci])
            for yi, (ax, sign) in enumerate(certs):
                Mf = (h + rho) - sign * (rbar[j][k][ax] - centers[c][ax]) \
                    + R + 1.0
                cons.append(sign * (rv[j][0][k][ax] - centers[c][ax])
                            >= h + rho - Mf * (1 - y[yi]))
    for (j, l, k, c) in rows:
        cons.append(zvar[(j, k, c)] + zvar[(l, k, c)] <= 1)
    out["n_z"] = n_z
    out["n_y"] = n_y
    out["n_rows"] = len(rows)
    out["build_time"] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)

    t0 = time.time()
    prob = cp.Problem(cp.Minimize(sum(cp.sum(cp.norm(F, 2, axis=1))
                                      for _, _, F in rv)), cons)
    try:
        prob.solve(solver="MOSEK",
                   mosek_params={"MSK_DPAR_MIO_MAX_TIME": float(mio_time),
                                 "MSK_DPAR_MIO_TOL_REL_GAP": 0.005})
    except cp.error.SolverError as exc:
        out["status"] = f"solver error: {str(exc)[:100]}"
        print(json.dumps({"status": out["status"]}), flush=True)
        return out
    out["solve_time"] = round(time.time() - t0, 1)
    out["status"] = prob.status
    try:
        out["mio_gap"] = round(float(
            prob.solver_stats.extra_stats.get("mio_obj_rel_gap", float("nan"))), 4)
    except Exception:
        pass
    if prob.status in ("optimal", "optimal_inaccurate") and rv[0][2].value is not None:
        fuel, hist, okbc = 0.0, [], True
        for j in range(M):
            P0 = sc.payloads[j]
            Fv = np.asarray(rv[j][2].value)
            rj = np.asarray(P0.x0.r, float).copy()
            vj = np.asarray(P0.x0.v, float).copy()
            hh = [rj.copy()]
            for k in range(N):
                P = np.asarray(psis[k])
                rj, vj = rj + sc.dt * vj, vj + sc.dt * (
                    P[:, :3] @ rj + P[:, 3:] @ vj + Fv[k] / P0.m)
                hh.append(rj.copy())
            if (np.linalg.norm(rj - P0.xf.r) > 1e-3
                    or np.linalg.norm(vj - P0.xf.v) > 1e-4):
                okbc = False
            fuel += float(np.linalg.norm(Fv, axis=1).sum())
            hist.append(np.asarray(hh))
        min_sep = min(np.linalg.norm(hist[j][k] - hist[l][k])
                      for (j, l) in pairs for k in range(N + 1))
        out["J_occ"] = round(fuel, 2)
        out["min_sep"] = round(min_sep, 1)
        out["bc_ok"] = okbc
        out["gap_vs_LB"] = round((fuel - LB) / fuel, 4)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("status", "J_occ", "min_sep", "bc_ok",
                               "gap_vs_LB", "solve_time", "mio_gap")}),
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--rcand", type=float, default=120.0)
    ap.add_argument("--mio-time", type=float, default=900.0)
    a = ap.parse_args()
    out = run(a.seed, a.N, a.tf, a.M, a.rcand, a.mio_time)
    p = Path(__file__).with_name("results") / f"occupancy_cubes_s{a.seed}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
