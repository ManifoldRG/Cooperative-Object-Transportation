"""Occupancy MISOCP oracle prototype (translational layer), center semantics.

Full-space tetrahedral partition seeded by the nominals: vertices = starts,
goals, pairwise conflict midpoints, samples along each nominal, Steiner
shells around the conflict cluster, bounding-box corners; scipy Delaunay
partitions the box into disjoint tets.

Encoding (center-in-cell):
  one-hot z[j,k,c] over candidate cells; z=1 => center r_jk inside cell c
  (big-M on the 4 face inequalities; the convex direction only, so no
  exclusion-certificate binaries are needed);
  collision: forbid simultaneous occupancy of cell PAIRS whose minimum
  distance is < delta (safe underestimate: centroid distance - radii),
  aggregated as clique rows  z_jkc + sum_{c' in conflict(c)} z_lkc' <= 1.
Soundness: centers in cells with pairwise clearance >= delta implies
||r_j - r_l|| >= delta. Positions confined to ||r_jk - rbar_jk|| <= R_CAND.

Benchmark (seed 3): LB 235.0, alternation (uncertified) 265.7.

Usage: py occupancy_misocp.py [--seed 3] [--sample-every 3] [--rcand 130]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).parent))
import route2_multi as r2  # noqa: E402

import cvxpy as cp  # noqa: E402


def icosahedron():
    p = (1 + 5 ** 0.5) / 2
    v = []
    for a in (-1, 1):
        for b in (-p, p):
            v += [np.array([0, a, b]), np.array([a, b, 0]), np.array([b, 0, a])]
    return np.array([x / np.linalg.norm(x) for x in v])


def build_partition(sc, refs, pairs, gaps, sample_every, box, dedupe=25.0):
    N = sc.N
    pts, conf = [], []
    for j in range(sc.M):
        rb = np.asarray(refs[j]["r"])
        pts.append(rb[0])
        pts.append(rb[N])
        for k in range(sample_every, N, sample_every):
            pts.append(rb[k])
    for (j, l) in pairs:
        if gaps[(j, l)] < sc.delta:
            d = np.linalg.norm(np.asarray(refs[j]["r"]) - np.asarray(refs[l]["r"]),
                               axis=1)
            k = int(np.argmin(d))
            m = 0.5 * (refs[j]["r"][k] + refs[l]["r"][k])
            pts.append(m)
            conf.append(m)
    # Steiner shells around the conflict cluster: delta-scale central cells
    if conf:
        c0 = np.mean(conf, axis=0)
        for rad in (0.7 * sc.delta, 1.4 * sc.delta):
            for d in icosahedron():
                pts.append(c0 + rad * d)
    V = []
    for p in pts:
        if all(np.linalg.norm(p - q) > dedupe for q in V):
            V.append(np.asarray(p, float))
    for sx in (-box, box):
        for sy in (-box, box):
            for sz in (-box, box):
                V.append(np.array([sx, sy, sz], float))
    V = np.array(V)
    tri = Delaunay(V, qhull_options="QJ")
    tets = []
    for simp in tri.simplices:
        vs = V[simp]
        cen = vs.mean(axis=0)
        A, b = [], []
        ok = True
        for f in range(4):
            idx = [i for i in range(4) if i != f]
            p0, p1, p2 = vs[idx]
            n = np.cross(p1 - p0, p2 - p0)
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                ok = False
                break
            n = n / nn
            bb = float(n @ p0)
            if n @ cen > bb:
                n, bb = -n, -bb
            A.append(n)
            b.append(bb)
        if ok:
            rad = float(np.max(np.linalg.norm(vs - cen, axis=1)))
            tets.append(dict(A=np.array(A), b=np.array(b), cen=cen, rad=rad,
                             vs=vs))
    return V, tets


def run(seed, N, tf, M, sample_every, R, mio_time):
    sc, psis, refs, pairs, gaps, n_conf = r2.crossing(seed, N, tf, M, n_points=8)
    LB = sum(r["fuel"] for r in refs)
    out = dict(seed=seed, M=M, delta=round(sc.delta, 1), R_cand=R,
               sample_every=sample_every, LB=round(LB, 2))

    t0 = time.time()
    V, tets = build_partition(sc, refs, pairs, gaps, sample_every, box=900.0)
    nT = len(tets)
    out["n_vertices"] = len(V)
    out["n_tets"] = nT

    # forbidden cell pairs: safe underestimate of inter-cell distance
    cen = np.array([t["cen"] for t in tets])
    rad = np.array([t["rad"] for t in tets])
    D = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2) \
        - rad[:, None] - rad[None, :]
    forbid = D < sc.delta           # includes the diagonal
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]

    # candidate cells per (j, k): center could lie in the cell within the
    # R-ball (max_f(a.rbar - b) <= R is a safe inclusion test)
    cand = {}
    for j in range(M):
        for k in range(N + 1):
            x = rbar[j][k]
            cs = [c for c, t in enumerate(tets)
                  if float(np.max(t["A"] @ x - t["b"])) <= R]
            cand[(j, k)] = np.array(cs, dtype=int)

    # active (j, k): some partner's candidate cell conflicts with ours
    active = set()
    for (j, l) in pairs:
        for k in range(N + 1):
            if forbid[np.ix_(cand[(j, k)], cand[(l, k)])].any():
                active.add((j, k))
                active.add((l, k))
    out["n_active"] = len(active)

    cons = []
    rv = [(cp.Variable((N + 1, 3)), cp.Variable((N + 1, 3)), cp.Variable((N, 3)))
          for _ in range(M)]
    for j, (r, v, F) in enumerate(rv):
        cons += r2._dyn(sc, psis, r, v, F, j)
        for k in range(1, N):
            cons.append(cp.norm(r[k] - rbar[j][k]) <= R)

    zvar, n_z = {}, 0
    for (j, k) in sorted(active):
        cs = cand[(j, k)]
        z = cp.Variable(len(cs), boolean=True)
        zvar[(j, k)] = (cs, z)
        n_z += len(cs)
        cons.append(cp.sum(z) == 1)
        # membership big-M, vectorized over the 4*len(cs) face rows
        A = np.vstack([tets[c]["A"] for c in cs])
        b = np.concatenate([tets[c]["b"] for c in cs])
        Mf = A @ rbar[j][k] + R - b + 1.0
        idx = np.repeat(np.arange(len(cs)), 4)
        cons.append(A @ rv[j][0][k] <= b + cp.multiply(Mf, 1 - z[idx]))
    out["n_z"] = n_z

    n_rows = 0
    for (j, l) in pairs:
        for k in range(N + 1):
            if (j, k) not in zvar or (l, k) not in zvar:
                continue
            cj, zj = zvar[(j, k)]
            cl, zl = zvar[(l, k)]
            B = forbid[np.ix_(cj, cl)].astype(float)
            if not B.any():
                continue
            cons.append(zj + B @ zl <= 1)
            n_rows += len(cj)
    out["n_clique_rows"] = n_rows
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
    if prob.status in ("optimal", "optimal_inaccurate"):
        fuel, hist, okbc = 0.0, [], True
        for j in range(M):
            P0 = sc.payloads[j]
            Fv = np.asarray(rv[j][2].value)
            rj = np.asarray(P0.x0.r, float).copy()
            vj = np.asarray(P0.x0.v, float).copy()
            h = [rj.copy()]
            for k in range(N):
                P = np.asarray(psis[k])
                rj, vj = rj + sc.dt * vj, vj + sc.dt * (
                    P[:, :3] @ rj + P[:, 3:] @ vj + Fv[k] / P0.m)
                h.append(rj.copy())
            if (np.linalg.norm(rj - P0.xf.r) > 1e-3
                    or np.linalg.norm(vj - P0.xf.v) > 1e-4):
                okbc = False
            fuel += float(np.linalg.norm(Fv, axis=1).sum())
            hist.append(np.asarray(h))
        min_sep = min(np.linalg.norm(hist[j][k] - hist[l][k])
                      for (j, l) in pairs for k in range(N + 1))
        out["J_occ"] = round(fuel, 2)
        out["min_sep"] = round(min_sep, 1)
        out["bc_ok"] = okbc
        out["gap_vs_LB"] = round((fuel - LB) / fuel, 4)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("status", "J_occ", "min_sep", "bc_ok",
                               "gap_vs_LB", "solve_time")}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--sample-every", type=int, default=3)
    ap.add_argument("--rcand", type=float, default=130.0)
    ap.add_argument("--mio-time", type=float, default=900.0)
    a = ap.parse_args()
    out = run(a.seed, a.N, a.tf, a.M, a.sample_every, a.rcand, a.mio_time)
    p = Path(__file__).with_name("results") / f"occupancy_misocp_s{a.seed}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
