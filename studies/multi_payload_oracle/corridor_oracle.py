"""Spatially-disjoint-corridor oracle prototype, payload-size-aware version.

Per-pair separation delta_jl = extent_j + extent_l + margin (instead of the
global delta = 2 max extent + margin). Pipeline:
  1. cube grid, side s = delta_global/2;
  2. prioritized A* (26-conn, Euclidean costs, 5 orders); payload j avoids
     committed corridor l dilated by ceil(delta_jl/s) cells;
  3. greedy prism merge + WIDENING pass: each prism face is inflated in s/2
     steps while it keeps box-distance >= delta_jl to every other payload's
     prisms (payload-size-aware fattening -> room to cut corners);
  4. per-payload keep-in MILP (monotone stage binaries, switching times free).
Benchmark lines: LB (collision-free translational), alternation under the
SAME per-pair deltas (fair), alternation under global delta (old 265.7).

Usage: py corridor_oracle.py [--seed 3]
"""
from __future__ import annotations

import argparse
import heapq
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

from multi_payload import translation  # noqa: E402


def axis_prices(sc, psis, refs, j, probe=50.0):
    """Marginal fuel per meter of mid-course detour, per Hill axis, from
    +-waypoint probes on the single-payload translational SOCP."""
    mid = sc.N // 2
    base = refs[j]["fuel"]
    r0 = np.asarray(refs[j]["r"][mid])
    w = np.zeros(3)
    for ax in range(3):
        worst = 0.0
        for sgn in (1.0, -1.0):
            wp = {mid: r0 + sgn * probe * np.eye(3)[ax]}
            sol = translation.solve_reference(sc, j, wp, psis)
            if sol["ok"]:
                worst = max(worst, (sol["fuel"] - base) / probe)
        w[ax] = max(worst, 1e-5)
    return w / w.min()

NBR = [np.array(d) for d in itertools.product((-1, 0, 1), repeat=3) if any(d)]


def astar(start, goal, blocked, hi_idx, s, W):
    """Weighted A*: edge cost s*||W o d||_2 (W = per-axis fuel prices)."""
    start, goal = tuple(start), tuple(goal)
    w0, w1, w2 = float(W[0]), float(W[1]), float(W[2])
    nbrs = [((d[0], d[1], d[2]),
             s * ((w0 * d[0]) ** 2 + (w1 * d[1]) ** 2
                  + (w2 * d[2]) ** 2) ** 0.5) for d in NBR]

    def h(c):
        return s * ((w0 * (c[0] - goal[0])) ** 2
                    + (w1 * (c[1] - goal[1])) ** 2
                    + (w2 * (c[2] - goal[2])) ** 2) ** 0.5

    openq = [(h(start), 0.0, start, None)]
    came, gbest = {}, {start: 0.0}
    nx, ny, nz = int(hi_idx[0]), int(hi_idx[1]), int(hi_idx[2])
    while openq:
        f, g, c, par = heapq.heappop(openq)
        if c in came:
            continue
        came[c] = par
        if c == goal:
            path = [c]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        for d, dc in nbrs:
            n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if n in came or n in blocked:
                continue
            if (n[0] < 0 or n[0] > nx or n[1] < 0 or n[1] > ny
                    or n[2] < 0 or n[2] > nz):
                continue
            ng = g + dc
            if ng < gbest.get(n, np.inf):
                gbest[n] = ng
                heapq.heappush(openq, (ng + h(n), ng, n, c))
    return None


def dilate(cells, g):
    out = set()
    offs = list(itertools.product(range(-g, g + 1), repeat=3))
    for c in cells:
        for d in offs:
            out.add((c[0] + d[0], c[1] + d[1], c[2] + d[2]))
    return out


def box_dist(b1, b2):
    gap = np.maximum(0.0, np.maximum(b2[0] - b1[1], b1[0] - b2[1]))
    return float(np.linalg.norm(gap))


def grow_prisms(path, lo, s):
    """Greedy merge of consecutive path cells into boxes [lo3, hi3] (meters),
    merging only while the bounding box adds no off-path cells."""
    cells = set(path)
    prisms, i = [], 0
    while i < len(path):
        jmax = i
        for j in range(i + 1, len(path)):
            seg = np.array(path[i:j + 1])
            lo_i, hi_i = seg.min(axis=0), seg.max(axis=0)
            boxcells = set(itertools.product(
                *[range(lo_i[a], hi_i[a] + 1) for a in range(3)]))
            if not boxcells <= cells:
                break
            jmax = j
        seg = np.array(path[i:jmax + 1])
        lo_i, hi_i = seg.min(axis=0), seg.max(axis=0)
        prisms.append([lo + s * lo_i, lo + s * (hi_i + 1.0)])
        i = jmax + 1
    return prisms


def widen(prisms, deltas, box_lo, box_hi, step, rounds=60):
    """Inflate every prism face while per-pair separation holds."""
    M = len(prisms)
    for _ in range(rounds):
        changed = False
        for j in range(M):
            for p in prisms[j]:
                for side in (0, 1):
                    for ax in range(3):
                        trial = [p[0].copy(), p[1].copy()]
                        if side == 0:
                            trial[0][ax] -= step
                            if trial[0][ax] < box_lo[ax]:
                                continue
                        else:
                            trial[1][ax] += step
                            if trial[1][ax] > box_hi[ax]:
                                continue
                        ok = all(box_dist(trial, q) >= deltas[j][l]
                                 for l in range(M) if l != j
                                 for q in prisms[l])
                        if ok:
                            p[0][:], p[1][:] = trial[0], trial[1]
                            changed = True
        if not changed:
            break
    return prisms


def keepin_solve(sc, psis, j, prisms):
    N, Q = sc.N, len(prisms)
    r = cp.Variable((N + 1, 3))
    v = cp.Variable((N + 1, 3))
    F = cp.Variable((N, 3))
    th = cp.Variable((N + 1, Q), boolean=True)
    cons = r2._dyn(sc, psis, r, v, F, j)
    Mbig = 4000.0
    cons += [th[:, 0] == 1, th[0, 1:] == 0, th[N, Q - 1] == 1]
    for q in range(1, Q):
        cons.append(th[:, q] <= th[:, q - 1])
        cons.append(th[1:, q] >= th[:-1, q])
    for k in range(N + 1):
        for q in range(Q):
            b = th[k, q] - (th[k, q + 1] if q + 1 < Q else 0)
            cons.append(r[k] >= prisms[q][0] - Mbig * (1 - b))
            cons.append(r[k] <= prisms[q][1] + Mbig * (1 - b))
    prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
    try:
        prob.solve(solver="MOSEK",
                   mosek_params={"MSK_DPAR_MIO_MAX_TIME": 120.0})
    except cp.error.SolverError:
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return float(prob.value), np.asarray(r.value)


def alternate_pp(sc, psis, pairs, refs, deltas, iters=20):
    """Alternation benchmark under per-pair deltas."""
    etas = r2.nominal_normals(refs, pairs, sc.N)
    best = None
    for _ in range(iters):
        rv = [(cp.Variable((sc.N + 1, 3)), cp.Variable((sc.N + 1, 3)),
               cp.Variable((sc.N, 3))) for _ in range(sc.M)]
        cons = []
        for j, (r, v, F) in enumerate(rv):
            cons += r2._dyn(sc, psis, r, v, F, j)
        for (j, l) in pairs:
            for k in range(sc.N + 1):
                cons.append(etas[(j, l)][k] @ (rv[j][0][k] - rv[l][0][k])
                            >= deltas[j][l])
        prob = cp.Problem(cp.Minimize(sum(cp.sum(cp.norm(F, 2, axis=1))
                                          for _, _, F in rv)), cons)
        try:
            prob.solve(solver="CLARABEL")
        except cp.error.SolverError:
            break
        if prob.status not in ("optimal", "optimal_inaccurate"):
            break
        if best is not None and prob.value > best - 1e-9:
            break
        best = float(prob.value)
        for (j, l) in pairs:
            for k in range(sc.N + 1):
                d = np.asarray(rv[j][0].value[k]) - np.asarray(rv[l][0].value[k])
                n = np.linalg.norm(d)
                if n > 1e-9:
                    etas[(j, l)][k] = d / n
    return best


def run(seed, N, tf, M):
    sc, psis, refs, pairs, gaps, _ = r2.crossing(seed, N, tf, M, n_points=8)
    LB = sum(r["fuel"] for r in refs)
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc.payloads]
    margin = sc.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    s = sc.delta / 4.0
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]
    Ws = [axis_prices(sc, psis, refs, j) for j in range(M)]
    out = dict(seed=seed, M=M, delta_global=round(sc.delta, 1),
               extents=[round(e, 1) for e in ext], margin=round(margin, 1),
               delta_pairs_minmax=[round(min(deltas[j][l] for (j, l) in pairs), 1),
                                   round(max(deltas[j][l] for (j, l) in pairs), 1)],
               cube=round(s, 1), LB=round(LB, 2),
               axis_prices=[[round(x, 2) for x in w] for w in Ws])

    allpts = np.vstack(rbar)
    lo = allpts.min(axis=0) - 12 * s
    hi = allpts.max(axis=0) + 12 * s
    hi_idx = np.floor((hi - lo) / s).astype(int)

    def cell(x):
        return tuple(np.clip(np.floor((x - lo) / s).astype(int), 0, hi_idx))

    t0 = time.time()
    orders = [list(range(M)), list(range(M))[::-1],
              sorted(range(M), key=lambda j: -np.linalg.norm(rbar[j][0] - rbar[j][N])),
              sorted(range(M), key=lambda j: np.linalg.norm(rbar[j][0] - rbar[j][N])),
              [(j + 2) % M for j in range(M)]]
    best = None
    for order in orders:
        paths, committed, ok = {}, [], True
        for j in order:
            blocked = set()
            for l in committed:
                blocked |= dilate(set(paths[l]), int(np.ceil(deltas[j][l] / s)))
            p = astar(cell(rbar[j][0]), cell(rbar[j][N]), blocked, hi_idx, s,
                      Ws[j])
            if p is None:
                ok = False
                break
            paths[j] = p
            committed.append(j)
        if ok:
            tot = sum(s * sum(np.linalg.norm(Ws[j] * (np.array(paths[j][i + 1])
                                                      - np.array(paths[j][i])))
                              for i in range(len(paths[j]) - 1))
                      for j in paths)
            if best is None or tot < best[0]:
                best = (tot, paths)
    if best is None:
        out["status"] = "no disjoint corridor set found"
        print(json.dumps(out), flush=True)
        return out
    _, paths = best
    out["search_time"] = round(time.time() - t0, 1)

    t0 = time.time()
    prisms = {j: grow_prisms(paths[j], lo, s) for j in range(M)}
    # shrink-check then widen with per-pair deltas
    sep0 = min(box_dist(p1, p2) - deltas[j][l]
               for (j, l) in pairs for p1 in prisms[j] for p2 in prisms[l])
    if sep0 < -1e-6:
        out["status"] = f"initial prism separation violated by {-sep0:.1f} m"
        print(json.dumps(out), flush=True)
        return out
    prisms = widen([prisms[j] for j in range(M)], deltas, lo, hi, s / 2.0)
    sepmargin = min(box_dist(p1, p2) - deltas[j][l]
                    for (j, l) in pairs for p1 in prisms[j] for p2 in prisms[l])
    out["prisms_per_payload"] = [len(prisms[j]) for j in range(M)]
    out["min_pair_sep_margin"] = round(sepmargin, 1)
    out["widen_time"] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)

    t0 = time.time()
    fuel_tot, hist = 0.0, []
    for j in range(M):
        res = keepin_solve(sc, psis, j, prisms[j])
        if res is None:
            out["status"] = f"keep-in MILP failed for payload {j}"
            print(json.dumps({"status": out["status"]}), flush=True)
            return out
        fj, rj = res
        fuel_tot += fj
        hist.append(rj)
    out["solve_time"] = round(time.time() - t0, 1)
    worst = min(min(np.linalg.norm(hist[j][k] - hist[l][k]) - deltas[j][l]
                    for k in range(N + 1)) for (j, l) in pairs)
    out["J_corridor"] = round(fuel_tot, 2)
    out["min_sep_margin"] = round(worst, 1)
    out["gap_vs_LB"] = round((fuel_tot - LB) / fuel_tot, 4)

    t0 = time.time()
    alt_pp = alternate_pp(sc, psis, pairs, refs, deltas)
    out["alternation_perpair"] = round(alt_pp, 2) if alt_pp else None
    out["alt_time"] = round(time.time() - t0, 1)
    if alt_pp:
        out["vs_alternation_perpair"] = round(fuel_tot / alt_pp - 1, 4)
    out["status"] = "ok"
    print(json.dumps({k: out[k] for k in ("status", "J_corridor",
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
    p = Path(__file__).with_name("results") / f"corridor_oracle_s{a.seed}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
