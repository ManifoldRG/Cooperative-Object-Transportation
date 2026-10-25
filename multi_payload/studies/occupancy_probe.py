"""Feasibility probe for the occupancy encoding: does the alternation
solution (collision-free, min_sep ~ delta) fit the partition's forbidden-pair
rules? Reports violations under (a) the radii-underestimate distance and
(b) the exact inter-cell distance (small QP per flagged pair).

Usage: py occupancy_probe.py [--seed 3] [--sample-every 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import route2_multi as r2  # noqa: E402
import sandwich_check as sw  # noqa: E402
import occupancy_misocp as om  # noqa: E402

import cvxpy as cp  # noqa: E402


def exact_dist(t1, t2):
    w1 = cp.Variable(4, nonneg=True)
    w2 = cp.Variable(4, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.norm(t1["vs"].T @ w1 - t2["vs"].T @ w2)),
                      [cp.sum(w1) == 1, cp.sum(w2) == 1])
    prob.solve(solver="CLARABEL")
    return float(prob.value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--M", type=int, default=5)
    ap.add_argument("--sample-every", type=int, default=3)
    a = ap.parse_args()

    sc, psis, refs, pairs, gaps, _ = r2.crossing(a.seed, a.N, a.tf, a.M,
                                                 n_points=8)
    best = sw.alternate_full(sc, psis, pairs, refs)
    fuel, rs, _ = best
    print(json.dumps(dict(alternation_fuel=round(fuel, 2))), flush=True)

    V, tets = om.build_partition(sc, refs, pairs, gaps, a.sample_every, 900.0)
    cen = np.array([t["cen"] for t in tets])
    rad = np.array([t["rad"] for t in tets])

    def cell_of(x):
        # the tet containing x (QJ joggle: take min max-violation)
        viol = [float(np.max(t["A"] @ x - t["b"])) for t in tets]
        return int(np.argmin(viol)), float(np.min(viol))

    assign = {}
    max_viol = 0.0
    for j in range(a.M):
        for k in range(a.N + 1):
            c, v = cell_of(np.asarray(rs[j][k]))
            assign[(j, k)] = c
            max_viol = max(max_viol, v)

    under_viol, exact_viol, checked = 0, 0, {}
    worst = []
    for (j, l) in pairs:
        for k in range(a.N + 1):
            c1, c2 = assign[(j, k)], assign[(l, k)]
            dU = (np.linalg.norm(cen[c1] - cen[c2]) - rad[c1] - rad[c2]
                  if c1 != c2 else -1.0)
            if dU < sc.delta:
                under_viol += 1
                key = (min(c1, c2), max(c1, c2))
                if key not in checked:
                    checked[key] = 0.0 if c1 == c2 else exact_dist(tets[c1],
                                                                  tets[c2])
                if checked[key] < sc.delta:
                    exact_viol += 1
                    worst.append(dict(pair=[j, l], k=k,
                                      sep=round(float(np.linalg.norm(
                                          rs[j][k] - rs[l][k])), 1),
                                      cell_dist=round(checked[key], 1)))
    print(json.dumps(dict(
        n_tets=len(tets), containment_max_viol=round(max_viol, 3),
        underestimate_violations=under_viol,
        exact_violations=exact_viol,
        distinct_cell_pairs_checked=len(checked),
        worst_examples=worst[:8])), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
