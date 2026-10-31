"""Quick peek: per-step balls (half-split) vs water-filled balls vs
water-filled capsule WINDOWS (hull of a window of consecutive balls).

Window safety (rigorous, conservative): for each pair (j,l) at step k, the
two window-hulls are certified >= delta_jl apart by a slab test along the
pair's same-step direction eta = unit(x_j[k]-x_l[k]):
    min_a [eta.x_j[a] - eps_j[a]] - max_b [eta.x_l[b] + eps_l[b]] >= delta_jl
over the windows a,b. Windows grow uniformly and freeze when a check fails.
The window hull contains the step ball, so fuel can only improve.

Usage: py capsule_peek.py [--Ms 20 30] [--seeds 0 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_methods import relax_iso  # noqa: E402
from benchmark_tubes import tubes_detail, waterfill_radii  # noqa: E402
from multi_payload import scenario, translation  # noqa: E402
import route2_multi as r2  # noqa: E402

import cvxpy as cp  # noqa: E402


def grow_windows(x, eps, pairs, deltas, N, M, wmax=6):
    """Symmetric window growth with the slab certificate."""
    w = np.zeros((M, N + 1), int)

    def ok(j, l, k, wj, wl):
        d = x[j][k] - x[l][k]
        n = np.linalg.norm(d)
        if n < 1e-9:
            return False
        eta = d / n
        aj = slice(max(1, k - wj), min(N, k + wj) + 1)
        al = slice(max(1, k - wl), min(N, k + wl) + 1)
        lo = np.min(x[j][aj] @ eta - eps[j][aj])
        hi = np.max(x[l][al] @ eta + eps[l][al])
        return lo - hi >= deltas[j][l]

    nbrs = [[l for l in range(M) if any((min(a, b), max(a, b)) == (min(j, l), max(j, l)) for (a, b) in pairs)] for j in range(M)]
    for level in range(1, wmax + 1):
        for j in range(M):
            for k in range(1, N):
                if w[j][k] < level - 1:
                    continue
                cand = level
                good = all(ok(j, l, k, cand, w[l][k]) and ok(l, j, k, w[l][k], cand)
                           for l in nbrs[j])
                if good:
                    w[j][k] = cand
    return w


def tubes_windows(sc, psis, x, eps, w, N, M):
    fuel = 0.0
    for j in range(M):
        r = cp.Variable((N + 1, 3))
        v = cp.Variable((N + 1, 3))
        F = cp.Variable((N, 3))
        cons = r2._dyn(sc, psis, r, v, F, j)
        for k in range(1, N):
            a0, a1 = max(1, k - w[j][k]), min(N, k + w[j][k]) + 1
            cen = x[j][a0:a1]
            ee = eps[j][a0:a1]
            m = len(cen)
            if m == 1:
                cons.append(cp.norm(r[k] - cen[0]) <= ee[0])
            else:
                lam = cp.Variable(m, nonneg=True)
                u = cp.Variable(3)
                cons += [cp.sum(lam) == 1,
                         r[k] == cen.T @ lam + u,
                         cp.norm(u) <= ee @ lam]
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        fuel += float(prob.value)
    return fuel


def run(M, seed, N=30, tf=500.0, headroom=60.0):
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    sc = scenario.generate("random_seq", M=M, n_points=8, n_agents=3 * M,
                           seed=seed, nu_deg=60, N=N, tf=tf,
                           ring_radius=ring, max_tries=5000)
    psis = translation.psi_list(sc)
    refs = translation.solve_all(sc, psis=psis)
    LB = sum(r["fuel"] for r in refs)
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc.payloads]
    margin = sc.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    tgt = {(j, l): deltas[j][l] + headroom for (j, l) in pairs}
    x, _, _ = relax_iso(rbar, pairs, tgt, N, M)
    if x is None:
        return dict(M=M, seed=seed, status="relax stalled")
    row = dict(M=M, seed=seed, LB=round(LB, 2))
    J1, _, _ = tubes_detail(sc, psis, pairs, deltas, x, N, M, alloc="half")
    J2, _, _ = tubes_detail(sc, psis, pairs, deltas, x, N, M, alloc="maximal")
    eps = waterfill_radii(x, deltas, N, M)
    t0 = time.perf_counter()
    w = grow_windows(x, eps, pairs, deltas, N, M)
    row["t_windows"] = round(time.perf_counter() - t0, 2)
    row["w_mean"] = round(float(w[:, 1:N].mean()), 2)
    row["w_frac_open"] = round(float((w[:, 1:N] >= 3).mean()), 2)
    J3 = tubes_windows(sc, psis, x, eps, w, N, M)
    for name, J in (("half", J1), ("waterfill", J2), ("windows", J3)):
        if J:
            row[f"J_{name}"] = round(J, 2)
            row[f"gap_{name}_pct"] = round(100 * (J - LB) / LB, 2)
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+", default=[20, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    out = []
    for M in a.Ms:
        for s in a.seeds:
            try:
                r = run(M, s)
            except Exception as exc:
                r = dict(M=M, seed=s, status=f"error: {str(exc)[:90]}")
            out.append(r)
            print(json.dumps(r), flush=True)
    p = Path(__file__).with_name("results") / "capsule_peek.json"
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
