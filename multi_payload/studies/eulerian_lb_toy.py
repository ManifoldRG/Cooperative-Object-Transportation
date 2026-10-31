"""Congestion-priced Eulerian (fluid) lower bound vs the nominal lower bound
and the inflation+tubes pipeline, on a tractable toy class.

Class: M agents in 2D, single-integrator dynamics (control = velocity),
energy cost sum_j int ||rdot_j||^2 dt, endpoints fixed, pairwise separation
||r_j - r_l|| >= delta at every time. Crossing geometry, the congested regime.

Validity of the fluid LB (every slack points the valid-LB way):
- A true trajectory maps to bilinear (4-cell) mass weights per step; the
  product coupling is a feasible flux whose quadratic cost exceeds the true
  step cost by at most the two snap variances <= h^2. Arc cost is therefore
  max(0, ||dy||^2 - h^2)/dt, an UNDER-estimate of the mapped true cost.
- Exclusion: mass of agent j lies within h/sqrt(2) of its true center, so
  with probe radius Rex = delta/2 - h/sqrt(2), any two agents >= delta apart
  can have mass inside a probe disk from AT MOST ONE of them; hence
  sum_j mass_j(disk) <= 1 holds for every rigid-feasible solution.
- Hop set is over-permissive (speed cap = vmax_mult x nominal + snap).

Numbers per instance: LB_nom (straight lines), LB_fluid (grid LP),
J_tubes (gauss inflation + decoupled tube QPs), J_alt (iterated fixed-normal
joint QP). Congestion recovery = (LB_fluid - LB_nom)/(J_alt - LB_nom).

Usage: py eulerian_lb_toy.py [--Ms 3 4] [--seeds 0 1 2] [--h 0.03] [--N 10]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_methods import relax_iso  # noqa: E402

import cvxpy as cp  # noqa: E402

DELTA = 0.22
T = 1.0


def instance(M, seed):
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi)
    starts, goals = [], []
    for j in range(M):
        th = phase + 2 * np.pi * j / M
        jit = rng.uniform(-0.2, 0.2) * np.pi / M
        starts.append(0.5 + 0.38 * np.array([np.cos(th), np.sin(th)]))
        goals.append(0.5 - 0.38 * np.array([np.cos(th + jit), np.sin(th + jit)]))
    return np.array(starts), np.array(goals)


def lb_nominal(starts, goals):
    return float(sum(np.sum((g - s) ** 2) / T for s, g in zip(starts, goals)))


def rigid_pipeline(starts, goals, N):
    M = len(starts)
    dt = T / N
    rbar = [np.linspace(starts[j], goals[j], N + 1) for j in range(M)]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    tgt = {p: DELTA + 0.05 for p in pairs}
    x, sweeps, resid = relax_iso(rbar, pairs, tgt, N, M)
    if x is None:
        return None, None
    J_tubes = 0.0
    for j in range(M):
        e = np.full(N + 1, 1e9)
        for l in range(M):
            if l != j:
                e = np.minimum(e, 0.5 * (np.linalg.norm(x[j] - x[l], axis=1) - DELTA))
        r = cp.Variable((N + 1, 2))
        cons = [r[0] == starts[j], r[N] == goals[j]]
        for k in range(1, N):
            cons.append(cp.norm(r[k] - x[j][k]) <= e[k])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.sum_squares(r[1:] - r[:-1])) / dt), cons)
        prob.solve(solver="CLARABEL")
        J_tubes += float(prob.value)
    etas = {}
    for (j, l) in pairs:
        etas[(j, l)] = [None] * (N + 1)
        for k in range(N + 1):
            d = x[j][k] - x[l][k]
            n = np.linalg.norm(d)
            etas[(j, l)][k] = d / n if n > 1e-9 else np.array([1.0, 0])
    best = None
    for _ in range(12):
        rv = [cp.Variable((N + 1, 2)) for _ in range(M)]
        cons = []
        for j in range(M):
            cons += [rv[j][0] == starts[j], rv[j][N] == goals[j]]
        for (j, l) in pairs:
            for k in range(N + 1):
                cons.append(etas[(j, l)][k] @ (rv[j][k] - rv[l][k]) >= DELTA)
        prob = cp.Problem(cp.Minimize(
            sum(cp.sum(cp.sum_squares(r[1:] - r[:-1])) for r in rv) / dt), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            break
        if best is not None and prob.value > best - 1e-9:
            break
        best = float(prob.value)
        for (j, l) in pairs:
            for k in range(N + 1):
                d = np.asarray(rv[j].value[k]) - np.asarray(rv[l].value[k])
                n = np.linalg.norm(d)
                if n > 1e-9:
                    etas[(j, l)][k] = d / n
    return J_tubes, best


def bilinear(p, n1, h):
    """4-cell bilinear weights of point p on the cell-center grid."""
    gx = np.clip(p[0] / h - 0.5, 0, n1 - 1 - 1e-9)
    gy = np.clip(p[1] / h - 0.5, 0, n1 - 1 - 1e-9)
    i0, j0 = int(gy), int(gx)
    fy, fx = gy - i0, gx - j0
    w = np.zeros(n1 * n1)
    for (di, dj, ww) in ((0, 0, (1 - fy) * (1 - fx)), (0, 1, (1 - fy) * fx),
                         (1, 0, fy * (1 - fx)), (1, 1, fy * fx)):
        i, j_ = min(i0 + di, n1 - 1), min(j0 + dj, n1 - 1)
        w[i * n1 + j_] += ww
    return w


def lb_fluid(starts, goals, N, h, vmax_mult=1.5):
    M = len(starts)
    dt = T / N
    n1 = int(round(1.0 / h))
    xs = (np.arange(n1) + 0.5) * h
    cy, cx = np.meshgrid(xs, xs, indexing="ij")
    cells = np.stack([cx.ravel(), cy.ravel()], axis=1)
    C = len(cells)
    snap = h * np.sqrt(2)
    vnom = max(np.linalg.norm(goals[j] - starts[j]) / T for j in range(M))
    hop = vmax_mult * vnom * dt + snap

    # arcs, built vectorized on index offsets
    r = int(np.ceil(hop / h))
    off = [(di, dj) for di in range(-r, r + 1) for dj in range(-r, r + 1)
           if h * np.hypot(di, dj) <= hop]
    af, at, ac = [], [], []
    ii, jj = np.divmod(np.arange(C), n1)
    for (di, dj) in off:
        ok = (ii + di >= 0) & (ii + di < n1) & (jj + dj >= 0) & (jj + dj < n1)
        src = np.nonzero(ok)[0]
        dst = src + di * n1 + dj
        d2 = (h * np.hypot(di, dj)) ** 2
        af.append(src); at.append(dst)
        ac.append(np.full(len(src), max(0.0, d2 - h * h) / dt))
    af = np.concatenate(af); at = np.concatenate(at); ac = np.concatenate(ac)
    A_ = len(af)
    O = sparse.csr_matrix((np.ones(A_), (af, np.arange(A_))), shape=(C, A_))
    I = sparse.csr_matrix((np.ones(A_), (at, np.arange(A_))), shape=(C, A_))

    # exclusion disk membership D (probe x cell)
    Rex = DELTA / 2 - h / np.sqrt(2)
    rr = int(np.floor(Rex / h))
    ents_r, ents_c = [], []
    for di in range(-rr, rr + 1):
        for dj in range(-rr, rr + 1):
            if h * np.hypot(di, dj) <= Rex:
                ok = (ii + di >= 0) & (ii + di < n1) & (jj + dj >= 0) & (jj + dj < n1)
                src = np.nonzero(ok)[0]
                ents_r.append(src); ents_c.append(src + di * n1 + dj)
    D = sparse.csr_matrix((np.ones(sum(len(e) for e in ents_r)),
                           (np.concatenate(ents_r), np.concatenate(ents_c))),
                          shape=(C, C))

    # variables: f[j,k] (A_ each, k=0..N-1), rho[j,k] (C each, k=1..N-1)
    nf, nr = M * N * A_, M * (N - 1) * C
    fbase = lambda j, k: (j * N + k) * A_
    rbase = lambda j, k: nf + (j * (N - 1) + (k - 1)) * C
    cost = np.concatenate([np.tile(ac, M * N), np.zeros(nr)])

    Z_CA = sparse.csr_matrix((C, A_)); Id = sparse.identity(C, format="csr")
    eq_blocks, eq_rhs = [], []
    for j in range(M):
        w0 = bilinear(starts[j], n1, h)
        wT = bilinear(goals[j], n1, h)
        # k=0 outflow = w0
        row = sparse.hstack([O if (jj2 == j and kk == 0) else Z_CA
                             for jj2 in range(M) for kk in range(N)] +
                            [sparse.csr_matrix((C, nr))], format="csr")
        eq_blocks.append(row); eq_rhs.append(w0)
        for k in range(1, N):
            # inflow(k-1) - rho(k) = 0 ; outflow(k) - rho(k) = 0
            for mat, kk_f in ((I, k - 1), (O, k)):
                cols_f = sparse.hstack([mat if (jj2 == j and kk == kk_f) else Z_CA
                                        for jj2 in range(M) for kk in range(N)],
                                       format="csr")
                cols_r = sparse.lil_matrix((C, nr))
                cols_r[:, rbase(j, k) - nf:rbase(j, k) - nf + C] = -Id
                eq_blocks.append(sparse.hstack([cols_f, cols_r.tocsr()], format="csr"))
                eq_rhs.append(np.zeros(C))
        # goal inflow = wT
        row = sparse.hstack([I if (jj2 == j and kk == N - 1) else Z_CA
                             for jj2 in range(M) for kk in range(N)] +
                            [sparse.csr_matrix((C, nr))], format="csr")
        eq_blocks.append(row); eq_rhs.append(wT)
    A_eq = sparse.vstack(eq_blocks, format="csr")
    b_eq = np.concatenate(eq_rhs)

    # capacity: sum_j D rho(j,k) <= 1 for k=1..N-1
    ub_blocks = []
    for k in range(1, N):
        cols = sparse.lil_matrix((C, nf + nr))
        for j in range(M):
            cols[:, rbase(j, k):rbase(j, k) + C] = D
        ub_blocks.append(cols.tocsr())
    A_ub = sparse.vstack(ub_blocks, format="csr")
    b_ub = np.ones(A_ub.shape[0])

    t0 = time.perf_counter()
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    free = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    return (float(res.fun) if res.status == 0 else None,
            float(free.fun) if free.status == 0 else None,
            round(time.perf_counter() - t0, 1), nf + nr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--h", type=float, default=0.03)
    ap.add_argument("--N", type=int, default=10)
    a = ap.parse_args()
    out = []
    for M in a.Ms:
        for s in a.seeds:
            starts, goals = instance(M, s)
            lbn = lb_nominal(starts, goals)
            Jt, Ja = rigid_pipeline(starts, goals, a.N)
            lbf, lbf_free, tlp, nv = lb_fluid(starts, goals, a.N, a.h)
            row = dict(M=M, seed=s, LB_nom=round(lbn, 4),
                       LB_fluid=round(lbf, 4) if lbf else None,
                       LB_fluid_free=round(lbf_free, 4) if lbf_free else None,
                       J_tubes=round(Jt, 4) if Jt else None,
                       J_alt=round(Ja, 4) if Ja else None,
                       lp_time=tlp, lp_vars=nv)
            if lbf and lbf_free and Ja and Ja > lbn + 1e-9:
                dlb = lbf - lbf_free   # congestion increment, bias cancelled
                row["congestion_increment"] = round(dlb, 4)
                row["premium"] = round(Ja - lbn, 4)
                row["recovery_pct"] = round(100 * dlb / (Ja - lbn), 1)
            out.append(row)
            print(json.dumps(row), flush=True)
    p = Path(__file__).with_name("results") / "eulerian_lb_toy.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
