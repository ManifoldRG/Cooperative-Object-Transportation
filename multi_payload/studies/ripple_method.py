"""Ripple method (user's): one-way geometric propagation of repulsions.

Per round, every same-step overlapping pair (j, l) is pushed apart by exactly
its overlap: point (j,k) moves by e = need/2 along the separation direction,
(l,k) by -e. Each push then ripples outward along its own chain in both
directions, k +/- 1 by e*r, k +/- 2 by e*r^2, ... (r = decay, 1/2 in the
original statement), each point touched once per push, stopping at pinned
steps. A ripple step is applied only if it moves that point no closer to any
other payload's point at the same timestep; the first blocked point stops the
ripple in that direction (a blocked link does not transmit). Endpoints and
step 1 (pinned by the boundary conditions) never move.

Rounds repeat until no overlap (success) or stagnation (total overlap not
improving over a window); stagnation doubles the timestep count, replans the
nominals and restarts, then bisects back for the smallest count that works.
Hand-off: half-split tubes + decoupled convex re-plans, certified gap vs LB.

Usage: py ripple_method.py [--decays 0.5 0.75 0.9] [--Ms 20 30] [--seeds 0 1]
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
from benchmark_tubes import tubes_detail  # noqa: E402
from chain_method import nominals  # noqa: E402
from multi_payload import scenario  # noqa: E402

HEADROOM = 60.0


def ripple(P, j, k, e, r, K, tgt_row):
    """Apply push e at (j,k) and ripple it outward along chain j. A ripple
    step is blocked only if it would create or deepen an overlap: approaching
    a payload that ends up below its target. Far-away payloads don't block,
    so min(d, target) is nondecreasing for every pair under ripples."""
    P[j, k] += e
    M = P.shape[0]
    others = np.arange(M) != j
    t2 = (tgt_row[others] ** 2)[:, None]                            # (M-1,1)
    for direc in (1, -1):
        idx = np.arange(k + direc, K if direc > 0 else 1, direc)   # free steps
        if not len(idx):
            continue
        n = np.abs(idx - k)
        d = (r ** n)[:, None] * e[None, :]                          # (n,3)
        cur = P[j, idx]                                             # (n,3)
        oth = P[others][:, idx]                                     # (M-1,n,3)
        before = np.sum((cur[None] - oth) ** 2, axis=2)
        after = np.sum((cur[None] + d[None] - oth) ** 2, axis=2)
        bad = (after < before - 1e-9) & (after < t2)
        ok = ~np.any(bad, axis=0)                                   # (n,)
        stop = np.argmin(ok) if not ok.all() else len(idx)
        if stop:
            P[j, idx[:stop]] += d[:stop]


def deconflict(rbar, pairs, tgt, r, max_rounds=3000, window=100, alpha=None):
    """alpha: if set, each overlapping pair separates by at most alpha per
    round (alpha/2 per side) instead of the full overlap at once."""
    P = np.array(rbar, float)                                       # (M,K+1,3)
    K = P.shape[1] - 1
    M = P.shape[0]
    T = np.zeros((M, M))
    for (j, l), v in tgt.items():
        T[j, l] = T[l, j] = v
    best, best_it = np.inf, 0
    worst = np.inf
    for it in range(max_rounds):
        worst, total = 0.0, 0.0
        for (j, l) in pairs:
            d = P[j, 2:K] - P[l, 2:K]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            need = tgt[(j, l)] - dist
            for i in np.nonzero(need > 0)[0]:
                k = i + 2
                dd = P[j, k] - P[l, k]
                dn = max(float(np.linalg.norm(dd)), 1e-9)
                nd = tgt[(j, l)] - dn
                if nd <= 0:
                    continue
                worst = max(worst, nd)
                total += nd
                step = nd if alpha is None else min(nd, alpha)
                e = 0.5 * step * dd / dn
                ripple(P, j, k, e, r, K, T[j])
                ripple(P, l, k, -e, r, K, T[l])
        if worst < 0.1:
            return P, it + 1, worst
        if total < best - 0.05:
            best, best_it = total, it
        elif it - best_it >= window:
            if worst <= HEADROOM / 2:     # only the headroom is short: accept
                return P, it + 1, worst
            return None, it + 1, worst
    return None, max_rounds, worst


def run(M, seed, r, K0=30, tf=500.0, Kmax=240, alpha=None):
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    sc0 = scenario.generate("random_seq", M=M, n_points=8, n_agents=3 * M,
                            seed=seed, nu_deg=60, N=K0, tf=tf,
                            ring_radius=ring, max_tries=5000)
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc0.payloads]
    margin = sc0.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    rho = [e + margin / 2 + HEADROOM / 2 for e in ext]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    tgt = {(j, l): rho[j] + rho[l] for (j, l) in pairs}
    row = dict(M=M, seed=seed, decay=r)

    cache = {}

    def nom(K):
        if K not in cache:
            cache[K] = nominals(sc0, K)
        return cache[K]

    sc, psis, rbar, LB0 = nom(K0)
    xg, _, _ = relax_iso(rbar, pairs, tgt, K0, M)
    if xg is not None:
        Jg, _, _ = tubes_detail(sc, psis, pairs, deltas, xg, K0, M)
        row["gap_gauss_pct"] = round(100 * (Jg - LB0) / LB0, 2) if Jg else None

    t0 = time.perf_counter()
    attempts = {}

    def attempt(K):
        if K not in attempts:
            attempts[K] = deconflict(nom(K)[2], pairs, tgt, r, alpha=alpha,
                                     max_rounds=20000 if alpha else 3000)
        return attempts[K]

    K, fail_K = K0, None
    while attempt(K)[0] is None:
        fail_K = K
        if K >= Kmax:
            row["status"] = f"stagnated up to Kmax (resid {attempt(K)[2]:.1f})"
            row["attempts"] = {k: [v[1], round(v[2], 1)] for k, v in attempts.items()}
            return row
        K *= 2
    if fail_K is not None:
        lo, hi = fail_K, K
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if attempt(mid)[0] is not None:
                hi = mid
            else:
                lo = mid
        K = hi
    P = attempt(K)[0]
    row["K_final"] = K
    row["attempts"] = {k: [v[1], round(v[2], 1)] for k, v in sorted(attempts.items())}
    row["t_ripple"] = round(time.perf_counter() - t0, 1)

    sck, psk, rbk, LBk = nom(K)
    x = [P[j].copy() for j in range(M)]
    Jc, rs, _ = tubes_detail(sck, psk, pairs, deltas, x, K, M)
    if Jc is None:
        row["status"] = "tube re-plan infeasible"
        return row
    row["gap_ripple_pct"] = round(100 * (Jc - LBk) / LBk, 2)
    row["max_dev"] = round(max(float(np.max(np.linalg.norm(x[j] - rbk[j], axis=1)))
                               for j in range(M)), 0)
    row["max_link_over_2rho"] = round(max(
        float(np.max(np.linalg.norm(np.diff(x[j], axis=0), axis=1))) / (2 * rho[j])
        for j in range(M)), 2)
    row["min_margin"] = round(min(
        float(np.min(np.linalg.norm(rs[j] - rs[l], axis=1))) - deltas[j][l]
        for (j, l) in pairs), 1)
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decays", type=float, nargs="+", default=[0.5, 0.75, 0.9])
    ap.add_argument("--Ms", type=int, nargs="+", default=[20, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--alphas", type=float, nargs="+", default=[None],
                    help="per-round separation step (m); omit for full overlap")
    a = ap.parse_args()
    out = []
    for alpha in a.alphas:
        for r in a.decays:
            for M in a.Ms:
                for s in a.seeds:
                    try:
                        row = run(M, s, r, alpha=alpha)
                    except Exception as exc:
                        row = dict(M=M, seed=s, decay=r,
                                   status=f"error: {str(exc)[:90]}")
                    row["alpha"] = alpha
                    out.append(row)
                    print(json.dumps(row), flush=True)
                    tag = "full" if alpha is None else "steps"
                    p = (Path(__file__).with_name("results") /
                         f"ripple_method_{tag}.json")
                    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
