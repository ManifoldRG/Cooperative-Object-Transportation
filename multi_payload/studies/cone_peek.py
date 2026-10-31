"""Monotone cone-step deconfliction (user's method), two variants, vs gauss.

Per round, while any relevant pair is below target: EVERY free point takes a
step alpha along the most fuel-efficient direction inside its away-cone
{d : d.(x - y) >= 0 for every other-payload point y within awareness radius}.
Preferred direction = back toward the nominal, projected into the cone
(cyclic halfspace projections); a point at its nominal that is locally
conflicted uses away-from-nearest instead; empty cone => the point holds.
Monotone: for any alpha, simultaneous moves never decrease a constrained
pairwise distance (cross terms are nonnegative).

Variant "step":    cones and target over same-timestep pairs only.
Variant "alltime": cones and target over ALL cross-time pairs (spatial).

Between rounds, segments failing A4 overlap get a global midpoint timestep
(all payloads interpolated). Finishes with the variable-dt tube solve.

Usage: py cone_peek.py [--alpha 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_methods import relax_iso  # noqa: E402
from benchmark_tubes import tubes_detail  # noqa: E402
from refine_peek import eps_half, refine_once, tubes_vardt  # noqa: E402
from multi_payload import scenario, translation  # noqa: E402


def _project_cone(v, normals):
    """Cyclic projection of v into {d : d.n >= 0}; unit vector or None."""
    for _ in range(12):
        dots = normals @ v
        i = int(np.argmin(dots))
        if dots[i] >= -1e-9:
            break
        v = v - dots[i] * normals[i]
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            return None
        v = v / nv
    if (normals @ v).min() < -1e-6:
        return None
    return v


def cone_dir(x, xbar, normals, viol_normals, locally_conflicted, tangent):
    """Direction in the not-toward cone {d : d.n >= 0 for all n}, preferring
    fuel (toward nominal). When the fuel-greedy direction is anti-normal its
    projection annihilates (every tangent ties the argmin): break the tie by
    sliding along the trajectory's local tangent. Unit vector or None."""
    u = xbar - x
    nu = np.linalg.norm(u)
    if nu < 1e-6:
        if not locally_conflicted:
            return None          # at nominal, nobody near: stay
        u = (viol_normals if len(viol_normals) else normals).mean(axis=0)
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            return None          # surrounded symmetrically: locked
    v = u / nu
    if not len(normals):
        return v
    out = _project_cone(v, normals)
    if out is not None:
        return out
    nt = np.linalg.norm(tangent)
    if nt > 1e-9:
        for cand in (tangent / nt, -tangent / nt):
            out = _project_cone(cand, normals)
            if out is not None:
                return out
    return None


def cone_relax(rbar_f, x, pairs, tgt, deltas, M, mode, alpha, max_iters=1200):
    """One cone-step loop on the current grid. Returns (x, iters, resid,
    locked_frac, resid_history). Exits early if the residual stops moving
    (< 0.01 m change over a 500-iteration window): locked, not slow."""
    K = len(x[0]) - 1
    free = np.zeros(K + 1, bool)
    free[2:K] = True
    locked_hits, moves = 0, 0
    hist, last_check = [], (0, np.inf)
    for it in range(max_iters):
        # residual + per-point constraint gathering
        worst = 0.0
        cons = [[[] for _ in range(K + 1)] for _ in range(M)]
        viol = [[[] for _ in range(K + 1)] for _ in range(M)]
        near = [[False] * (K + 1) for _ in range(M)]
        for (j, l) in pairs:
            t = tgt[(j, l)]
            # "minmono": a pair is a constraint only if it is violated or
            # could be closed to below target this round (two points closing
            # at most 2*alpha). Non-violated pairs may approach, but never
            # below target, so min(d, target) is nondecreasing for every pair.
            R = t + (2 * alpha + 1e-6 if mode == "minmono" else 4 * alpha)
            if mode in ("step", "minmono"):
                d = x[j] - x[l]
                dist = np.linalg.norm(d, axis=1)
                worst = max(worst, float(np.max(t - dist[2:K])) if K > 2 else 0)
                for k in np.nonzero(dist < R)[0]:
                    n = d[k] / max(dist[k], 1e-9)
                    if free[k]:
                        cons[j][k].append(n)
                        cons[l][k].append(-n)
                        if dist[k] < t:
                            near[j][k] = near[l][k] = True
                            viol[j][k].append(n)
                            viol[l][k].append(-n)
            else:
                D = cdist(x[j], x[l])
                sub = D[2:K][:, 2:K]
                if sub.size:
                    worst = max(worst, float(t - sub.min()))
                ks, kls = np.nonzero(D < R)
                for a, b in zip(ks, kls):
                    n = x[j][a] - x[l][b]
                    nn = np.linalg.norm(n)
                    n = n / max(nn, 1e-9)
                    if free[a]:
                        cons[j][a].append(n)
                        if nn < t:
                            near[j][a] = True
                            viol[j][a].append(n)
                    if free[b]:
                        cons[l][b].append(-n)
                        if nn < t:
                            near[l][b] = True
                            viol[l][b].append(-n)
        if it % max(1, max_iters // 8) == 0:
            hist.append(round(float(worst), 1))
        if worst < 0.1:
            return (x, it, float(worst),
                    locked_hits / max(1, moves + locked_hits), hist)
        if it - last_check[0] >= 500:
            if last_check[1] - worst < 0.01:
                return (x, it, float(worst),
                        locked_hits / max(1, moves + locked_hits), hist)
            last_check = (it, worst)
        for j in range(M):
            for k in range(2, K):
                ns = np.array(cons[j][k]) if cons[j][k] else np.zeros((0, 3))
                vs = np.array(viol[j][k]) if viol[j][k] else np.zeros((0, 3))
                v = cone_dir(x[j][k], rbar_f[j][k], ns, vs, near[j][k],
                             x[j][k + 1] - x[j][k - 1])
                if v is None:
                    if near[j][k]:
                        locked_hits += 1
                    continue
                step = min(alpha, np.linalg.norm(rbar_f[j][k] - x[j][k])
                           if not near[j][k] and len(ns) == 0 else alpha)
                x[j][k] = x[j][k] + step * v
                moves += 1
    return (x, max_iters, float(worst),
            locked_hits / max(1, moves + locked_hits), hist)


def run(M, seed, mode, alpha, N=30, tf=500.0, headroom=60.0, rounds=4):
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
    dt0 = tf / N
    out = dict(mode=mode)
    if mode == "alltime":
        # pinned-vs-pinned spatial conflicts are unfixable on this layout
        frozen = [0, 1, N]
        for (j, l) in pairs:
            if cdist(rbar[j][frozen], rbar[l][frozen]).min() < deltas[j][l]:
                return dict(mode=mode, status="structurally infeasible")
        for (j, l) in pairs:   # exclude pinned-pinned from the residual via tgt on free block only (handled in cone_relax residual slice)
            pass
    t0 = time.perf_counter()
    times = np.linspace(0.0, tf, N + 1)
    x = [r.copy() for r in rbar]
    rbar_f = [r.copy() for r in rbar]
    iters_hist, lock_hist = [], []
    budget = max(1500, int(round(4500 / max(alpha, 0.3))))
    for _ in range(rounds):
        x, iters, resid, lockf, hist = cone_relax(rbar_f, x, pairs, tgt,
                                                  deltas, M, mode, alpha,
                                                  max_iters=budget)
        iters_hist.append(iters)
        lock_hist.append(round(lockf, 3))
        if resid >= 0.1:
            out["status"] = f"stalled ({resid:.1f})"
            out["resid_history"] = hist
            out["iters"] = iters_hist
            return out
        times, x, nbad = refine_once(times, x, pairs, deltas, M)
        if nbad == 0:
            break
        rbar_f = [np.array([np.interp(times, np.linspace(0, tf, len(r)),
                                      r[:, c]) for c in range(3)]).T
                  if len(r) != len(times) else r for r in rbar_f]
        rbar_f = [r.copy() for r in x] if False else rbar_f
    out["t_relax"] = round(time.perf_counter() - t0, 1)
    out["iters"] = iters_hist
    out["locked_frac"] = lock_hist
    out["grid"] = len(times) - 1
    eps = eps_half(x, pairs, deltas, M)
    J = tubes_vardt(sc, x, eps, times)
    if J:
        out["gap_pct"] = round(100 * (J / dt0 - LB) / LB, 2)
    out["dev"] = round(max(float(np.max(np.linalg.norm(
        x[j][:1] * 0 + x[j] - np.array([np.interp(times, np.linspace(0, tf, N + 1), rbar[j][:, c]) for c in range(3)]).T, axis=1))) for j in range(M)), 0)
    out["status"] = "ok"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--mode", default="step", choices=["step", "minmono"])
    a = ap.parse_args()
    out = []
    for (M, s) in [(20, 0), (20, 1)]:
        for alpha in a.alphas:
            row = dict(M=M, seed=s, alpha=alpha)
            try:
                row[a.mode] = run(M, s, a.mode, alpha)
            except Exception as exc:
                row[a.mode] = dict(status=f"error: {str(exc)[:80]}")
            out.append(row)
            print(json.dumps(row), flush=True)
    p = Path(__file__).with_name("results") / f"cone_sweep_{a.mode}.json"
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
