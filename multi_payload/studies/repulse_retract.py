"""Repulse-then-retract (user's method): full deconfliction first, then
walk every chain back toward its fuel-optimal nominal.

Phase 1 (repel): every free point (j,k) is pushed by an inverse-square
repulsion from the other payloads' points at the same timestep,
    f = sum_l T_jl^2 (x_j - x_l) / |x_j - x_l|^3,
stepping alpha * min(1, |f|) along f, with the Rigid Link constraint
respected (a step that would stretch a link past L_j is halved, else the
point holds). Fuel is ignored. Rounds repeat until every pair clears its
target (same timestep) and every step segment clears delta; if progress
stalls the timestep count is doubled and phase 1 restarts.

Phase 2 (retract): chain by chain (in payload order), point by point from
the first free step to the last, each point steps up to alpha toward its
nominal point, projected so it approaches no nearby point and lengthens no
near-limit link, then checked exactly: no same-step pair below its target,
neither adjacent segment closer than delta to any other payload's segment,
both links within L_j (step halved up to 3 times, else hold). Rounds repeat
until nothing moves. Safe at every step, by the same checks.

Hand-off: segment-sized corridors + decoupled fuel re-plans (safe at all
times), certified gap vs LB; compared with gauss (h=30) on the same instance.

Usage: py repulse_retract.py [--hs 1 15 30] [--Ms 20 30] [--seeds 0 1]
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
from chain_method import _project_cone, links_ok, nominals, step_targets  # noqa: E402
from expand_method import _seg_closest, cont_margin, seg_tubes  # noqa: E402
from multi_payload import scenario  # noqa: E402


def seg_dist(a0, a1, b0, b1):
    """Closest approach of two payloads moving linearly over one step."""
    d0, d1 = a0 - b0, a1 - b1
    e = d1 - d0
    ee = float(e @ e)
    t = 0.0 if ee < 1e-12 else min(1.0, max(0.0, -float(d0 @ e) / ee))
    return float(np.linalg.norm(d0 + t * e))


def overlap(P, T, K):
    M = P.shape[0]
    worst = 0.0
    for j in range(M):
        for l in range(j + 1, M):
            d = np.linalg.norm(P[j, 2:K] - P[l, 2:K], axis=1)
            worst = max(worst, float(np.max(T[j, l, 2:K] - d)))
    return worst


def seg_clear(P, D, K):
    """Worst segment deficit below delta over all pairs and steps."""
    M = P.shape[0]
    worst = 0.0
    for j in range(M):
        for l in range(j + 1, M):
            dist, _ = _seg_closest(P[j] - P[l])
            worst = max(worst, float(np.max(D[j, l] - dist)))
    return worst


def link_ok_at(P, j, k, new, L):
    for q in (k - 1, k + 1):
        if np.linalg.norm(new - P[j, q]) > max(L, np.linalg.norm(P[j, k] - P[j, q])) + 1e-9:
            return False
    return True


def repel(P, T, D, L, K, alpha, max_rounds=3000, window=200):
    M = P.shape[0]
    best, best_it = np.inf, 0
    for it in range(max_rounds):
        w1, w2 = overlap(P, T, K), seg_clear(P, D, K)
        if w1 < 0.1 and w2 < 1e-6:
            return True, it
        score = w1 + w2
        if score < best - 0.05:
            best, best_it = score, it
        elif it - best_it >= window:
            return False, it
        for j in range(M):
            oth = np.arange(M) != j
            for k in range(2, K):
                d = P[j, k][None] - P[oth, k]
                dn = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                f = np.sum((T[j, oth, k] ** 2 / dn ** 3)[:, None] * d, axis=0)
                fn = float(np.linalg.norm(f))
                if fn < 1e-12:
                    continue
                step = alpha * min(1.0, fn)
                for _ in range(4):
                    new = P[j, k] + step * f / fn
                    if link_ok_at(P, j, k, new, L[j]):
                        P[j, k] = new
                        break
                    step *= 0.5
    return False, max_rounds


def retract(P, R, T, D, L, K, alpha, max_rounds=400):
    M = P.shape[0]
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        moved = 0
        for j in range(M):
            oth = np.arange(M) != j
            for k in range(2, K):
                p = P[j, k]
                g = R[j, k] - p
                gn = float(np.linalg.norm(g))
                if gn < 1e-3:
                    continue
                O = P[oth, k]
                dvec = p[None] - O
                dist = np.linalg.norm(dvec, axis=1)
                tg = T[j, oth, k]
                mag = min(alpha, gn)
                normals = [dv / max(dn, 1e-9) for dv, dn, tt in zip(dvec, dist, tg)
                           if dn < tt + mag]
                for q in (k - 1, k + 1):
                    e = p - P[j, q]
                    en = np.linalg.norm(e)
                    if en > L[j] - mag and en > 1e-9:
                        normals.append(-e / en)
                v = _project_cone(g / gn, np.array(normals) if normals else np.zeros((0, 3)))
                if v is None:
                    continue
                step = mag * max(float(v @ (g / gn)), 0.0)
                for _ in range(4):
                    if step < 1e-3:
                        break
                    new = p + step * v
                    ok = np.all(np.linalg.norm(new[None] - O, axis=1) >= tg - 1e-9)
                    ok = ok and link_ok_at(P, j, k, new, L[j])
                    if ok:
                        idx = np.nonzero(oth)[0]
                        for l in idx:
                            for (a, b) in ((k - 1, k), (k, k + 1)):
                                pa = new if a == k else P[j, a]
                                pb = new if b == k else P[j, b]
                                if seg_dist(pa, pb, P[l, a], P[l, b]) < D[j, l] - 1e-9:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if ok:
                        P[j, k] = new
                        moved += 1
                        break
                    step *= 0.5
        if moved == 0:
            break
    return rounds


def run(M, seed, h, alpha=5.0, link_rho=1.0, K0=30, tf=500.0, Kmax=240,
        gauss_h=30.0):
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    sc0 = scenario.generate("random_seq", M=M, n_points=8, n_agents=3 * M,
                            seed=seed, nu_deg=60, N=K0, tf=tf,
                            ring_radius=ring, max_tries=5000)
    ext = [float(np.max(np.linalg.norm(Pl.points, axis=1))) for Pl in sc0.payloads]
    margin = sc0.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    rho = [e + margin / 2 + h / 2 for e in ext]
    L = [link_rho * r for r in rho]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    row = dict(M=M, seed=seed, h=h, alpha=alpha)

    # gauss baseline (safe at all times) at its own headroom
    sc, psis, rbar, LB0 = nominals(sc0, K0)
    rho_g = [e + margin / 2 + gauss_h / 2 for e in ext]
    xg, _, _ = relax_iso(rbar, pairs, {(j, l): rho_g[j] + rho_g[l] for (j, l) in pairs},
                         K0, M)
    if xg is not None:
        Jg, _ = seg_tubes(sc, psis, [np.asarray(v) for v in xg], pairs, deltas, K0, M)
        row["gap_gauss_pct"] = round(100 * (Jg - LB0) / LB0, 2) if Jg else None

    t0 = time.perf_counter()
    K = K0
    while not links_ok(nominals(sc0, K)[2], L):
        K *= 2
    while True:
        sck, psk, rbk, LBk = nominals(sc0, K)
        tg = step_targets(rbk, pairs, deltas, L, h)
        T = np.zeros((M, M, K + 1))
        Dm = np.zeros((M, M))
        for (j, l) in pairs:
            T[j, l] = T[l, j] = tg[(j, l)]
            Dm[j, l] = Dm[l, j] = deltas[j][l]
        P = np.array(rbk, float)
        ok, it1 = repel(P, T, Dm, L, K, alpha)
        if ok or K >= Kmax:
            break
        K *= 2
    row["K"] = K
    row["repel_rounds"] = it1
    if not ok:
        row["status"] = "repulsion did not clear"
        return row
    row["dev_after_repel"] = round(float(max(np.max(np.linalg.norm(P[j] - rbk[j], axis=1))
                                            for j in range(M))), 0)
    row["t_repel"] = round(time.perf_counter() - t0, 1)
    t1 = time.perf_counter()
    row["retract_rounds"] = retract(P, np.array(rbk), T, Dm, L, K, alpha)
    row["t_retract"] = round(time.perf_counter() - t1, 1)
    row["dev_after_retract"] = round(float(max(np.max(np.linalg.norm(P[j] - rbk[j], axis=1))
                                              for j in range(M))), 0)
    Jc, rs = seg_tubes(sck, psk, [P[j] for j in range(M)], pairs, deltas, K, M)
    if Jc is None:
        row["status"] = f"segment tubes: {rs}"
        return row
    row["gap_pct"] = round(100 * (Jc - LBk) / LBk, 2)
    cw, cb, cn = cont_margin(rs, pairs, deltas)
    row["cont_min_margin"] = round(cw, 3)
    row["cont_violations"] = f"{cb}/{cn}"
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hs", type=float, nargs="+", default=[1.0, 15.0, 30.0])
    ap.add_argument("--Ms", type=int, nargs="+", default=[20, 30])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    a = ap.parse_args()
    out = []
    for h in a.hs:
        for M in a.Ms:
            for s in a.seeds:
                try:
                    row = run(M, s, h)
                except Exception as exc:
                    row = dict(M=M, seed=s, h=h, status=f"error: {str(exc)[:100]}")
                out.append(row)
                print(json.dumps(row), flush=True)
                p = Path(__file__).with_name("results") / "repulse_retract.json"
                p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
