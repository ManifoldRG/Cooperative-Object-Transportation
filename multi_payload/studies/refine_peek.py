"""User's refinement idea, measured: pure projections (no smoothing) on a
time grid that is locally refined wherever consecutive spheres of a payload
fail the A4 overlap test. Refinement is GLOBAL in time (all payloads get the
inserted step, interpolated) so same-step safety semantics stay valid.
Compared against gauss and pure-projection on the uniform grid, same
instances, fuel normalized to the uniform grid's metric (sum ||F|| dt / dt0).

Usage: py refine_peek.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_methods import relax_iso  # noqa: E402
from benchmark_tubes import tubes_detail  # noqa: E402
from multi_payload import scenario, translation  # noqa: E402

import cvxpy as cp  # noqa: E402
from spacecraft_libraries.new_opts import th_psi_matrix  # noqa: E402

CAP = 150.0


def eps_half(x, pairs, deltas, M):
    K = len(x[0]) - 1
    eps = [np.full(K + 1, CAP) for _ in range(M)]
    for (j, l) in pairs:
        d = np.linalg.norm(x[j] - x[l], axis=1)
        m = 0.5 * (d - deltas[j][l])
        eps[j] = np.minimum(eps[j], m)
        eps[l] = np.minimum(eps[l], m)
    return eps


def relax_pure(x, pairs, tgt, max_sweeps=4000, tol=0.1):
    """Pure same-step projections on the current grid. Indices 0,1,K frozen;
    frozen-frozen steps excluded from the residual (safe by placement)."""
    K = len(x[0]) - 1
    sl = slice(2, K)   # active indices
    sweeps, worst = 0, np.inf
    while sweeps < max_sweeps and worst >= tol:
        worst = 0.0
        for (j, l) in pairs:
            d = x[j][sl] - x[l][sl]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            need = tgt[(j, l)] - dist
            m = need > 0
            if m.any():
                worst = max(worst, float(need[m].max()))
                g = d[m] / dist[m][:, None]
                x[j][sl][m] += 0.5 * need[m][:, None] * g
                x[l][sl][m] -= 0.5 * need[m][:, None] * g
        sweeps += 1
    return x, sweeps, float(worst)


def refine_once(times, x, pairs, deltas, M):
    """Insert a global midpoint time inside every segment where some payload
    fails A4 overlap: ||x[i+1]-x[i]|| > eps_i + eps_{i+1}."""
    eps = eps_half(x, pairs, deltas, M)
    K = len(times) - 1
    bad = set()
    for j in range(M):
        step = np.linalg.norm(np.diff(x[j], axis=0), axis=1)
        for i in range(1, K):   # never split the pinned 0-1 segment
            if step[i] > eps[j][i] + eps[j][i + 1]:
                bad.add(i)
    if not bad:
        return times, x, 0
    new_t, new_x = [], [[] for _ in range(M)]
    for i in range(K):
        new_t.append(times[i])
        for j in range(M):
            new_x[j].append(x[j][i])
        if i in bad:
            new_t.append(0.5 * (times[i] + times[i + 1]))
            for j in range(M):
                new_x[j].append(0.5 * (x[j][i] + x[j][i + 1]))
    new_t.append(times[K])
    for j in range(M):
        new_x[j].append(x[j][K])
    return np.array(new_t), [np.array(v) for v in new_x], len(bad)


def tubes_vardt(sc, x, eps, times):
    """Decoupled tube solves on the non-uniform grid; fuel = sum ||F|| dt."""
    K = len(times) - 1
    dts = np.diff(times)
    psis = [np.asarray(th_psi_matrix(sc.mu, sc.a, sc.e, t))[3:6, :]
            for t in times[:-1]]
    fuel = 0.0
    for j in range(sc.M):
        P = sc.payloads[j]
        r = cp.Variable((K + 1, 3))
        v = cp.Variable((K + 1, 3))
        F = cp.Variable((K, 3))
        cons = [r[0] == P.x0.r, v[0] == P.x0.v, r[K] == P.xf.r, v[K] == P.xf.v]
        for k in range(K):
            Ps = psis[k]
            cons += [r[k + 1] == r[k] + dts[k] * v[k],
                     v[k + 1] == v[k] + dts[k] * (Ps[:, :3] @ r[k] +
                                                  Ps[:, 3:] @ v[k] + F[k] / P.m)]
        for k in range(1, K):
            if eps[j][k] < CAP:
                cons.append(cp.norm(r[k] - x[j][k]) <= eps[j][k])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(dts, cp.norm(F, 2, axis=1)))), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        fuel += float(prob.value)
    return fuel


def run(M, seed, N=30, tf=500.0, headroom=60.0, rounds=4):
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
    row = dict(M=M, seed=seed, LB=round(LB, 1))

    # uniform-grid baselines
    for name, al in (("gauss", 0.25), ("pure", 0.0)):
        xx, sw, res = relax_iso(rbar, pairs, tgt, N, M, alpha=al)
        if xx is None:
            row[name] = f"stall({res:.1f})"
            continue
        J, _, _ = tubes_detail(sc, psis, pairs, deltas, xx, N, M, alloc="half")
        row[name] = round(100 * (J - LB) / LB, 2) if J else None

    # pure projections + A4 refinement
    t0 = time.perf_counter()
    times = np.linspace(0.0, tf, N + 1)
    x = [r.copy() for r in rbar]
    inserted = []
    for _ in range(rounds):
        x, sw, res = relax_pure(x, pairs, tgt)
        if res >= 0.1:
            row["refined"] = f"stall({res:.1f})"
            break
        times, x, nbad = refine_once(times, x, pairs, deltas, M)
        inserted.append(nbad)
        if nbad == 0:
            break
    else:
        x, sw, res = relax_pure(x, pairs, tgt)
    if "refined" not in row:
        eps = eps_half(x, pairs, deltas, M)
        Jr = tubes_vardt(sc, x, eps, times)
        row["t_refined"] = round(time.perf_counter() - t0, 2)
        row["grid"] = len(times) - 1
        row["inserted_per_round"] = inserted
        row["refined"] = (round(100 * (Jr / dt0 - LB) / LB, 2) if Jr else None)
        # exact same-step verification on the final grid
        row["min_margin"] = round(min(
            float(np.min(np.linalg.norm(x[j] - x[l], axis=1))) - deltas[j][l]
            for (j, l) in pairs), 1)
    return row


def main():
    out = []
    for (M, s) in [(20, 0), (20, 1), (30, 0), (30, 1)]:
        try:
            r = run(M, s)
        except Exception as exc:
            r = dict(M=M, seed=s, status=f"error: {str(exc)[:90]}")
        out.append(r)
        print(json.dumps(r), flush=True)
    p = Path(__file__).with_name("results") / "refine_peek.json"
    p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
