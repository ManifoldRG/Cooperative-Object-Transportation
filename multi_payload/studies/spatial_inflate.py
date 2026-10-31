"""Spatial inflation vs same-step inflation.

Same-step inflation (the pipeline's relax_iso) separates trajectories at
EQUAL time indices only: time-sharing of space is preserved, tubes are
per-step balls, and safety is time-indexed.

Spatial inflation (this study) separates the trajectories as POINT SETS:
min over ALL index pairs (k, k') of ||x_j[k] - x_l[k']|| >= delta_jl + h.
The resulting tubes are pairwise disjoint as spatial regions, so an agent
that stays inside its own tube is safe with NO clock at all (schedule-free
execution, arbitrary delays). The price is extra deviation where trajectories
previously time-shared space; this study measures that price.

Both methods run on the same instances, drawn with the random_joint layout
(all 2M endpoints jointly pairwise-separated: a start within delta of another
agent's goal makes spatial disjointness structurally infeasible). Re-plans
use the same per-step-ball convex tube solve for both; for the spatial method
the radii come from the cross-time clearance, so its J is a conservative
upper bound (free retiming inside the tube could only improve it).

Usage: py spatial_inflate.py [--Ms 10 20 30 40 50] [--seeds 0 ... 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_methods import relax_iso  # noqa: E402
from benchmark_tubes import tubes_detail  # noqa: E402
from multi_payload import scenario, translation  # noqa: E402


def relax_spatial(rbar, pairs, tgt, deltas, N, M, max_sweeps=3000, alpha=0.25,
                  sigma=1.5, tol=0.1):
    """Cross-time PBD: push every violating point PAIR (k of j, k' of l)
    apart along its gap. Indices {0, 1, N} are pinned (boundary conditions);
    a pinned point pushes its full need onto the free one. Two pinned points
    closer than delta_jl make the instance structurally infeasible; pinned
    pairs that satisfy delta_jl but not the inflated target are safe and are
    simply excluded from pushes and from the termination residual."""
    x = [r.copy() for r in rbar]
    frozen = np.zeros(N + 1, bool)
    frozen[[0, 1, N]] = True

    # structural check: pinned-vs-pinned closer than the SAFETY clearance
    for (j, l) in pairs:
        D = cdist(rbar[j][frozen], rbar[l][frozen])
        if float(D.min()) < deltas[j][l] - 1e-9:
            return None, 0, float(D.min()), (j, l)

    def sweep(smooth):
        worst = 0.0
        disp = [np.zeros_like(x[j]) for j in range(M)]
        for (j, l) in pairs:
            D = cdist(x[j], x[l])
            viol = tgt[(j, l)] - D
            viol[np.ix_(frozen, frozen)] = 0.0   # unfixable but already safe
            ks, kls = np.nonzero(viol > 0)
            if len(ks) == 0:
                continue
            worst = max(worst, float(viol[ks, kls].max()))
            g = x[j][ks] - x[l][kls]
            dn = np.maximum(np.linalg.norm(g, axis=1), 1e-9)
            g = g / dn[:, None]
            need = viol[ks, kls]
            fj = ~frozen[ks]
            fl = ~frozen[kls]
            both = fj & fl
            share_j = np.where(both, 0.5, np.where(fj, 1.0, 0.0))
            share_l = np.where(both, 0.5, np.where(fl, 1.0, 0.0))
            np.add.at(disp[j], ks, (share_j * need)[:, None] * g)
            np.add.at(disp[l], kls, -(share_l * need)[:, None] * g)
        for j in range(M):
            x[j] += disp[j]
            if smooth:
                u = x[j] - rbar[j]
                us = gaussian_filter1d(u, sigma=sigma, axis=0)
                u = (1 - alpha) * u + alpha * us
                u[0] = 0.0
                u[1] = 0.0
                u[N] = 0.0
                x[j] = rbar[j] + u
        return worst

    sweeps, worst = 0, np.inf
    while sweeps < 800 and worst >= 1.0:
        worst = sweep(True)
        sweeps += 1
    while sweeps < max_sweeps and worst >= tol:
        worst = sweep(False)
        sweeps += 1
    return (x if worst < tol else None), sweeps, float(worst), None


def spatial_eps(x, pairs, deltas, N, M):
    """Per-step radii from CROSS-TIME clearance: disjoint spatial tubes."""
    eps = [np.full(N + 1, 150.0) for _ in range(M)]
    for (j, l) in pairs:
        D = cdist(x[j], x[l])
        m_j = 0.5 * (D.min(axis=1) - deltas[j][l])
        m_l = 0.5 * (D.min(axis=0) - deltas[j][l])
        eps[j] = np.minimum(eps[j], m_j)
        eps[l] = np.minimum(eps[l], m_l)
    return eps


def tubes_with_eps(sc, psis, x, eps, N, M):
    import cvxpy as cp
    import route2_multi as r2
    fuel = 0.0
    for j in range(M):
        r = cp.Variable((N + 1, 3))
        v = cp.Variable((N + 1, 3))
        F = cp.Variable((N, 3))
        cons = r2._dyn(sc, psis, r, v, F, j)
        for k in range(1, N):
            if eps[j][k] < 150.0:
                cons.append(cp.norm(r[k] - x[j][k]) <= eps[j][k])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
        prob.solve(solver="CLARABEL")
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        fuel += float(prob.value)
    return fuel


def run_instance(M, seed, N, tf, headroom=60.0):
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    for _ in range(8):
        try:
            sc = scenario.generate("random_joint", M=M, n_points=8,
                                   n_agents=3 * M, seed=seed, nu_deg=60, N=N,
                                   tf=tf, ring_radius=ring, max_tries=5000)
            break
        except RuntimeError:
            ring *= 1.15
    else:
        return dict(M=M, seed=seed, status="placement failed")
    psis = translation.psi_list(sc)
    refs = translation.solve_all(sc, psis=psis)
    if not all(r["ok"] for r in refs):
        return dict(M=M, seed=seed, status="nominal failed")
    LB = sum(r["fuel"] for r in refs)
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]
    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc.payloads]
    margin = sc.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    tgt = {(j, l): deltas[j][l] + headroom for (j, l) in pairs}
    row = dict(M=M, seed=seed, ring_radius=round(ring, 0), LB=round(LB, 2))

    # same-step pipeline on the same instance
    t0 = time.perf_counter()
    x1, sw1, res1 = relax_iso(rbar, pairs, tgt, N, M)
    row["t_relax_step"] = round(time.perf_counter() - t0, 2)
    row["sweeps_step"] = sw1
    if x1 is not None:
        t0 = time.perf_counter()
        fuel, rs, _ = tubes_detail(sc, psis, pairs, deltas, x1, N, M)
        row["t_tubes_step"] = round(time.perf_counter() - t0, 2)
        if fuel:
            row["J_step"] = round(fuel, 2)
            row["gap_step_pct"] = round(100 * (fuel - LB) / LB, 2)
    else:
        row["status_step"] = f"relax stalled ({res1:.2f})"

    # spatial pipeline
    t0 = time.perf_counter()
    x2, sw2, res2, bad = relax_spatial(rbar, pairs, tgt, deltas, N, M)
    row["t_relax_spatial"] = round(time.perf_counter() - t0, 2)
    row["sweeps_spatial"] = sw2
    if x2 is None:
        row["status_spatial"] = (f"structurally infeasible pair {bad}" if bad
                                 else f"relax stalled ({res2:.2f})")
        row["status"] = "ok"
        return row
    # exact verification of spatial disjointness
    mm = min(float(cdist(x2[j], x2[l]).min()) - deltas[j][l] for (j, l) in pairs)
    row["spatial_min_margin"] = round(mm, 1)
    row["max_dev_spatial"] = round(max(float(np.max(np.linalg.norm(
        x2[j] - rbar[j], axis=1))) for j in range(M)), 1)
    eps = spatial_eps(x2, pairs, deltas, N, M)
    t0 = time.perf_counter()
    fuel2 = tubes_with_eps(sc, psis, x2, eps, N, M)
    row["t_tubes_spatial"] = round(time.perf_counter() - t0, 2)
    if fuel2:
        row["J_spatial"] = round(fuel2, 2)
        row["gap_spatial_pct"] = round(100 * (fuel2 - LB) / LB, 2)
        if "J_step" in row:
            row["spatial_premium_pct"] = round(
                100 * (fuel2 - row["J_step"]) / row["J_step"], 2)
    row["status"] = "ok"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    a = ap.parse_args()
    rows = []
    for M in a.Ms:
        for s in a.seeds:
            try:
                row = run_instance(M, s, a.N, a.tf)
            except Exception as exc:
                row = dict(M=M, seed=s, status=f"error: {str(exc)[:100]}")
            rows.append(row)
            print(json.dumps(row), flush=True)
            p = Path(__file__).with_name("results") / "spatial_inflate.json"
            p.write_text(json.dumps(rows, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
