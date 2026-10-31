"""Tubes-only scaling benchmark with certificate: inflation (isotropic) +
M decoupled ball-corridor SOCPs, random scenarios, M = 10..100 step 10.

Per instance: nominals -> LB = sum ell_j; vectorized isotropic relaxation;
per-payload tube SOCPs (fully decoupled). Reports LB, J_tubes and the
certified suboptimality gap (J_tubes - LB)/LB, per-stage times, the slowest
single tube solve (= parallel wall-clock), and the exact pairwise separation
margin. If any tube is infeasible, the relaxation is re-run once with doubled
headroom.

Usage: py benchmark_tubes.py [--Ms 10 20 ... 100] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import cvxpy as cp  # noqa: E402

from benchmark_methods import relax_iso  # noqa: E402
from multi_payload import scenario, translation  # noqa: E402
import route2_multi as r2  # noqa: E402

CAP = 150.0


def waterfill_radii(x, deltas, N, M):
    """Uniform ball growth with freeze-on-contact, per timestep: the
    lexicographic-maximin allocation of the pairwise surpluses. Dominates the
    half-split radii pointwise (a ball at its half-split level has violated
    nothing, so growth never stops below it)."""
    eps = np.full((M, N + 1), CAP)
    for k in range(N + 1):
        S = np.full((M, M), np.inf)
        for j in range(M):
            for l in range(j + 1, M):
                S[j, l] = S[l, j] = (np.linalg.norm(x[j][k] - x[l][k])
                                     - deltas[j][l])
        r = np.zeros(M)
        active = np.ones(M, bool)
        while active.any():
            # next freeze: min over constraints involving an active ball
            t_best, who = np.inf, None
            for j in np.nonzero(active)[0]:
                tj = CAP - r[j]
                for l in range(M):
                    if l == j:
                        continue
                    slack = S[j, l] - r[j] - r[l]
                    tj = min(tj, slack / 2 if active[l] else slack)
                if tj < t_best:
                    t_best, who = tj, j
            r[active] += max(0.0, t_best)
            active[who] = False
        eps[:, k] = np.minimum(r, CAP)
    return eps


def tubes_detail(sc, psis, pairs, deltas, x, N, M, alloc="half"):
    """Decoupled tube SOCPs. Returns (fuel, rs, t_each) or (None, j, None)."""
    fuel, rs, t_each = 0.0, [], []
    wf = waterfill_radii(x, deltas, N, M) if alloc == "maximal" else None
    for j in range(M):
        if wf is not None:
            e = wf[j]
        else:
            e = np.full(N + 1, CAP)
            for l in range(M):
                if l != j:
                    e = np.minimum(e, 0.5 * (np.linalg.norm(x[j] - x[l], axis=1)
                                             - deltas[j][l]))
        t0 = time.perf_counter()
        r = cp.Variable((N + 1, 3))
        v = cp.Variable((N + 1, 3))
        F = cp.Variable((N, 3))
        cons = r2._dyn(sc, psis, r, v, F, j)
        for k in range(1, N):
            if e[k] < CAP:
                cons.append(cp.norm(r[k] - x[j][k]) <= e[k])
        prob = cp.Problem(cp.Minimize(cp.sum(cp.norm(F, 2, axis=1))), cons)
        try:
            prob.solve(solver="CLARABEL")
        except cp.error.SolverError:
            return None, j, None
        t_each.append(time.perf_counter() - t0)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            return None, j, None
        fuel += float(prob.value)
        rs.append(np.asarray(r.value))
    return fuel, rs, t_each


def run_instance(M, seed, N, tf, layout="random", profile="gauss",
                 alloc="half"):
    ring = 500.0 * max(1.0, M / 20.0) ** (1.0 / 3.0)
    if layout == "random_seq":
        # dart-throwing placement: exact constant-density ring, no auto-grow
        sc = scenario.generate(layout, M=M, n_points=8, n_agents=3 * M,
                               seed=seed, nu_deg=60, N=N, tf=tf,
                               ring_radius=ring, max_tries=5000)
    else:
        for _ in range(8):  # all-pairs rejection sampling thins out at large M
            try:
                sc = scenario.generate(layout, M=M, n_points=8, n_agents=3 * M,
                                       seed=seed, nu_deg=60, N=N, tf=tf,
                                       ring_radius=ring, max_tries=5000)
                break
            except RuntimeError:
                ring *= 1.15
        else:
            return dict(M=M, seed=seed, status="placement failed")
    psis = translation.psi_list(sc)

    t0 = time.perf_counter()
    refs = translation.solve_all(sc, psis=psis)
    t_nom = time.perf_counter() - t0
    if not all(r["ok"] for r in refs):
        return dict(M=M, seed=seed, status="nominal failed")
    LB = sum(r["fuel"] for r in refs)
    rbar = [np.asarray(refs[j]["r"]) for j in range(M)]

    ext = [float(np.max(np.linalg.norm(P.points, axis=1))) for P in sc.payloads]
    margin = sc.delta - 2 * max(ext)
    deltas = [[ext[j] + ext[l] + margin for l in range(M)] for j in range(M)]
    pairs = [(j, l) for j in range(M) for l in range(j + 1, M)]
    n_conf = sum(min(np.linalg.norm(rbar[j] - rbar[l], axis=1))
                 < deltas[j][l] for (j, l) in pairs)
    row = dict(M=M, seed=seed, ring_radius=round(ring, 0), n_pairs=len(pairs),
               n_conflicts=int(n_conf), LB=round(LB, 2),
               t_nominals=round(t_nom, 2))

    row["profile"] = profile
    for headroom in (60.0, 120.0):
        tgt = {(j, l): deltas[j][l] + headroom for (j, l) in pairs}
        t0 = time.perf_counter()
        x, sweeps, resid = relax_iso(rbar, pairs, tgt, N, M, profile=profile)
        t_relax = time.perf_counter() - t0
        if x is None:
            row["status"] = f"relax stalled (resid {resid:.2f})"
            return row
        t0 = time.perf_counter()
        fuel, rs, t_each = tubes_detail(sc, psis, pairs, deltas, x, N, M,
                                        alloc=alloc)
        t_tubes = time.perf_counter() - t0
        if fuel is not None:
            break
    row.update(headroom=headroom, t_relax=round(t_relax, 2),
               relax_sweeps=sweeps, t_tubes=round(t_tubes, 2))
    if fuel is None:
        row["status"] = f"tube infeasible for payload {rs}"
        return row

    mm = min(float(np.min(np.linalg.norm(rs[j] - rs[l], axis=1)))
             - deltas[j][l] for (j, l) in pairs)
    row.update(t_tube_max=round(max(t_each), 3),
               J_tubes=round(fuel, 2),
               gap_pct=round(100 * (fuel - LB) / LB, 2),
               min_margin=round(mm, 1), status="ok")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=int, nargs="+",
                    default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--tf", type=float, default=500.0)
    ap.add_argument("--layout", default="random")
    ap.add_argument("--profile", default="gauss")
    ap.add_argument("--alloc", default="half",
                    help="tube radii: 'half' (half-split) or 'maximal' "
                         "(uniform growth to contact, water-filling)")
    ap.add_argument("--out", default="benchmark_tubes.json")
    a = ap.parse_args()
    rows = []
    for M in a.Ms:
        for s in a.seeds:
            try:
                row = run_instance(M, s, a.N, a.tf, layout=a.layout,
                                   profile=a.profile, alloc=a.alloc)
            except Exception as exc:
                row = dict(M=M, seed=s, status=f"error: {str(exc)[:100]}")
            rows.append(row)
            print(json.dumps(row), flush=True)
            p = Path(__file__).with_name("results") / a.out
            p.write_text(json.dumps(rows, indent=1))
    print("done", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
